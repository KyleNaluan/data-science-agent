from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from ..config import ThresholdConfig, load_thresholds
from ..llm.base import LLMClient
from ..report.renderer import render_report
from ..tools.base import Tool
from ..tools.join_key import JoinKeyInferenceTool
from ..uncertainty import (
    TINY_DATASET_THRESHOLD,
    UncertaintyTrigger,
    handle_uncertainty,
    is_interactive,
)
from .state import AgentState

_SYSTEM_PROMPT_BASE = """\
You are a data science agent conducting exploratory data analysis on a CSV file.
You have access to tools that compute aggregate statistics over the dataset.
IMPORTANT: You must never request or reference individual rows of data.

Follow this analysis sequence:
{steps}
{stop_step}. Once all tools have run, stop calling tools and respond with a plain text message.
"""


def _build_system_prompt(tools: list[Tool]) -> str:
    # Auto-generated from the registered tool list — do not hand-edit the sequence here.
    # To change the order or add a step, update _DEFAULT_TOOLS in cli.py instead.
    steps = "\n".join(
        f"{i + 1}. Call {t.name} — {t.description.splitlines()[0]}"
        for i, t in enumerate(tools)
    )
    return _SYSTEM_PROMPT_BASE.format(steps=steps, stop_step=len(tools) + 1)


def _col_summary(df: pd.DataFrame) -> str:
    parts = []
    for col in df.columns:
        parts.append(f"  - {col} ({df[col].dtype})")
    return "\n".join(parts)


def _build_initial_user_message(df: pd.DataFrame, csv_path: str) -> dict:
    body = (
        f"Analyze the following CSV dataset.\n\n"
        f"File: {csv_path}\n"
        f"Shape: {len(df)} rows × {len(df.columns)} columns\n"
        f"Columns:\n{_col_summary(df)}"
    )
    return {"role": "user", "content": body}


def _extract_tool_use_blocks(message: dict) -> list[dict]:
    content = message.get("content") or []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _extract_reasoning_text(message: dict) -> str:
    content = message.get("content") or []
    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return " ".join(texts).strip()


def _safe_filename(col_name: str) -> str:
    return re.sub(r"[^\w\-]", "_", col_name)


# ---------------------------------------------------------------------------
# Chart generation helpers (called from the deliver node)
# ---------------------------------------------------------------------------

def _generate_histograms(dist_columns: list[dict], charts_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col in dist_columns:
        counts = col["bin_counts"]
        edges = col["bin_edges"]
        if not counts or not edges:
            continue

        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            edges[:-1], counts, width=widths, align="edge",
            color="steelblue", edgecolor="white", linewidth=0.5,
        )
        ax.set_title(f"Distribution of {col['column']}")
        ax.set_xlabel(col["column"])
        ax.set_ylabel("Count")
        ax.text(
            0.98, 0.97,
            f"skew={col.get('skew', 0):.2f}, kurt={col.get('kurtosis', 0):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray",
        )
        fig.tight_layout()
        safe = _safe_filename(col["column"])
        fig.savefig(charts_dir / f"{safe}.png", dpi=96)
        plt.close(fig)


def _generate_outlier_plots(
    outlier_columns: list[dict],
    df: pd.DataFrame,
    charts_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col in outlier_columns:
        col_name = col["column"]
        if col_name not in df.columns:
            continue

        series = df[col_name].dropna()
        if len(series) == 0:
            continue

        iqr_indices = set(col["iqr_method"]["outlier_indices"])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(series.values, orientation="horizontal", widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6),
                   medianprops=dict(color="white", linewidth=2))

        if iqr_indices:
            outlier_vals = series[series.index.isin(iqr_indices)].values
            ax.scatter(outlier_vals, [1] * len(outlier_vals),
                       color="crimson", s=50, zorder=5,
                       label=f"IQR outliers (n={len(outlier_vals)})")
            ax.legend(fontsize=8)

        ax.set_title(f"Outliers: {col_name}")
        ax.set_xlabel(col_name)
        ax.set_yticks([])
        fig.tight_layout()
        safe = _safe_filename(col_name)
        fig.savefig(charts_dir / f"{safe}_outliers.png", dpi=96)
        plt.close(fig)


def _generate_correlation_heatmap(corr_data: dict, charts_dir: Path) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = corr_data.get("matrix", {})
    columns = corr_data.get("columns", [])
    if not columns or len(columns) < 2:
        return

    n = len(columns)
    arr = np.zeros((n, n))
    for i, col_a in enumerate(columns):
        for j, col_b in enumerate(columns):
            v = matrix.get(col_a, {}).get(col_b, 0.0)
            arr[i, j] = v if math.isfinite(v) else 0.0

    fig_size = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(columns, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(arr[i, j]) < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap (Pearson r)")
    fig.tight_layout()
    fig.savefig(charts_dir / "correlation_heatmap.png", dpi=96)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    llm_client: LLMClient,
    tools: list[Tool],
    interactive: bool | None = None,
    thresholds: ThresholdConfig | None = None,
    schema_hints: dict[str, str] | None = None,
) -> Any:
    """
    Build and compile the LangGraph state machine.

    Graph flow:
      ingest → llm_call → tool_dispatch → uncertainty_check → llm_call → ...
                        ↘ report_assembly → deliver → END

    interactive controls the uncertainty checkpoint behaviour:
      None  → auto-detect via sys.stdin.isatty() at runtime
      True  → always prompt the user
      False → always apply defaults and record FlaggedAssumptions (test mode)

    thresholds: ThresholdConfig controlling when uncertainty triggers fire.
      None → load from ds_agent.toml / built-in defaults.

    schema_hints: pre-parsed column→type mapping from an optional schema file.
      None → no schema hints applied.
    """
    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    tool_defs = [t.to_anthropic_def() for t in tools]
    system_prompt = _build_system_prompt(tools)
    _thresholds = thresholds if thresholds is not None else load_thresholds()
    _join_tool = JoinKeyInferenceTool()

    def _assumption_key(a: dict) -> tuple[str, str]:
        ctx = a.get("context", {})
        tt = a.get("trigger_type", "")
        col = ctx.get("column") or ctx.get("trigger_id") or ""
        return (tt, col)

    def ingest(state: AgentState) -> dict:
        df = pd.read_csv(state["csv_path"])
        flagged: list[dict] = []

        extra_paths = state.get("extra_csv_paths") or []
        if extra_paths:
            _interactive = interactive if interactive is not None else is_interactive()
            extra_df = pd.read_csv(extra_paths[0])
            join_result = _join_tool.run(df, extra_df=extra_df)
            confidence = join_result["confidence"]
            candidates = join_result["candidates"]

            if confidence == "high" and join_result["best_candidate"]:
                best = join_result["best_candidate"]
                df = df.merge(
                    extra_df,
                    left_on=best["col_a"],
                    right_on=best["col_b"],
                    how="inner",
                )
            elif candidates:
                best = candidates[0]
                trigger = UncertaintyTrigger(
                    trigger_type="ambiguous_join_key",
                    question=(
                        f"Found {len(candidates)} join key candidate(s). "
                        f"Best: '{best['col_a']}' ↔ '{best['col_b']}' "
                        f"(score={best['score']:.2f}, confidence={confidence}). "
                        "Proceed with best candidate?"
                    ),
                    default="proceed",
                    context={
                        "candidates": [
                            f"{c['col_a']}↔{c['col_b']}" for c in candidates[:3]
                        ],
                        "best_col_a": best["col_a"],
                        "best_col_b": best["col_b"],
                        "confidence": confidence,
                    },
                )
                _, assumption = handle_uncertainty(trigger, interactive=_interactive)
                if assumption:
                    flagged.append(assumption.to_dict())
                # Proceed with best candidate regardless (default = "proceed")
                df = df.merge(
                    extra_df,
                    left_on=best["col_a"],
                    right_on=best["col_b"],
                    how="inner",
                )

        initial_message = _build_initial_user_message(df, state["csv_path"])
        return {
            "df": df,
            "messages": [initial_message],
            "tool_results": {},
            "trace_entries": [],
            "report": "",
            "flagged_assumptions": flagged,
        }

    def llm_call(state: AgentState) -> dict:
        response = llm_client.complete(
            system=system_prompt,
            messages=state["messages"],
            tools=tool_defs,
        )
        return {"messages": [response]}

    def tool_dispatch(state: AgentState) -> dict:
        latest = state["messages"][-1]
        tool_use_blocks = _extract_tool_use_blocks(latest)
        reasoning = _extract_reasoning_text(latest)

        new_tool_results = dict(state["tool_results"])
        new_trace_entries: list[dict] = []
        tool_result_messages: list[dict] = []
        step_base = len(state["trace_entries"])

        for i, block in enumerate(tool_use_blocks):
            tool_name = block["name"]
            tool_args = block.get("input") or {}
            tool_id = block["id"]

            tool = tool_map.get(tool_name)
            if tool is None:
                output = {"error": f"Unknown tool: {tool_name}"}
            else:
                output = tool.run(state["df"], **tool_args)

            new_tool_results[tool_name] = output

            new_trace_entries.append({
                "step": step_base + i + 1,
                "reasoning": reasoning,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_output": output,
            })

            tool_result_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(output, default=str),
                    }
                ],
            })

        return {
            "tool_results": new_tool_results,
            "trace_entries": new_trace_entries,
            "messages": tool_result_messages,
        }

    def uncertainty_check(state: AgentState) -> dict:
        _interactive = interactive if interactive is not None else is_interactive()

        existing_checked = {_assumption_key(a) for a in state["flagged_assumptions"]}
        new_flagged: list[dict] = []
        type_overrides: dict[str, str] = {}

        # --- tiny dataset check ---
        schema = state["tool_results"].get("schema_inference")
        if schema:
            row_count = schema.get("row_count", 0)
            tiny_key = ("tiny_dataset", "__tiny_dataset__")
            if row_count < TINY_DATASET_THRESHOLD and tiny_key not in existing_checked:
                trigger = UncertaintyTrigger(
                    trigger_type="tiny_dataset",
                    question=(
                        f"Dataset has only {row_count} rows "
                        f"(minimum recommended: {TINY_DATASET_THRESHOLD}). "
                        "Statistical findings may be unreliable. Proceed with defaults?"
                    ),
                    default="proceed",
                    context={"trigger_id": "__tiny_dataset__", "row_count": row_count},
                )
                _, assumption = handle_uncertainty(trigger, interactive=_interactive)
                if assumption:
                    new_flagged.append(assumption.to_dict())

            # --- ambiguous column type checks ---
            for col in schema.get("columns", []):
                if not col.get("is_ambiguous"):
                    continue
                col_name = col["name"]
                check_key = ("ambiguous_column_type", col_name)
                if check_key in existing_checked:
                    continue
                trigger = UncertaintyTrigger(
                    trigger_type="ambiguous_column_type",
                    question=(
                        f"Column '{col_name}' has an ambiguous type: "
                        f"{col.get('ambiguity_reason', '')}. "
                        f"Inferred as '{col['inferred_type']}'. "
                        "What is its true type? "
                        "(numeric / categorical / datetime / identifier / boolean / text)"
                    ),
                    default=col["inferred_type"],
                    context={
                        "column": col_name,
                        "ambiguity_reason": col.get("ambiguity_reason", ""),
                    },
                )
                resolved, assumption = handle_uncertainty(trigger, interactive=_interactive)
                if assumption:
                    new_flagged.append(assumption.to_dict())
                else:
                    type_overrides[col_name] = resolved

            # --- conflicting schema hints check (issue #7) ---
            if schema_hints:
                for col in schema.get("columns", []):
                    col_name = col["name"]
                    if col_name not in schema_hints:
                        continue
                    hint_type = schema_hints[col_name]
                    inferred_type = col["inferred_type"]
                    check_key = ("conflicting_schema_hints", col_name)
                    if check_key in existing_checked:
                        continue
                    if hint_type != inferred_type:
                        trigger = UncertaintyTrigger(
                            trigger_type="conflicting_schema_hints",
                            question=(
                                f"Schema file specifies '{col_name}' as '{hint_type}', "
                                f"but inference suggests '{inferred_type}'. "
                                "Which type should be used?"
                            ),
                            default=hint_type,
                            context={
                                "column": col_name,
                                "hint_type": hint_type,
                                "inferred_type": inferred_type,
                            },
                        )
                        resolved, assumption = handle_uncertainty(trigger, interactive=_interactive)
                        if assumption:
                            new_flagged.append(assumption.to_dict())
                        else:
                            type_overrides[col_name] = resolved

        # Apply type overrides to schema in tool_results
        updated_tool_results = state["tool_results"]
        if type_overrides and schema:
            updated_cols = [
                {**col, "inferred_type": type_overrides[col["name"]], "is_ambiguous": False}
                if col["name"] in type_overrides else col
                for col in schema.get("columns", [])
            ]
            updated_schema = {**schema, "columns": updated_cols}
            updated_tool_results = {**state["tool_results"], "schema_inference": updated_schema}

        # --- high missing rate checks (issue #4) ---
        mv_data = state["tool_results"].get("missing_value_analysis")
        if mv_data:
            mv_threshold = _thresholds.missing_value_rate
            for col in mv_data.get("columns", []):
                col_name = col["column"]
                check_key = ("high_missing_rate", col_name)
                if check_key in existing_checked:
                    continue
                if col["missing_rate"] > mv_threshold:
                    trigger = UncertaintyTrigger(
                        trigger_type="high_missing_rate",
                        question=(
                            f"Column '{col_name}' has {col['missing_rate'] * 100:.1f}% missing values "
                            f"(threshold: {mv_threshold * 100:.0f}%). "
                            f"Recommendation: {col['imputation_recommendation']}. "
                            "Proceed with analysis?"
                        ),
                        default="proceed",
                        context={
                            "column": col_name,
                            "missing_rate": col["missing_rate"],
                            "threshold": mv_threshold,
                            "imputation_recommendation": col["imputation_recommendation"],
                        },
                    )
                    _, assumption = handle_uncertainty(trigger, interactive=_interactive)
                    if assumption:
                        new_flagged.append(assumption.to_dict())

        result: dict = {"flagged_assumptions": new_flagged}
        if type_overrides:
            result["tool_results"] = updated_tool_results
        return result

    def report_assembly(state: AgentState) -> dict:
        report = render_report(
            tool_results=state["tool_results"],
            metadata={"csv_path": state["csv_path"]},
            flagged_assumptions=state["flagged_assumptions"],
        )
        return {"report": report}

    def deliver(state: AgentState) -> dict:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        charts_dir = out / "charts"
        needs_charts = (
            state["tool_results"].get("distribution_analysis")
            or state["tool_results"].get("outlier_detection")
            or state["tool_results"].get("correlation_analysis")
        )
        if needs_charts:
            charts_dir.mkdir(exist_ok=True)

        dist_data = state["tool_results"].get("distribution_analysis")
        if dist_data and dist_data.get("columns"):
            _generate_histograms(dist_data["columns"], charts_dir)

        outlier_data = state["tool_results"].get("outlier_detection")
        if outlier_data and outlier_data.get("columns"):
            _generate_outlier_plots(outlier_data["columns"], state["df"], charts_dir)

        corr_data = state["tool_results"].get("correlation_analysis")
        if corr_data and corr_data.get("columns"):
            _generate_correlation_heatmap(corr_data, charts_dir)

        (out / "report.md").write_text(state["report"], encoding="utf-8")

        trace_path = out / "trace.jsonl"
        trace_path.write_text(
            "\n".join(
                __import__("json").dumps(e, default=str)
                for e in state["trace_entries"]
            ) + ("\n" if state["trace_entries"] else ""),
            encoding="utf-8",
        )
        return {}

    def route_after_llm(state: AgentState) -> str:
        latest = state["messages"][-1]
        if _extract_tool_use_blocks(latest):
            return "tool_dispatch"
        return "report_assembly"

    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_dispatch", tool_dispatch)
    graph.add_node("uncertainty_check", uncertainty_check)
    graph.add_node("report_assembly", report_assembly)
    graph.add_node("deliver", deliver)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "llm_call")
    graph.add_conditional_edges("llm_call", route_after_llm, {
        "tool_dispatch": "tool_dispatch",
        "report_assembly": "report_assembly",
    })
    graph.add_edge("tool_dispatch", "uncertainty_check")
    graph.add_edge("uncertainty_check", "llm_call")
    graph.add_edge("report_assembly", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()
