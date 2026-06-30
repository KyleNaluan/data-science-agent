from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from ..llm.base import LLMClient
from ..report.renderer import render_report
from ..tools.base import Tool
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


def build_graph(
    llm_client: LLMClient,
    tools: list[Tool],
    interactive: bool | None = None,
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
    """
    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    tool_defs = [t.to_anthropic_def() for t in tools]
    system_prompt = _build_system_prompt(tools)

    def ingest(state: AgentState) -> dict:
        df = pd.read_csv(state["csv_path"])
        initial_message = _build_initial_user_message(df, state["csv_path"])
        return {
            "df": df,
            "messages": [initial_message],
            "tool_results": {},
            "trace_entries": [],
            "report": "",
            "flagged_assumptions": [],
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

        schema = state["tool_results"].get("schema_inference")
        if not schema:
            return {"flagged_assumptions": []}

        existing_contexts = {
            a.get("context", {}).get("column") or a.get("context", {}).get("trigger_id")
            for a in state["flagged_assumptions"]
        }

        new_flagged: list[dict] = []
        type_overrides: dict[str, str] = {}

        row_count = schema.get("row_count", 0)
        tiny_id = "__tiny_dataset__"
        if row_count < TINY_DATASET_THRESHOLD and tiny_id not in existing_contexts:
            trigger = UncertaintyTrigger(
                trigger_type="tiny_dataset",
                question=(
                    f"Dataset has only {row_count} rows "
                    f"(minimum recommended: {TINY_DATASET_THRESHOLD}). "
                    "Statistical findings may be unreliable. Proceed with defaults?"
                ),
                default="proceed",
                context={"trigger_id": tiny_id, "row_count": row_count},
            )
            _, assumption = handle_uncertainty(trigger, interactive=_interactive)
            if assumption:
                new_flagged.append(assumption.to_dict())

        for col in schema.get("columns", []):
            if not col.get("is_ambiguous"):
                continue
            col_name = col["name"]
            if col_name in existing_contexts:
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

        updated_tool_results = state["tool_results"]
        if type_overrides:
            updated_cols = [
                {**col, "inferred_type": type_overrides[col["name"]], "is_ambiguous": False}
                if col["name"] in type_overrides else col
                for col in schema.get("columns", [])
            ]
            updated_schema = {**schema, "columns": updated_cols}
            updated_tool_results = {**state["tool_results"], "schema_inference": updated_schema}

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

        dist_data = state["tool_results"].get("distribution_analysis")
        if dist_data and dist_data.get("columns"):
            charts_dir = out / "charts"
            charts_dir.mkdir(exist_ok=True)
            _generate_histograms(dist_data["columns"], charts_dir)

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
