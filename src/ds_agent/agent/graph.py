from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from ..llm.base import LLMClient
from ..report.renderer import render_report
from ..tools.base import Tool
from .state import AgentState

_SYSTEM_PROMPT = """\
You are a data science agent conducting exploratory data analysis on a CSV file.
You have access to tools that compute aggregate statistics over the dataset.
IMPORTANT: You must never request or reference individual rows of data.
Call the available tools to gather statistics about the dataset.
When you have gathered sufficient information, stop calling tools and respond with a plain text message.
"""


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


def build_graph(llm_client: LLMClient, tools: list[Tool]) -> Any:
    """
    Build and compile the LangGraph state machine.

    The graph implements a Reason → Act → Observe loop:
      ingest → llm_call → tool_dispatch → llm_call → ... → report_assembly → deliver → END

    The llm_client is injected so tests can swap in a FakeLLMClient.
    """
    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    tool_defs = [t.to_anthropic_def() for t in tools]

    def ingest(state: AgentState) -> dict:
        df = pd.read_csv(state["csv_path"])
        initial_message = _build_initial_user_message(df, state["csv_path"])
        return {
            "df": df,
            "messages": [initial_message],
            "tool_results": {},
            "trace_entries": [],
            "report": "",
        }

    def llm_call(state: AgentState) -> dict:
        response = llm_client.complete(
            system=_SYSTEM_PROMPT,
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

    def report_assembly(state: AgentState) -> dict:
        report = render_report(
            tool_results=state["tool_results"],
            metadata={"csv_path": state["csv_path"]},
        )
        return {"report": report}

    def deliver(state: AgentState) -> dict:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

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
    graph.add_node("report_assembly", report_assembly)
    graph.add_node("deliver", deliver)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "llm_call")
    graph.add_conditional_edges("llm_call", route_after_llm, {
        "tool_dispatch": "tool_dispatch",
        "report_assembly": "report_assembly",
    })
    graph.add_edge("tool_dispatch", "llm_call")
    graph.add_edge("report_assembly", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()
