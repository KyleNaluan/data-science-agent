from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import typer

from .agent.graph import build_graph
from .llm.base import LLMClient
from .tools.schema_inference import SchemaInferenceTool

app = typer.Typer(help="Data science agent -- EDA + report generation.")

_DEFAULT_TOOLS = [SchemaInferenceTool()]


def _build_llm_client(provider: str | None) -> LLMClient:
    """Instantiate the correct LLM client.

    If provider is None, auto-select based on which API key is present in the
    environment (.env is already loaded by this point):
      GROQ_API_KEY present  -> GroqClient
      ANTHROPIC_API_KEY present -> AnthropicClient
    Raises a clear error if neither key is set.
    """
    resolved = provider or (
        "groq" if os.environ.get("GROQ_API_KEY") else
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
        None
    )

    if resolved == "groq":
        from .llm.groq import GroqClient
        return GroqClient()
    elif resolved == "anthropic":
        from .llm.anthropic import AnthropicClient
        return AnthropicClient()
    else:
        raise RuntimeError(
            "No LLM provider configured. Set GROQ_API_KEY or ANTHROPIC_API_KEY in your .env file, "
            "or pass --provider explicitly."
        )


def run_agent(csv_path: str, output_dir: str, llm_client: LLMClient) -> None:
    """
    Run the agent pipeline end-to-end.

    Accepts an injected LLM client so callers (tests, CLI, future service wrappers)
    can substitute any conforming implementation without touching the graph.
    """
    compiled = build_graph(llm_client=llm_client, tools=_DEFAULT_TOOLS)
    initial_state = {
        "csv_path": csv_path,
        "output_dir": output_dir,
        "df": None,
        "messages": [],
        "tool_results": {},
        "trace_entries": [],
        "report": "",
    }
    compiled.invoke(initial_state)


@app.command()
def analyze(
    csv_path: Path = typer.Argument(..., help="Path to the CSV file to analyze."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to write report and trace. Defaults to a timestamped subdirectory next to the CSV.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider: 'anthropic' or 'groq'. Auto-detected from .env if omitted.",
    ),
) -> None:
    if not csv_path.exists():
        typer.echo(f"Error: file not found: {csv_path}", err=True)
        raise typer.Exit(1)

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = csv_path.parent / f"ds_agent_run_{stamp}"

    try:
        llm_client = _build_llm_client(provider)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    provider_name = llm_client.__class__.__name__
    typer.echo(f"Provider: {provider_name}")
    typer.echo(f"Analyzing {csv_path} -> {output_dir}")

    try:
        run_agent(csv_path=str(csv_path), output_dir=str(output_dir), llm_client=llm_client)
    except Exception as exc:
        typer.echo(f"Agent run failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Report written to {output_dir}/report.md")
    typer.echo(f"Trace written to  {output_dir}/trace.jsonl")


def main() -> None:
    typer.run(analyze)
