from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

import typer

from .agent.graph import build_graph
from .config import load_thresholds
from .llm.base import LLMClient
from .tools.correlation import CorrelationTool
from .tools.distribution import DistributionTool
from .tools.feature_suggestion import FeatureSuggestionTool
from .tools.missing_value import MissingValueTool
from .tools.outlier import OutlierTool
from .tools.schema_file_parser import parse_schema_file
from .tools.schema_inference import SchemaInferenceTool

app = typer.Typer(help="Data science agent -- EDA + report generation.")

_DEFAULT_TOOLS = [
    SchemaInferenceTool(),
    MissingValueTool(),
    DistributionTool(),
    OutlierTool(),
    CorrelationTool(),
    FeatureSuggestionTool(),
]


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


def run_agent(
    csv_path: str,
    output_dir: str,
    llm_client: LLMClient,
    interactive: bool | None = None,
    missing_threshold: float | None = None,
    schema_hints: dict[str, str] | None = None,
    extra_csv_paths: list[str] | None = None,
) -> None:
    """
    Run the agent pipeline end-to-end.

    Accepts an injected LLM client so callers (tests, CLI, future service wrappers)
    can substitute any conforming implementation without touching the graph.

    interactive controls uncertainty checkpoints:
      None  → auto-detect via sys.stdin.isatty()
      True  → always prompt the user
      False → always apply defaults and record FlaggedAssumptions

    missing_threshold overrides the missing-value-rate threshold from config/defaults.
    schema_hints: pre-parsed column→type mapping from a schema file (issue #7).
    extra_csv_paths: additional CSV paths to join with primary (issue #8).
    """
    thresholds = load_thresholds(
        **({} if missing_threshold is None else {"missing_value_rate": missing_threshold})
    )
    compiled = build_graph(
        llm_client=llm_client,
        tools=_DEFAULT_TOOLS,
        interactive=interactive,
        thresholds=thresholds,
        schema_hints=schema_hints,
    )
    initial_state = {
        "csv_path": csv_path,
        "output_dir": output_dir,
        "df": None,
        "messages": [],
        "tool_results": {},
        "trace_entries": [],
        "report": "",
        "flagged_assumptions": [],
        "extra_csv_paths": extra_csv_paths or [],
    }
    compiled.invoke(initial_state)


@app.command()
def analyze(
    csv_path: Path = typer.Argument(..., help="Path to the primary CSV file to analyze."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory to write report and trace. Defaults to a timestamped subdirectory next to the CSV.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider: 'anthropic' or 'groq'. Auto-detected from .env if omitted.",
    ),
    missing_threshold: Optional[float] = typer.Option(
        None,
        "--missing-threshold",
        help=(
            "Missing-value rate threshold (0.0–1.0) above which an uncertainty checkpoint fires. "
            "Overrides the value in ds_agent.toml for this run only. Default: 0.20."
        ),
    ),
    schema_file: Optional[Path] = typer.Option(
        None,
        "--schema-file",
        help=(
            "Optional JSON schema file specifying expected column types. "
            "Format: {\"columns\": {\"col_name\": \"type\", ...}}. "
            "Conflicts with inferred types trigger an uncertainty checkpoint."
        ),
    ),
    extra_csv: Optional[List[Path]] = typer.Option(
        None,
        "--extra-csv",
        help=(
            "Additional CSV file(s) to join with the primary CSV. "
            "Can be specified multiple times. Join keys are inferred automatically."
        ),
    ),
) -> None:
    if not csv_path.exists():
        typer.echo(f"Error: file not found: {csv_path}", err=True)
        raise typer.Exit(1)

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = csv_path.parent / f"ds_agent_run_{stamp}"

    schema_hints: dict[str, str] | None = None
    if schema_file is not None:
        if not schema_file.exists():
            typer.echo(f"Error: schema file not found: {schema_file}", err=True)
            raise typer.Exit(1)
        try:
            schema_hints = parse_schema_file(schema_file)
        except (ValueError, OSError) as exc:
            typer.echo(f"Error reading schema file: {exc}", err=True)
            raise typer.Exit(1)

    extra_csv_paths = [str(p) for p in (extra_csv or [])]

    try:
        llm_client = _build_llm_client(provider)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    provider_name = llm_client.__class__.__name__
    typer.echo(f"Provider: {provider_name}")
    typer.echo(f"Analyzing {csv_path} -> {output_dir}")

    try:
        run_agent(
            csv_path=str(csv_path),
            output_dir=str(output_dir),
            llm_client=llm_client,
            missing_threshold=missing_threshold,
            schema_hints=schema_hints,
            extra_csv_paths=extra_csv_paths,
        )
    except Exception as exc:
        typer.echo(f"Agent run failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Report written to {output_dir}/report.md")
    typer.echo(f"Trace written to  {output_dir}/trace.jsonl")


def main() -> None:
    typer.run(analyze)
