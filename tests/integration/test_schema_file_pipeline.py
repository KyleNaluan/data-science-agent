"""
Integration tests for optional schema file input (issue #7).
All tests use FakeLLMClient — no real API calls.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient
from ds_agent.tools.schema_file_parser import parse_schema_file

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SIMPLE_CSV = str(FIXTURES_DIR / "simple.csv")

_SCHEMA_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running schema inference."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


def _make_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_SCHEMA_RESPONSES)


class TestAgreingSchemaFile:
    def test_agree_run_completes_without_checkpoint(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_agree.json")
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=hints,
        )
        assert (tmp_path / "report.md").exists()

    def test_agree_run_produces_no_conflict_flagged_assumptions(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_agree.json")
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=hints,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "conflicting_schema_hints" not in content

    def test_no_schema_file_works_identically(self, tmp_path):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=None,
        )
        assert (tmp_path / "report.md").exists()


class TestConflictingSchemaFile:
    def test_conflict_produces_flagged_assumption_non_interactive(self, tmp_path):
        # schema_hint_conflict.json says age=categorical, salary=categorical
        # schema inference should say age=numeric, salary=numeric → conflict
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_conflict.json")
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=hints,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Flagged assumptions" in content

    def test_conflict_names_the_conflicting_column(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_conflict.json")
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=hints,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "age" in content or "salary" in content

    def test_conflict_mentions_hint_type(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_conflict.json")
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=hints,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # schema file says categorical for age/salary
        assert "categorical" in content

    def test_interactive_conflict_prompts_and_completes(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_conflict.json")
        with patch("builtins.input", return_value="categorical"):
            run_agent(
                csv_path=SIMPLE_CSV,
                output_dir=str(tmp_path),
                llm_client=_make_llm(),
                interactive=True,
                schema_hints=hints,
            )
        assert (tmp_path / "report.md").exists()

    def test_interactive_conflict_produces_no_flagged_assumptions(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_conflict.json")
        with patch("builtins.input", return_value="categorical"):
            run_agent(
                csv_path=SIMPLE_CSV,
                output_dir=str(tmp_path),
                llm_client=_make_llm(),
                interactive=True,
                schema_hints=hints,
            )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "conflicting_schema_hints" not in content

    def test_all_five_sections_present(self, tmp_path):
        hints = parse_schema_file(FIXTURES_DIR / "schema_hint_conflict.json")
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            schema_hints=hints,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        for section in [
            "## Executive Summary",
            "## Data Quality Scorecard",
            "## Distributions",
            "## Correlations",
            "## Feature Engineering Recommendations",
        ]:
            assert section in content
