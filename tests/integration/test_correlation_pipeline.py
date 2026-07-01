"""
Integration tests for correlation analysis (issue #6).
All tests use FakeLLMClient — no real API calls.
"""
import json
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CORR_CSV = str(FIXTURES_DIR / "correlated_data.csv")

_CORR_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running schema inference."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running correlation analysis."},
            {"type": "tool_use", "id": "tu_002", "name": "correlation_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


def _make_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_CORR_RESPONSES)


class TestCorrelationChartGeneration:
    def test_heatmap_png_created(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        assert (tmp_path / "charts" / "correlation_heatmap.png").exists()

    def test_heatmap_referenced_in_report(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "correlation_heatmap.png" in content


class TestCorrelationSection:
    def test_correlations_section_populated(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        corr_section = content.split("## Correlations")[1].split("##")[0]
        assert "_Not yet analyzed in this run._" not in corr_section

    def test_standout_pairs_named_in_narrative(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        corr_section = content.split("## Correlations")[1].split("## Feature Engineering")[0]
        # correlated_data.csv has a & b perfectly correlated, a & c strongly negative
        assert "a" in corr_section and "b" in corr_section

    def test_full_matrix_table_present(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Full Correlation Matrix" in content

    def test_correlation_values_in_report(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        corr_section = content.split("## Correlations")[1].split("## Feature Engineering")[0]
        assert "1.000" in corr_section  # diagonal

    def test_all_five_sections_present(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
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


class TestAggregateOnlyCorrelation:
    def test_trace_has_no_raw_rows(self, tmp_path):
        run_agent(
            csv_path=CORR_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        corr_entries = [
            json.loads(l) for l in lines
            if json.loads(l).get("tool_name") == "correlation_analysis"
        ]
        assert len(corr_entries) == 1
        output = corr_entries[0]["tool_output"]
        assert "row_values" not in output
        assert "raw_data" not in output
        for pair in output.get("ranked_pairs", []):
            assert "row_values" not in pair
