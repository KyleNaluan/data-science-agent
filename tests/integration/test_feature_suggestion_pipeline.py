"""
Integration tests for feature engineering suggestions (issue #9).
All tests use FakeLLMClient — no real API calls.
"""
import json
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FEAT_CSV = str(FIXTURES_DIR / "feature_suggestion_data.csv")

# Schema → distribution → feature_suggestion → stop
_FEAT_RESPONSES = [
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
            {"type": "text", "text": "Computing distributions."},
            {"type": "tool_use", "id": "tu_002", "name": "distribution_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Generating feature engineering suggestions."},
            {"type": "tool_use", "id": "tu_003", "name": "feature_suggestion", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


def _make_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_FEAT_RESPONSES)


class TestFeatureEngineeringSectionPopulated:
    def test_section_not_placeholder(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        feat_section = content.split("## Feature Engineering Recommendations")[1]
        assert "_Not yet analyzed in this run._" not in feat_section

    def test_section_contains_suggestion_for_price(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        feat_section = content.split("## Feature Engineering Recommendations")[1]
        assert "price" in feat_section

    def test_section_contains_suggestion_for_category(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        feat_section = content.split("## Feature Engineering Recommendations")[1]
        assert "category" in feat_section

    def test_section_cites_specific_skew_value(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        feat_section = content.split("## Feature Engineering Recommendations")[1]
        assert "skew=" in feat_section

    def test_section_cites_unique_count_for_categorical(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        feat_section = content.split("## Feature Engineering Recommendations")[1]
        assert "20" in feat_section

    def test_section_cites_correlation_value(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        feat_section = content.split("## Feature Engineering Recommendations")[1]
        assert "r=" in feat_section

    def test_all_five_sections_present(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
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


class TestAggregateOnly:
    def test_trace_has_no_raw_rows(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        feat_entries = [
            json.loads(l) for l in lines
            if json.loads(l).get("tool_name") == "feature_suggestion"
        ]
        assert len(feat_entries) == 1
        output = feat_entries[0]["tool_output"]
        assert "row_values" not in output
        assert "raw_data" not in output
        for s in output.get("suggestions", []):
            assert "row_values" not in s

    def test_suggestions_output_references_no_individual_rows(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        feat_entries = [
            json.loads(l) for l in lines
            if json.loads(l).get("tool_name") == "feature_suggestion"
        ]
        output = feat_entries[0]["tool_output"]
        # Each suggestion should only list column names, not row-level values
        for s in output.get("suggestions", []):
            assert isinstance(s["columns"], list)
            assert all(isinstance(c, str) for c in s["columns"])


class TestNoTransformationApplied:
    def test_dataframe_unchanged_after_feature_suggestion(self, tmp_path):
        import pandas as pd
        original_df = pd.read_csv(FEAT_CSV)
        original_shape = original_df.shape
        original_columns = list(original_df.columns)

        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        # Re-read the source CSV — it should be unchanged
        reloaded_df = pd.read_csv(FEAT_CSV)
        assert list(reloaded_df.columns) == original_columns
        assert reloaded_df.shape == original_shape
