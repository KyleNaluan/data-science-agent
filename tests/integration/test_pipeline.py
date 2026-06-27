import json
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SIMPLE_CSV = str(FIXTURES_DIR / "simple.csv")

_SCHEMA_INFERENCE_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I will run schema inference to understand the dataset structure."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Schema analysis is complete. I have all the information needed."},
        ],
    },
]


@pytest.fixture
def fake_llm():
    return FakeLLMClient(responses=_SCHEMA_INFERENCE_RESPONSES)


class TestFullPipelineSmoke:
    def test_report_file_created(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        assert (tmp_path / "report.md").exists()

    def test_trace_file_created(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        assert (tmp_path / "trace.jsonl").exists()

    def test_report_has_all_five_sections(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        for section in [
            "## Executive Summary",
            "## Data Quality Scorecard",
            "## Distributions",
            "## Correlations",
            "## Feature Engineering Recommendations",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_data_quality_scorecard_has_column_names(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        for col in ["id", "age", "salary", "department", "hire_date"]:
            assert col in content, f"Column {col!r} not found in report"

    def test_placeholder_sections_present(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "_Not yet analyzed in this run._" in content


class TestTraceLog:
    def test_one_entry_per_tool_call(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_trace_entry_structure(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["tool_name"] == "schema_inference"
        assert "reasoning" in entry
        assert "tool_args" in entry
        assert "tool_output" in entry
        assert entry["step"] == 1

    def test_trace_tool_output_is_aggregate_only(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        output = entry["tool_output"]

        assert "row_count" in output
        assert "columns" in output
        for col in output["columns"]:
            # Must not contain raw row data
            assert "row_values" not in col
            assert "raw_data" not in col
            assert "samples" not in col

    def test_trace_reasoning_captured(self, tmp_path, fake_llm):
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake_llm)
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[0])
        assert "schema inference" in entry["reasoning"].lower()


class TestFakeLLMUsedExclusively:
    def test_llm_called_exactly_twice(self, tmp_path):
        fake = FakeLLMClient(responses=_SCHEMA_INFERENCE_RESPONSES)
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=fake)
        assert fake.call_count == 2

    def test_no_tool_call_path_produces_report(self, tmp_path):
        """If LLM skips tools entirely, report still has all 5 sections."""
        no_tool_llm = FakeLLMClient(responses=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "No tools needed."}],
            }
        ])
        run_agent(csv_path=SIMPLE_CSV, output_dir=str(tmp_path), llm_client=no_tool_llm)
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        for section in [
            "## Executive Summary",
            "## Data Quality Scorecard",
            "## Distributions",
            "## Correlations",
            "## Feature Engineering Recommendations",
        ]:
            assert section in content
