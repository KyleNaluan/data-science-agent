import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.schema_inference import SchemaInferenceTool, infer_schema

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _df(**kwargs: list) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestTypeInference:
    def test_numeric_integer(self):
        df = _df(value=[1, 5, 3, 100, 2, 77, 9, 42, 8, 30, 15, 22])
        result = infer_schema(df)
        col = result.columns[0]
        assert col.inferred_type == "numeric"
        assert not col.is_ambiguous

    def test_numeric_float(self):
        df = _df(salary=[40000.5, 55000.0, 72500.25, 90000.0, 45000.0])
        result = infer_schema(df)
        assert result.columns[0].inferred_type == "numeric"

    def test_categorical_low_cardinality_string(self):
        df = _df(dept=["Eng", "HR", "Sales", "Eng", "HR", "Sales"] * 20)
        result = infer_schema(df)
        assert result.columns[0].inferred_type == "categorical"

    def test_datetime_iso_strings(self):
        df = _df(date=["2024-01-15", "2023-06-30", "2022-12-01", "2021-03-22", "2020-07-04"])
        result = infer_schema(df)
        assert result.columns[0].inferred_type == "datetime"

    def test_datetime_native(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])})
        result = infer_schema(df)
        assert result.columns[0].inferred_type == "datetime"

    def test_boolean_column(self):
        df = _df(flag=[True, False, True, True, False])
        result = infer_schema(df)
        assert result.columns[0].inferred_type == "boolean"

    def test_identifier_sequential_int(self):
        # Sequential IDs should be flagged as ambiguous identifiers
        df = _df(customer_id=list(range(1, 501)))
        result = infer_schema(df)
        col = result.columns[0]
        assert col.inferred_type == "identifier"
        assert col.is_ambiguous
        assert "numeric ID" in (col.ambiguity_reason or "")

    def test_identifier_not_triggered_for_small_int(self):
        # Few unique integers should stay as numeric, not identifier
        df = _df(score=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result = infer_schema(df)
        assert result.columns[0].inferred_type == "numeric"


class TestNullHandling:
    def test_null_rate(self):
        df = _df(x=[1.0, None, 3.0, None, 5.0, 6.0, 7.0, None, 9.0, 10.0])
        result = infer_schema(df)
        col = result.columns[0]
        assert col.null_count == 3
        assert abs(col.null_rate - 0.3) < 0.01

    def test_fully_null_column(self):
        df = pd.DataFrame({"x": pd.Series([None, None, None], dtype="object")})
        result = infer_schema(df)
        col = result.columns[0]
        assert col.null_count == 3
        assert col.null_rate == 1.0


class TestAggregateOnlyOutput:
    def test_numeric_stats_are_aggregates(self):
        df = _df(age=[25, 30, 35, 40, 45])
        result = infer_schema(df)
        col = result.columns[0]
        # Must have aggregate stats
        assert col.mean is not None
        assert col.std is not None
        assert col.min_val is not None
        assert col.max_val is not None
        # Must not have raw row values
        d = asdict(col)
        assert "row_values" not in d
        assert "raw_data" not in d

    def test_categorical_has_value_counts_not_rows(self):
        df = _df(color=["red", "blue", "red", "green", "blue", "red"])
        result = infer_schema(df)
        col = result.columns[0]
        assert col.top_value_counts is not None
        # Value counts are {str: int}, not a list of raw values
        for k, v in col.top_value_counts.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    def test_tool_run_returns_dict(self):
        df = _df(x=[1, 2, 3], y=["a", "b", "a"])
        tool = SchemaInferenceTool()
        output = tool.run(df)
        assert isinstance(output, dict)
        assert "columns" in output
        assert "row_count" in output
        assert output["row_count"] == 3


class TestBenchmarkAccuracy:
    """Load simple.csv and verify ≥5/6 columns get the expected type."""

    EXPECTED_TYPES = {
        "id": "identifier",
        "age": "numeric",
        "salary": "numeric",
        "department": "categorical",
        "hire_date": "datetime",
        "notes": None,  # fully null — accept any type
    }

    def test_type_accuracy(self, simple_csv):
        import pandas as pd
        df = pd.read_csv(simple_csv)
        result = infer_schema(df)
        type_map = {c.name: c.inferred_type for c in result.columns}

        correct = 0
        total = 0
        for col_name, expected in self.EXPECTED_TYPES.items():
            if expected is None:
                continue
            total += 1
            if type_map.get(col_name) == expected:
                correct += 1
            else:
                print(f"  MISMATCH: {col_name} expected={expected} got={type_map.get(col_name)}")

        accuracy = correct / total
        assert accuracy >= 0.95, f"Type inference accuracy {accuracy:.0%} < 95% ({correct}/{total} correct)"

    def test_ambiguous_id_detection(self, ambiguous_id_csv):
        import pandas as pd
        df = pd.read_csv(ambiguous_id_csv)
        result = infer_schema(df)
        type_map = {c.name: c.inferred_type for c in result.columns}
        assert type_map["customer_id"] == "identifier"
        assert type_map["product_id"] == "identifier"
        assert type_map["revenue"] == "numeric"
        assert type_map["region"] == "categorical"
