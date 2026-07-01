"""Unit tests for MissingValueTool (issue #4)."""
import pandas as pd
import pytest

from ds_agent.tools.missing_value import MissingValueTool, _imputation_recommendation


def _df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestMissingValueToolOutput:
    def test_returns_dict_with_columns_key(self):
        df = _df(a=[1, 2, None])
        result = MissingValueTool().run(df)
        assert "columns" in result

    def test_returns_row_count(self):
        df = _df(a=[1, 2, 3, 4, 5])
        result = MissingValueTool().run(df)
        assert result["row_count"] == 5

    def test_counts_missing_correctly(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        result = MissingValueTool().run(df)
        col = result["columns"][0]
        assert col["missing_count"] == 2
        assert col["missing_rate"] == pytest.approx(0.4, abs=1e-4)

    def test_zero_missing(self):
        df = _df(a=[1, 2, 3])
        result = MissingValueTool().run(df)
        col = result["columns"][0]
        assert col["missing_count"] == 0
        assert col["missing_rate"] == 0.0

    def test_all_missing(self):
        df = pd.DataFrame({"a": [None, None, None]})
        result = MissingValueTool().run(df)
        col = result["columns"][0]
        assert col["missing_count"] == 3
        assert col["missing_rate"] == pytest.approx(1.0, abs=1e-4)

    def test_overall_missing_rate_computed(self):
        # 1 missing out of 6 cells (2 rows × 3 cols)
        df = pd.DataFrame({"a": [1, None], "b": [3, 4], "c": [5, 6]})
        result = MissingValueTool().run(df)
        assert result["total_missing_cells"] == 1
        assert result["overall_missing_rate"] == pytest.approx(1 / 6, abs=1e-3)

    def test_multiple_columns_all_reported(self):
        df = _df(a=[1, 2, 3], b=["x", None, "z"])
        result = MissingValueTool().run(df)
        names = [c["column"] for c in result["columns"]]
        assert "a" in names
        assert "b" in names

    def test_dtype_field_present(self):
        df = _df(a=[1.0, 2.0, None])
        result = MissingValueTool().run(df)
        assert "dtype" in result["columns"][0]


class TestAggregateOnlyOutput:
    def test_no_raw_row_values_in_output(self):
        df = _df(value=[10, 20, None, 40, 50])
        result = MissingValueTool().run(df)
        col = result["columns"][0]
        assert "row_values" not in col
        assert "raw_data" not in col
        assert "samples" not in col

    def test_only_aggregate_stats_present(self):
        df = _df(x=[1, None, 3])
        result = MissingValueTool().run(df)
        col = result["columns"][0]
        expected_keys = {"column", "missing_count", "missing_rate", "dtype", "imputation_recommendation"}
        assert expected_keys.issubset(col.keys())


class TestImputationRecommendations:
    def test_no_missing_returns_no_missing(self):
        rec = _imputation_recommendation("f", 0.0, None)
        assert "No missing values" in rec

    def test_critically_high_missingness(self):
        rec = _imputation_recommendation("f", 0.95, None)
        assert "drop" in rec.lower() or "critically" in rec.lower()

    def test_high_missingness(self):
        rec = _imputation_recommendation("f", 0.6, None)
        assert "50%" in rec or "high" in rec.lower()

    def test_numeric_symmetric_gets_mean_median(self):
        rec = _imputation_recommendation("f", 0.1, 0.3)
        assert "mean" in rec.lower() or "median" in rec.lower()

    def test_numeric_skewed_gets_median(self):
        rec = _imputation_recommendation("f", 0.1, 2.5)
        assert "median" in rec.lower()

    def test_categorical_gets_mode_or_unknown(self):
        rec = _imputation_recommendation("O", 0.1, None)
        assert "mode" in rec.lower() or "unknown" in rec.lower()

    def test_datetime_gets_forward_fill(self):
        rec = _imputation_recommendation("M", 0.05, None)
        assert "forward" in rec.lower() or "interpolat" in rec.lower()

    def test_boolean_gets_mode_or_indicator(self):
        rec = _imputation_recommendation("b", 0.05, None)
        assert "mode" in rec.lower() or "indicator" in rec.lower()

    def test_imputation_recommendation_in_tool_output(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = MissingValueTool().run(df)
        col = result["columns"][0]
        assert "imputation_recommendation" in col
        assert len(col["imputation_recommendation"]) > 0


class TestThresholdBoundary:
    def test_missing_data_fixture_above_threshold(self):
        # age column has 5/20 = 25% missing — above the default 20% threshold
        import pandas as pd
        from pathlib import Path
        fixture = Path(__file__).parent.parent / "fixtures" / "missing_data.csv"
        df = pd.read_csv(fixture)
        result = MissingValueTool().run(df)
        age_col = next(c for c in result["columns"] if c["column"] == "age")
        assert age_col["missing_rate"] > 0.20

    def test_salary_column_below_threshold(self):
        from pathlib import Path
        fixture = Path(__file__).parent.parent / "fixtures" / "missing_data.csv"
        df = pd.read_csv(fixture)
        result = MissingValueTool().run(df)
        salary_col = next(c for c in result["columns"] if c["column"] == "salary")
        assert salary_col["missing_rate"] <= 0.20
