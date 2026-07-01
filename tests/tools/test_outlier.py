"""Unit tests for OutlierTool (issue #5)."""
import pandas as pd
import pytest

from ds_agent.tools.outlier import OutlierTool, _describe_outliers


def _df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestOutlierDetection:
    def test_hand_placed_outlier_detected_by_iqr(self):
        # values 1-24 are normal; 100 is a clear outlier
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert col["iqr_method"]["outlier_count"] >= 1

    def test_hand_placed_outlier_detected_by_zscore(self):
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert col["zscore_method"]["outlier_count"] >= 1

    def test_clean_data_no_outliers(self):
        # Uniform data with no outliers
        values = [float(i) for i in range(1, 21)]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert col["iqr_method"]["outlier_count"] == 0
        assert col["zscore_method"]["outlier_count"] == 0

    def test_both_counts_reported_separately(self):
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert "iqr_method" in col
        assert "zscore_method" in col
        assert "outlier_count" in col["iqr_method"]
        assert "outlier_count" in col["zscore_method"]

    def test_iqr_bounds_present(self):
        df = _df(v=list(range(1, 21)))
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert "lower_bound" in col["iqr_method"]
        assert "upper_bound" in col["iqr_method"]
        assert "q1" in col["iqr_method"]
        assert "q3" in col["iqr_method"]

    def test_zscore_stats_present(self):
        df = _df(v=list(range(1, 21)))
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert "mean" in col["zscore_method"]
        assert "std" in col["zscore_method"]
        assert "threshold" in col["zscore_method"]

    def test_outlier_indices_are_integers(self):
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        for idx in col["iqr_method"]["outlier_indices"]:
            assert isinstance(idx, int)

    def test_skips_column_with_fewer_than_4_values(self):
        df = _df(tiny=[1, 2, 3])
        result = OutlierTool().run(df)
        assert result["columns"] == []

    def test_skips_non_numeric_columns(self):
        # Need 4+ rows so score column isn't skipped by the n < 4 guard
        df = _df(name=["Alice", "Bob", "Carol", "Dave"], score=[1, 2, 3, 4])
        result = OutlierTool().run(df)
        names = [c["column"] for c in result["columns"]]
        assert "name" not in names
        assert "score" in names

    def test_custom_zscore_threshold(self):
        # With a very low threshold, even moderate values become outliers
        df = _df(v=list(range(1, 21)))
        result_strict = OutlierTool().run(df, zscore_threshold=0.5)
        result_loose = OutlierTool().run(df, zscore_threshold=3.0)
        strict_count = result_strict["columns"][0]["zscore_method"]["outlier_count"]
        loose_count = result_loose["columns"][0]["zscore_method"]["outlier_count"]
        assert strict_count >= loose_count

    def test_column_filter(self):
        df = _df(a=list(range(1, 21)), b=list(range(100, 120)))
        result = OutlierTool().run(df, columns=["a"])
        assert len(result["columns"]) == 1
        assert result["columns"][0]["column"] == "a"

    def test_constant_column_zscore_is_zero_outliers(self):
        df = _df(c=[5.0] * 20)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert col["zscore_method"]["outlier_count"] == 0


class TestAggregateOnlyOutput:
    def test_no_raw_row_data_in_output(self):
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        assert "row_values" not in col
        assert "raw_data" not in col

    def test_outlier_indices_not_full_rows(self):
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        col = result["columns"][0]
        # Indices are acceptable (aggregate-level), but full row dicts are not
        for idx in col["iqr_method"]["outlier_indices"]:
            assert not isinstance(idx, dict)


class TestNarrative:
    def test_narrative_present(self):
        values = list(range(1, 25)) + [100]
        df = _df(v=values)
        result = OutlierTool().run(df)
        assert "narrative" in result["columns"][0]
        assert len(result["columns"][0]["narrative"]) > 0

    def test_no_outliers_narrative(self):
        narrative = _describe_outliers("x", iqr_count=0, zscore_count=0, n=20)
        assert "no outliers" in narrative.lower()

    def test_disagreement_surfaced_in_narrative(self):
        # IQR finds 2, zscore finds 5 — disagreement
        narrative = _describe_outliers("x", iqr_count=2, zscore_count=5, n=100)
        assert "disagree" in narrative.lower() or "both" in narrative.lower()

    def test_iqr_only_narrative(self):
        narrative = _describe_outliers("x", iqr_count=3, zscore_count=0, n=100)
        assert "IQR" in narrative

    def test_zscore_only_narrative(self):
        narrative = _describe_outliers("x", iqr_count=0, zscore_count=2, n=100)
        assert "z-score" in narrative.lower() or "zscore" in narrative.lower() or "z-score" in narrative

    def test_narrative_includes_column_name(self):
        narrative = _describe_outliers("revenue", iqr_count=1, zscore_count=1, n=50)
        assert "revenue" in narrative
