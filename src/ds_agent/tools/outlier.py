from __future__ import annotations

import pandas as pd

from .base import Tool


def _describe_outliers(col: str, iqr_count: int, zscore_count: int, n: int) -> str:
    if iqr_count == 0 and zscore_count == 0:
        return f"**{col}** (n={n}): No outliers detected by either IQR or z-score method."

    parts = []
    if iqr_count > 0:
        parts.append(f"IQR method: {iqr_count} outlier(s)")
    if zscore_count > 0:
        parts.append(f"z-score method: {zscore_count} outlier(s)")

    pct = max(iqr_count, zscore_count) / max(n, 1) * 100
    severity = "severe" if pct > 5 else "moderate" if pct > 1 else "minor"

    if iqr_count > 0 and zscore_count > 0:
        if iqr_count != zscore_count:
            agreement = " (methods disagree on count — both are reported)"
        else:
            agreement = " (both methods agree)"
    elif iqr_count > 0:
        agreement = " (IQR only — z-score method found none)"
    else:
        agreement = " (z-score only — IQR method found none)"

    return (
        f"**{col}** (n={n}): {severity.capitalize()} outlier presence detected — "
        + ", ".join(parts) + agreement + "."
    )


class OutlierTool(Tool):
    @property
    def name(self) -> str:
        return "outlier_detection"

    @property
    def description(self) -> str:
        return (
            "Detect outliers in numeric columns using both IQR and z-score methods. "
            "Returns aggregate statistics only — outlier counts, bounds, and indices (ADR-0002). "
            "Both method counts are reported separately to surface any disagreements."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Numeric column names to analyze. If omitted, all numeric columns.",
                },
                "zscore_threshold": {
                    "type": "number",
                    "description": "Z-score threshold for outlier detection. Default 3.0.",
                },
            },
            "required": [],
        }

    def run(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        zscore_threshold: float = 3.0,
        **kwargs,
    ) -> dict:
        if columns is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
        else:
            numeric_cols = [
                c for c in columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]

        results = []
        for col_name in numeric_cols:
            series = df[col_name].dropna()
            n = len(series)
            if n < 4:
                continue

            # IQR method
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            iqr_mask = (series < lower_iqr) | (series > upper_iqr)
            iqr_indices = [int(i) for i in series.index[iqr_mask].tolist()]
            iqr_count = int(iqr_mask.sum())

            # Z-score method
            mean = float(series.mean())
            std = float(series.std())
            if std == 0:
                zscore_count = 0
                zscore_indices: list[int] = []
            else:
                z_scores = ((series - mean) / std).abs()
                zscore_mask = z_scores > zscore_threshold
                zscore_indices = [int(i) for i in series.index[zscore_mask].tolist()]
                zscore_count = int(zscore_mask.sum())

            results.append({
                "column": col_name,
                "sample_count": n,
                "iqr_method": {
                    "outlier_count": iqr_count,
                    "lower_bound": round(lower_iqr, 6),
                    "upper_bound": round(upper_iqr, 6),
                    "q1": round(q1, 6),
                    "q3": round(q3, 6),
                    "iqr": round(iqr, 6),
                    "outlier_indices": iqr_indices,
                },
                "zscore_method": {
                    "outlier_count": zscore_count,
                    "threshold": zscore_threshold,
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "outlier_indices": zscore_indices,
                },
                "narrative": _describe_outliers(col_name, iqr_count, zscore_count, n),
            })

        return {"columns": results}
