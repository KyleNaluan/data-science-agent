from __future__ import annotations

import math

import pandas as pd

from .base import Tool


class FeatureSuggestionTool(Tool):
    @property
    def name(self) -> str:
        return "feature_suggestion"

    @property
    def description(self) -> str:
        return (
            "Generate feature engineering candidates grounded in dataset properties: "
            "high-skew numeric columns → transform suggestions, high-cardinality categoricals → "
            "encoding strategies, correlated numeric pairs → interaction term candidates. "
            "Returns aggregate-only recommendations (ADR-0002). "
            "Call after distribution_analysis, missing_value_analysis, and correlation_analysis. "
            "Populates the Feature Engineering Recommendations report section."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skew_threshold": {
                    "type": "number",
                    "description": "Minimum |skew| to flag for a transform suggestion. Default 1.5.",
                },
                "high_cardinality_threshold": {
                    "type": "integer",
                    "description": (
                        "Unique value count above which a categorical column is considered "
                        "high cardinality. Default 10."
                    ),
                },
                "correlation_threshold": {
                    "type": "number",
                    "description": "Minimum |r| to suggest an interaction term. Default 0.7.",
                },
            },
            "required": [],
        }

    def run(
        self,
        df: pd.DataFrame,
        skew_threshold: float = 1.5,
        high_cardinality_threshold: int = 10,
        correlation_threshold: float = 0.7,
        **kwargs,
    ) -> dict:
        suggestions: list[dict] = []

        # --- Numeric columns: high skew → transform suggestions ---
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col_name in numeric_cols:
            series = df[col_name].dropna()
            if len(series) < 3:
                continue
            skew = float(series.skew())
            if abs(skew) >= skew_threshold:
                if skew > 0:
                    transform = (
                        "log1p transform"
                        if float(series.min()) >= 0
                        else "Box-Cox or Yeo-Johnson transform"
                    )
                else:
                    transform = "square or exponential transform"
                priority = "high" if abs(skew) >= 2.5 else "medium"
                suggestions.append({
                    "columns": [col_name],
                    "suggestion_type": "log_transform",
                    "rationale": (
                        f"'{col_name}' is highly skewed (skew={skew:.2f}); "
                        f"consider a {transform} to normalize the distribution before modeling"
                    ),
                    "priority": priority,
                })

        # --- Categorical columns: high cardinality → encoding strategy suggestions ---
        cat_cols = df.select_dtypes(include=["str", "object", "category"]).columns.tolist()
        for col_name in cat_cols:
            unique_count = int(df[col_name].nunique())
            if unique_count > high_cardinality_threshold:
                suggestions.append({
                    "columns": [col_name],
                    "suggestion_type": "encoding",
                    "rationale": (
                        f"'{col_name}' has high cardinality ({unique_count} unique values); "
                        "one-hot encoding would produce many sparse columns — "
                        "consider target encoding, ordinal encoding, or embeddings"
                    ),
                    "priority": "high",
                })
            elif unique_count > 1:
                suggestions.append({
                    "columns": [col_name],
                    "suggestion_type": "encoding",
                    "rationale": (
                        f"'{col_name}' is a low-cardinality categorical ({unique_count} unique values); "
                        "one-hot encoding is appropriate"
                    ),
                    "priority": "low",
                })

        # --- Correlated numeric pairs → interaction term suggestions ---
        if len(numeric_cols) >= 2:
            numeric_df = df[numeric_cols].dropna(how="all")
            if len(numeric_df) >= 3:
                corr = numeric_df.corr(method="pearson")
                for i, col_a in enumerate(numeric_cols):
                    for j, col_b in enumerate(numeric_cols):
                        if j <= i:
                            continue
                        try:
                            r = float(corr.loc[col_a, col_b])
                        except KeyError:
                            continue
                        if not math.isfinite(r):
                            continue
                        if abs(r) >= correlation_threshold:
                            direction = "positively" if r > 0 else "negatively"
                            suggestions.append({
                                "columns": [col_a, col_b],
                                "suggestion_type": "interaction_term",
                                "rationale": (
                                    f"'{col_a}' and '{col_b}' are strongly {direction} correlated "
                                    f"(r={r:.3f}); consider an interaction term or investigate "
                                    "multicollinearity before including both in a model"
                                ),
                                "priority": "medium",
                            })

        return {"suggestions": suggestions}
