from __future__ import annotations

import pandas as pd

from .base import Tool

_HIGH_CONFIDENCE_THRESHOLD = 0.80
_AMBIGUOUS_THRESHOLD = 0.40


def _name_score(col_a: str, col_b: str) -> float:
    a, b = col_a.lower().strip(), col_b.lower().strip()
    if a == b:
        return 1.0
    # Common ID suffix normalisation
    for suffix in ("_id", "_key", "_code", "_num"):
        a_stripped = a[: -len(suffix)] if a.endswith(suffix) else a
        b_stripped = b[: -len(suffix)] if b.endswith(suffix) else b
        if a_stripped == b_stripped and a_stripped:
            return 0.75
    return 0.0


def _dtype_compatible(dtype_a: str, dtype_b: str) -> bool:
    def _kind(d: str) -> str:
        if any(k in d for k in ("int", "float", "uint")):
            return "numeric"
        if any(k in d for k in ("object", "string", "category")):
            return "string"
        return d

    return _kind(dtype_a) == _kind(dtype_b)


def _overlap_score(series_a: pd.Series, series_b: pd.Series) -> float:
    set_a = set(series_a.dropna().unique())
    set_b = set(series_b.dropna().unique())
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def _uniqueness_score(series: pd.Series) -> float:
    n = len(series.dropna())
    return series.nunique() / n if n > 0 else 0.0


def score_candidates(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[dict]:
    candidates = []
    for col_a in df_a.columns:
        for col_b in df_b.columns:
            name_s = _name_score(col_a, col_b)
            if name_s == 0.0:
                continue
            if not _dtype_compatible(str(df_a[col_a].dtype), str(df_b[col_b].dtype)):
                continue

            overlap = _overlap_score(df_a[col_a], df_b[col_b])
            uniq = (_uniqueness_score(df_a[col_a]) + _uniqueness_score(df_b[col_b])) / 2
            score = round(0.40 * name_s + 0.30 * overlap + 0.30 * uniq, 4)

            candidates.append({
                "col_a": col_a,
                "col_b": col_b,
                "score": score,
                "name_score": round(name_s, 4),
                "overlap_score": round(overlap, 4),
                "uniqueness_score": round(uniq, 4),
                "dtype_a": str(df_a[col_a].dtype),
                "dtype_b": str(df_b[col_b].dtype),
            })

    candidates.sort(key=lambda c: -c["score"])
    return candidates


class JoinKeyInferenceTool(Tool):
    @property
    def name(self) -> str:
        return "join_key_inference"

    @property
    def description(self) -> str:
        return (
            "Score candidate join keys across two DataFrames by column-name similarity, "
            "dtype compatibility, and value overlap/uniqueness. "
            "Returns candidate scores only — no row-level join previews (ADR-0002)."
        )

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, df: pd.DataFrame, extra_df: pd.DataFrame | None = None, **kwargs) -> dict:
        if extra_df is None:
            return {"candidates": [], "confidence": "no_second_dataframe", "best_candidate": None}

        candidates = score_candidates(df, extra_df)

        if not candidates:
            confidence = "no_candidates"
            best = None
        elif candidates[0]["score"] >= _HIGH_CONFIDENCE_THRESHOLD:
            confidence = "high"
            best = candidates[0]
        elif candidates[0]["score"] >= _AMBIGUOUS_THRESHOLD:
            confidence = "ambiguous"
            best = candidates[0]
        else:
            confidence = "low"
            best = None

        return {
            "candidates": candidates[:10],
            "confidence": confidence,
            "best_candidate": best,
            "high_confidence_threshold": _HIGH_CONFIDENCE_THRESHOLD,
            "ambiguous_threshold": _AMBIGUOUS_THRESHOLD,
        }
