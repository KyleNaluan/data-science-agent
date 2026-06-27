from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import pandas as pd

from .base import Tool

ColumnType = Literal["numeric", "categorical", "datetime", "identifier", "boolean", "text"]

_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"           # ISO: 2024-01-15
    r"|^\d{2}/\d{2}/\d{4}"          # US: 01/15/2024
    r"|^\d{4}/\d{2}/\d{2}"          # 2024/01/15
    r"|^\d{2}-\d{2}-\d{4}"          # 15-01-2024
)

_TOP_VALUE_COUNTS_LIMIT = 10


@dataclass
class ColumnStats:
    name: str
    inferred_type: ColumnType
    dtype: str
    row_count: int
    null_count: int
    null_rate: float
    unique_count: int
    mean: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    top_value_counts: Optional[dict[str, int]]
    is_ambiguous: bool
    ambiguity_reason: Optional[str]


@dataclass
class SchemaResult:
    row_count: int
    column_count: int
    columns: list[ColumnStats]

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [asdict(c) for c in self.columns],
        }


def _is_integer_valued(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    try:
        return (non_null == non_null.round()).all()
    except Exception:
        return False


def _looks_like_datetime_strings(series: pd.Series) -> bool:
    sample = series.dropna().head(50).astype(str)
    if len(sample) == 0:
        return False
    match_rate = sample.str.match(_DATE_RE).mean()
    if match_rate < 0.8:
        return False
    try:
        pd.to_datetime(series.dropna().head(100), errors="raise")
        return True
    except Exception:
        return False


def _infer_column(
    series: pd.Series,
    row_count: int,
) -> tuple[ColumnType, bool, Optional[str]]:
    """Return (inferred_type, is_ambiguous, ambiguity_reason)."""
    kind = series.dtype.kind  # 'b', 'i', 'u', 'f', 'O', 'M', etc.
    non_null = series.dropna()
    unique_count = int(series.nunique(dropna=True))
    unique_rate = unique_count / max(row_count, 1)

    if kind == "b":
        return "boolean", False, None

    if kind == "M":
        return "datetime", False, None

    if kind in ("i", "u", "f"):
        if unique_rate > 0.9 and unique_count > 50 and _is_integer_valued(non_null):
            return (
                "identifier",
                True,
                "high-cardinality integer column — may be a numeric ID rather than a measurement",
            )
        return "numeric", False, None

    if kind == "O":
        if _looks_like_datetime_strings(series):
            return "datetime", False, None

        if unique_rate < 0.05 or unique_count <= 20:
            return "categorical", False, None

        avg_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
        if avg_len > 50:
            return "text", False, None

        if unique_count <= 50:
            return "categorical", False, None

        if unique_count > 100 and avg_len < 30:
            return (
                "categorical",
                True,
                "high-cardinality string column — may be free-form text or a categorical with many values",
            )

        return "text", False, None

    return "categorical", False, None


def infer_schema(df: pd.DataFrame) -> SchemaResult:
    """Infer schema from a DataFrame.

    Returns aggregate-only statistics — no raw row data (ADR-0002).
    """
    row_count = len(df)
    columns: list[ColumnStats] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_rate = round(null_count / max(row_count, 1), 4)
        unique_count = int(series.nunique(dropna=True))

        inferred_type, is_ambiguous, ambiguity_reason = _infer_column(series, row_count)

        mean = std = min_val = max_val = None
        top_value_counts = None
        non_null = series.dropna()

        if inferred_type == "numeric" and series.dtype.kind in ("i", "u", "f"):
            if len(non_null) > 0:
                mean = round(float(non_null.mean()), 6)
                min_val = float(non_null.min())
                max_val = float(non_null.max())
            if len(non_null) > 1:
                std = round(float(non_null.std()), 6)

        if inferred_type in ("categorical", "boolean"):
            vc = series.value_counts(dropna=True).head(_TOP_VALUE_COUNTS_LIMIT)
            top_value_counts = {str(k): int(v) for k, v in vc.items()}

        columns.append(
            ColumnStats(
                name=str(col),
                inferred_type=inferred_type,
                dtype=str(series.dtype),
                row_count=row_count,
                null_count=null_count,
                null_rate=null_rate,
                unique_count=unique_count,
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                top_value_counts=top_value_counts,
                is_ambiguous=is_ambiguous,
                ambiguity_reason=ambiguity_reason,
            )
        )

    return SchemaResult(row_count=row_count, column_count=len(df.columns), columns=columns)


class SchemaInferenceTool(Tool):
    @property
    def name(self) -> str:
        return "schema_inference"

    @property
    def description(self) -> str:
        return (
            "Infer column types (numeric, categorical, datetime, identifier, boolean, text), "
            "cardinality, and nullity for the loaded DataFrame. "
            "Returns aggregate statistics only — no raw row data (ADR-0002)."
        )

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, df: pd.DataFrame, **kwargs) -> dict:
        return infer_schema(df).to_dict()
