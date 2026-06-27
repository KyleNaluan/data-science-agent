from __future__ import annotations

_PLACEHOLDER = "_Not yet analyzed in this run._"


def _render_data_quality_scorecard(schema: dict) -> str:
    cols = schema.get("columns", [])
    rows = len(cols)
    if rows == 0:
        return "_No columns found._\n"

    lines = [
        f"**Dataset:** {schema.get('row_count', '?')} rows × {schema.get('column_count', '?')} columns\n",
        "| Column | Type | Nullity | Cardinality | Notes |",
        "|--------|------|---------|-------------|-------|",
    ]
    for col in cols:
        null_pct = f"{col['null_rate'] * 100:.1f}%"
        note = ""
        if col.get("is_ambiguous"):
            note = f"[!] {col.get('ambiguity_reason', 'ambiguous')}"
        lines.append(
            f"| {col['name']} | {col['inferred_type']} | {null_pct} | {col['unique_count']} | {note} |"
        )

    flagged = [c for c in cols if c.get("is_ambiguous")]
    if flagged:
        lines.append("")
        lines.append("**Flagged columns** (require review before downstream analysis):\n\n")
        for col in flagged:
            lines.append(f"- **{col['name']}**: {col.get('ambiguity_reason', '')}")

    return "\n".join(lines) + "\n"


def render_report(tool_results: dict, metadata: dict | None = None) -> str:
    """
    Assemble the fixed 5-section markdown report from accumulated tool results.

    tool_results: keyed by tool name (e.g. "schema_inference" -> schema dict)
    metadata: optional dict with "csv_path" for the report header
    """
    csv_path = (metadata or {}).get("csv_path", "unknown")

    sections: list[str] = []

    sections.append(f"# EDA Report\n\n**Source:** `{csv_path}`\n")

    sections.append("## Executive Summary\n\n" + _PLACEHOLDER)

    schema = tool_results.get("schema_inference")
    if schema:
        scorecard_body = _render_data_quality_scorecard(schema)
    else:
        scorecard_body = _PLACEHOLDER
    sections.append("## Data Quality Scorecard\n\n" + scorecard_body)

    sections.append("## Distributions\n\n" + _PLACEHOLDER)
    sections.append("## Correlations\n\n" + _PLACEHOLDER)
    sections.append("## Feature Engineering Recommendations\n\n" + _PLACEHOLDER)

    return "\n\n".join(sections) + "\n"
