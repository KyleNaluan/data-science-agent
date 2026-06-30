from __future__ import annotations

import re

_PLACEHOLDER = "_Not yet analyzed in this run._"


def _safe_filename(col_name: str) -> str:
    return re.sub(r"[^\w\-]", "_", col_name)


def _render_data_quality_scorecard(
    schema: dict,
    flagged_assumptions: list[dict] | None = None,
) -> str:
    cols = schema.get("columns", [])
    if not cols:
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

    flagged_cols = [c for c in cols if c.get("is_ambiguous")]
    if flagged_cols:
        lines.append("")
        lines.append("**Flagged columns** (require review before downstream analysis):\n\n")
        for col in flagged_cols:
            lines.append(f"- **{col['name']}**: {col.get('ambiguity_reason', '')}")

    assumptions = flagged_assumptions or []
    if assumptions:
        lines.append("")
        lines.append("**Flagged assumptions** (agent proceeded using documented defaults):\n")
        for assumption in assumptions:
            ctx = assumption.get("context", {})
            trigger_type = assumption.get("trigger_type", "")
            if trigger_type == "ambiguous_column_type":
                col_name = ctx.get("column", "?")
                value = assumption.get("assumption", "?")
                reason = ctx.get("ambiguity_reason", "")
                lines.append(f"- **{col_name}**: type assumed to be `{value}` ({reason})")
            elif trigger_type == "tiny_dataset":
                row_count = ctx.get("row_count", "?")
                lines.append(
                    f"- **tiny_dataset**: dataset has only {row_count} rows; "
                    "analysis proceeded with defaults"
                )
            else:
                lines.append(f"- **{trigger_type}**: assumed `{assumption.get('assumption', '?')}`")

    return "\n".join(lines) + "\n"


def _render_distributions(dist_data: dict) -> str:
    cols = dist_data.get("columns", [])
    if not cols:
        return "_No numeric columns found for distribution analysis._\n"

    lines: list[str] = []
    for col in cols:
        safe = _safe_filename(col["column"])
        lines.append(f"### {col['column']}")
        lines.append("")
        lines.append(col["narrative"])
        lines.append(
            f"- Skew: {col['skew']:.4f} | "
            f"Excess kurtosis: {col['kurtosis']:.4f} | "
            f"n={col['sample_count']}"
        )
        lines.append("")
        lines.append(f"![{col['column']} histogram](charts/{safe}.png)")
        lines.append("")

    return "\n".join(lines)


def render_report(
    tool_results: dict,
    metadata: dict | None = None,
    flagged_assumptions: list[dict] | None = None,
) -> str:
    """
    Assemble the fixed 5-section markdown report from accumulated tool results.

    tool_results: keyed by tool name (e.g. "schema_inference" -> schema dict)
    metadata: optional dict with "csv_path" for the report header
    flagged_assumptions: non-interactive uncertainty resolutions to surface in the scorecard
    """
    csv_path = (metadata or {}).get("csv_path", "unknown")

    sections: list[str] = []

    sections.append(f"# EDA Report\n\n**Source:** `{csv_path}`\n")

    sections.append("## Executive Summary\n\n" + _PLACEHOLDER)

    schema = tool_results.get("schema_inference")
    if schema:
        scorecard_body = _render_data_quality_scorecard(
            schema, flagged_assumptions=flagged_assumptions
        )
    else:
        scorecard_body = _PLACEHOLDER
    sections.append("## Data Quality Scorecard\n\n" + scorecard_body)

    dist_data = tool_results.get("distribution_analysis")
    if dist_data:
        dist_body = _render_distributions(dist_data)
    else:
        dist_body = _PLACEHOLDER
    sections.append("## Distributions\n\n" + dist_body)

    sections.append("## Correlations\n\n" + _PLACEHOLDER)
    sections.append("## Feature Engineering Recommendations\n\n" + _PLACEHOLDER)

    return "\n\n".join(sections) + "\n"
