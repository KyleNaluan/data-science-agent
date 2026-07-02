from __future__ import annotations

import re

PLACEHOLDER = "_Not yet analyzed in this run._"
_PLACEHOLDER = PLACEHOLDER  # kept for internal use


def _safe_filename(col_name: str) -> str:
    return re.sub(r"[^\w\-]", "_", col_name)


def _render_data_quality_scorecard(
    schema: dict,
    flagged_assumptions: list[dict] | None = None,
    missing_value_data: dict | None = None,
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
            elif trigger_type == "high_missing_rate":
                col_name = ctx.get("column", "?")
                rate = ctx.get("missing_rate", 0)
                rec = ctx.get("imputation_recommendation", "")
                lines.append(
                    f"- **{col_name}**: {rate * 100:.1f}% missing values — {rec}"
                )
            elif trigger_type == "conflicting_schema_hints":
                col_name = ctx.get("column", "?")
                hint = ctx.get("hint_type", "?")
                inferred = ctx.get("inferred_type", "?")
                lines.append(
                    f"- **{col_name}**: schema file specifies `{hint}`, "
                    f"inference suggested `{inferred}` — used schema file hint"
                )
            elif trigger_type == "ambiguous_join_key":
                candidates = ctx.get("candidates", [])
                lines.append(
                    f"- **ambiguous_join_key**: proceeded with best join candidate "
                    + (f"({candidates[0]})" if candidates else "")
                )
            else:
                lines.append(f"- **{trigger_type}**: assumed `{assumption.get('assumption', '?')}`")

    # Missing value breakdown (issue #4)
    if missing_value_data:
        mv_cols = missing_value_data.get("columns", [])
        cols_with_missing = [c for c in mv_cols if c["missing_count"] > 0]
        if cols_with_missing:
            lines.append("")
            lines.append("### Missing Value Breakdown\n")
            lines.append("| Column | Missing Count | Missing Rate | Imputation Recommendation |")
            lines.append("|--------|--------------|--------------|--------------------------|")
            for c in mv_cols:
                rate_str = f"{c['missing_rate'] * 100:.1f}%"
                lines.append(
                    f"| {c['column']} | {c['missing_count']} | {rate_str} "
                    f"| {c['imputation_recommendation']} |"
                )

    return "\n".join(lines) + "\n"


def _render_distributions(
    dist_data: dict,
    outlier_data: dict | None = None,
) -> str:
    cols = dist_data.get("columns", [])
    if not cols:
        return "_No numeric columns found for distribution analysis._\n"

    outlier_by_col: dict[str, dict] = {}
    if outlier_data:
        for ocol in outlier_data.get("columns", []):
            outlier_by_col[ocol["column"]] = ocol

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

        # Outlier callout for this column (issue #5)
        if col["column"] in outlier_by_col:
            ocol = outlier_by_col[col["column"]]
            lines.append(ocol["narrative"])
            iqr_c = ocol["iqr_method"]["outlier_count"]
            zsc_c = ocol["zscore_method"]["outlier_count"]
            lines.append(
                f"- IQR outliers: {iqr_c} | "
                f"Z-score outliers: {zsc_c}"
            )
            lines.append("")
            lines.append(f"![{col['column']} outliers](charts/{safe}_outliers.png)")
            lines.append("")

    return "\n".join(lines)


def _render_correlations(corr_data: dict) -> str:
    columns = corr_data.get("columns", [])
    ranked_pairs = corr_data.get("ranked_pairs", [])
    narrative_pairs = corr_data.get("narrative_pairs", [])
    matrix = corr_data.get("matrix", {})

    if not columns or len(columns) < 2:
        return "_Not enough numeric columns for correlation analysis._\n"

    lines: list[str] = []

    lines.append("### Notable Relationships\n")
    if narrative_pairs:
        for narrative in narrative_pairs:
            lines.append(f"- {narrative}")
    else:
        lines.append("_No strong correlations found (|r| ≥ 0.3)._")
    lines.append("")

    lines.append("![Correlation Heatmap](charts/correlation_heatmap.png)\n")

    lines.append("### Full Correlation Matrix\n")
    header = "| |" + "|".join(f" {c} " for c in columns) + "|"
    separator = "|---|" + "|".join("---" for _ in columns) + "|"
    lines.append(header)
    lines.append(separator)
    for col_a in columns:
        row_vals = [
            f"{matrix.get(col_a, {}).get(col_b, 0.0):.3f}"
            for col_b in columns
        ]
        lines.append(f"| **{col_a}** |" + "|".join(row_vals) + "|")

    return "\n".join(lines)


def _render_feature_engineering(feat_data: dict) -> str:
    suggestions = feat_data.get("suggestions", [])
    if not suggestions:
        return "_No feature engineering suggestions generated for this dataset._\n"

    lines: list[str] = []
    high = [s for s in suggestions if s.get("priority") == "high"]
    medium = [s for s in suggestions if s.get("priority") == "medium"]
    low = [s for s in suggestions if s.get("priority") == "low"]

    for label, group in [("High Priority", high), ("Medium Priority", medium), ("Low Priority", low)]:
        if not group:
            continue
        lines.append(f"### {label}\n")
        for s in group:
            cols = s.get("columns", [])
            col_display = " × ".join(f"**{c}**" for c in cols) if cols else "?"
            stype = s.get("suggestion_type", "").replace("_", " ").title()
            lines.append(f"- **{stype}** — {col_display}: {s.get('rationale', '')}")
        lines.append("")

    return "\n".join(lines)


def render_report(
    tool_results: dict,
    metadata: dict | None = None,
    flagged_assumptions: list[dict] | None = None,
    executive_summary: str | None = None,
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
    exec_body = executive_summary if executive_summary else _PLACEHOLDER
    sections.append("## Executive Summary\n\n" + exec_body)

    schema = tool_results.get("schema_inference")
    missing_value_data = tool_results.get("missing_value_analysis")
    if schema:
        scorecard_body = _render_data_quality_scorecard(
            schema,
            flagged_assumptions=flagged_assumptions,
            missing_value_data=missing_value_data,
        )
    else:
        scorecard_body = _PLACEHOLDER
    sections.append("## Data Quality Scorecard\n\n" + scorecard_body)

    dist_data = tool_results.get("distribution_analysis")
    outlier_data = tool_results.get("outlier_detection")
    if dist_data:
        dist_body = _render_distributions(dist_data, outlier_data=outlier_data)
    elif outlier_data:
        # Outlier data present but no distribution data — still render outlier findings
        dist_body = _render_distributions({"columns": []}, outlier_data=outlier_data)
        if dist_body.startswith("_No numeric"):
            dist_body = _PLACEHOLDER
    else:
        dist_body = _PLACEHOLDER
    sections.append("## Distributions\n\n" + dist_body)

    corr_data = tool_results.get("correlation_analysis")
    if corr_data:
        corr_body = _render_correlations(corr_data)
    else:
        corr_body = _PLACEHOLDER
    sections.append("## Correlations\n\n" + corr_body)

    feat_data = tool_results.get("feature_suggestion")
    if feat_data is not None:
        feat_body = _render_feature_engineering(feat_data)
    else:
        feat_body = _PLACEHOLDER
    sections.append("## Feature Engineering Recommendations\n\n" + feat_body)

    return "\n\n".join(sections) + "\n"
