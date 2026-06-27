## What to build

An outlier-detection Tool applying both IQR and z-score methods, flagging outlier counts/indices per column and rendering outlier plots saved via the Slice 4 chart mechanism. Outlier findings render within the Distributions section, alongside each column's distribution narrative, since the fixed report template has no dedicated outliers section. Confirm this placement before building if it doesn't match expectations.

## Acceptance criteria

- [ ] Outlier Tool is unit-tested against fixture DataFrames with known, hand-placed outliers, verifying both IQR and z-score methods agree on the expected flagged points
- [ ] A run against an outlier-bearing fixture produces an outlier plot chart and a narrative callout within the Distributions section
- [ ] Tool output is aggregate-only (counts, thresholds, outlier indices) — never the full row a flagged outlier belongs to
- [ ] Disagreement between IQR and z-score methods on a given point is surfaced rather than silently resolved (e.g. both counts are reported, not just one)

## Blocked by

- Slice 4: Distribution analysis + chart output
