## What to build

A correlation-analysis Tool computing a full correlation matrix plus a callout of the most notable pairwise relationships, populating the Correlations section with a heatmap chart (via the Slice 4 chart mechanism) and a narrative summary of the standout pairs.

## Acceptance criteria

- [ ] Correlation Tool is unit-tested against fixture DataFrames with known correlation structure
- [ ] A run against a fixture with known correlated columns produces a heatmap chart and a Correlations section narrative naming the standout pairs (not just a dump of the full matrix)
- [ ] Tool output is the matrix values and ranked pairs only — never row-level data
- [ ] The full matrix is available in the report (e.g. as a table or embedded chart) in addition to the narrative callouts

## Blocked by

- Slice 4: Distribution analysis + chart output
