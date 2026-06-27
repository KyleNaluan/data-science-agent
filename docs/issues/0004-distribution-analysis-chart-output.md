## What to build

A distribution-analysis Tool computing histograms, skew, and kurtosis for numeric columns, populating the Distributions report section with plain-language narrative findings. This is the first slice to produce chart artifacts: histogram PNGs are saved to a `charts/` subdirectory alongside the report and referenced from the Distributions section via relative markdown image links.

## Acceptance criteria

- [ ] Distribution Tool is unit-tested against fixture DataFrames with known skew/kurtosis properties
- [ ] A run against a numeric-heavy fixture produces histogram PNGs in a `charts/` subdirectory and a populated Distributions section referencing them via relative paths
- [ ] An integration test verifies referenced chart files actually exist on disk, not just that the report text mentions a chart
- [ ] Distribution narrative describes findings in plain language, not just raw statistics
- [ ] Tool output remains aggregate-only (histogram bin counts/edges, skew/kurtosis values) — never raw row values

## Blocked by

- Slice 1: Foundation — ingest, schema inference, report skeleton
