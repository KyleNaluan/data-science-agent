## What to build

Upgrade the Executive Summary section from the Slice 1 placeholder to a real LLM-synthesized narrative that draws on the populated Data Quality Scorecard, Distributions, Correlations, and Feature Engineering Recommendations sections, surfacing the handful of findings most relevant to a non-technical reader.

## Acceptance criteria

- [ ] A full run against a fixture exercising most prior slices' capabilities produces an Executive Summary that references specific findings from the other sections (not generic boilerplate)
- [ ] Executive Summary generation is covered by an integration test using a fake LLM client, asserting that the summary references content present in the other sections (structural check, not exact-prose matching)
- [ ] Placeholder behavior from Slice 1 is fully replaced — no "not yet analyzed" text remains in a complete run

## Blocked by

- Slice 3: Missing-value analysis + threshold config
- Slice 4: Distribution analysis + chart output
- Slice 5: Outlier detection
- Slice 6: Correlation analysis
- Slice 7: Feature engineering suggestions
