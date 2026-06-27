## What to build

A feature-suggestion-generation Tool that proposes candidates for encoding, binning, scaling, or interaction terms, grounded explicitly in findings from the missing-value (Slice 3), distribution (Slice 4), and correlation (Slice 6) tools — e.g. a highly skewed column suggests a log transform, a high-cardinality categorical suggests an encoding strategy, two highly correlated numeric columns suggest an interaction term. Populates the Feature Engineering Recommendations section. Suggestions are recommendations only; V1 does not apply any transformation automatically.

## Acceptance criteria

- [ ] Suggestion Tool is unit-tested against fixture DataFrames with known properties (a deliberately skewed column, a high-cardinality categorical, a correlated pair), verifying it proposes the expected suggestion category for each
- [ ] Every suggestion in the report cites the specific EDA finding that grounded it (e.g. "column X is highly skewed (skew=2.3); consider a log transform") rather than a generic recommendation
- [ ] No data transformation is actually applied to the dataset — suggestions are descriptive output only
- [ ] Tool output remains aggregate-only

## Blocked by

- Slice 3: Missing-value analysis + threshold config
- Slice 4: Distribution analysis + chart output
- Slice 6: Correlation analysis
