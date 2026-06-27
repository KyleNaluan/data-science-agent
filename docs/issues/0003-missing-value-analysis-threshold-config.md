## What to build

A missing-value-analysis Tool that computes the missing-value rate per column and imputation recommendations, populating the Data Quality Scorecard. When a column's missing-value rate exceeds a configured threshold (default 20%, per the original concept doc), it's raised as an Uncertainty trigger through the Slice 2 mechanism.

This slice also introduces the threshold-configuration system: a checked-in config file holding default thresholds (starting with the missing-value-rate threshold), falling back to built-in defaults when the file is absent, with individual thresholds overridable via CLI flags for a single run. Later slices that introduce their own thresholds (e.g. join-confidence) add keys to this same config rather than building a new mechanism.

## Acceptance criteria

- [ ] Missing-value Tool is unit-tested directly against fixture DataFrames with known missingness rates, including a fixture at exactly the default threshold boundary
- [ ] A fixture with missingness above the threshold raises the Slice 2 checkpoint/flagged-assumption mechanism; a fixture below it does not
- [ ] A config file can override the missing-value threshold, verified by a test that sets a stricter/looser threshold and confirms the trigger fires/doesn't fire accordingly
- [ ] A CLI flag can override the threshold for a single run without modifying the config file
- [ ] The Data Quality Scorecard includes a missing-value breakdown per column with imputation recommendations
- [ ] No row-level data appears in Tool output (per ADR-0002) — only per-column missingness statistics

## Blocked by

- Slice 2: Uncertainty mechanism — ambiguous column type + tiny dataset
