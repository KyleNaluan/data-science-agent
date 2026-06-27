## What to build

The Uncertainty trigger → Human checkpoint / Flagged assumption mechanism described in the PRD. When schema inference (Slice 1) flags a column type as ambiguous (e.g. a numeric-looking identifier vs. a true numeric measurement), or when the dataset is below a minimum-row-count threshold, the agent raises an Uncertainty trigger. If the session is interactive (TTY detected), the agent pauses, surfaces the ambiguity as a direct question, waits for the user's answer, and resumes — potentially re-entering the schema-inference graph node with the clarified type. If the session is non-interactive, the agent proceeds using a documented default for that trigger type and records a Flagged assumption entry in the Data Quality Scorecard.

Build this mechanism generically, keyed by trigger type and its documented default — later slices (missing-value thresholds, conflicting schema hints, ambiguous joins) plug into it rather than each inventing their own pause/flag logic.

## Acceptance criteria

- [ ] An interactive run against a fixture with a deliberately ambiguous column type pauses, displays the question, accepts an answer, and resumes with the clarified type reflected in the final report
- [ ] A non-interactive (piped) run against the same fixture completes without blocking, applies the documented default, and records a Flagged assumption entry in the Data Quality Scorecard naming the column and the assumption made
- [ ] A fixture with fewer rows than the minimum-dataset-size default triggers the same checkpoint/flagged-assumption behavior
- [ ] Both the interactive and non-interactive paths are covered by integration tests using a fake LLM client, asserting on the presence/content of the checkpoint prompt or the flagged-assumption report entry — not on exact LLM prose
- [ ] The mechanism is implemented generically (keyed by trigger type and default), not as a one-off special case for column-type ambiguity

## Blocked by

- Slice 1: Foundation — ingest, schema inference, report skeleton
