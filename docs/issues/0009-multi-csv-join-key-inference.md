## What to build

Extend ingestion to accept multiple CSV file paths. A join-key-inference Tool scores candidate join keys across the supplied files by column-name match, dtype compatibility, and cardinality/uniqueness overlap. High-confidence matches join automatically; ambiguous or multi-candidate cases raise an Uncertainty trigger ("ambiguous join key") through the Slice 2 mechanism. Once joined, the rest of the pipeline (Slices 1–8) operates on the resulting single dataframe, producing one unified report.

## Acceptance criteria

- [ ] Join-key-inference Tool is unit-tested against fixture file pairs with known join keys (clear match, ambiguous match, no match) and produces the expected confidence classification for each
- [ ] A run against two CSVs with an unambiguous shared key joins automatically and produces a single unified report
- [ ] A run against two CSVs with an ambiguous or multi-candidate key raises the checkpoint/flagged-assumption mechanism, naming the candidate keys considered
- [ ] Single-CSV runs (Slice 1 behavior) are unaffected by this change
- [ ] Tool output remains aggregate-only (key candidates and confidence scores) — never row-level join previews

## Blocked by

- Slice 2: Uncertainty mechanism — ambiguous column type + tiny dataset
