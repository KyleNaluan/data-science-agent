## What to build

Accept an optional schema/metadata file alongside the CSV, parsed and used as a hint during schema inference (Slice 1). When the supplied schema conflicts with what schema inference would otherwise infer, this is raised as a new Uncertainty trigger ("conflicting schema hints") through the Slice 2 mechanism, rather than silently preferring one source over the other.

## Acceptance criteria

- [ ] A run with a schema file that agrees with inferred types completes without any extra checkpoint/flag, and the report reflects the supplied schema
- [ ] A run with a schema file that conflicts with inferred types (e.g. file says categorical, inference says numeric) raises the checkpoint/flagged-assumption mechanism, naming the specific conflict
- [ ] Schema-file parsing is unit-tested independently of the rest of the pipeline
- [ ] Absence of a schema file is handled identically to current Slice 1 behavior (fully optional input)

## Blocked by

- Slice 2: Uncertainty mechanism — ambiguous column type + tiny dataset
