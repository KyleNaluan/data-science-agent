## What to build

A CLI entry point that accepts a single CSV path and runs a complete (if shallow) pass through the full V1 pipeline shape: ingestion/validation, a LangGraph state machine implementing the Reason → Act → Observe loop, a Tool abstraction with the first real Tool (schema inference: column types, cardinality, nullity), an injectable LLM-client interface used by the orchestration layer (with a fake/scripted implementation usable in tests), a fixed-section report renderer (Executive Summary, Data Quality Scorecard, Distributions, Correlations, Feature Engineering Recommendations — sections beyond Data Quality Scorecard rendered as explicit placeholders until later slices populate them), and a local structured JSONL trace log capturing every Reason/Act/Observe step (reasoning text, tool name + args, tool output) written alongside the report.

This slice establishes the architecture every later slice builds on: the agent calls predefined Tools rather than generating code (ADR-0001), Tool outputs are restricted to column-level aggregates only and never row-level data (ADR-0002), and core orchestration logic stays decoupled from CLI-specific I/O so a future service wrapper is additive rather than a rewrite.

## Acceptance criteria

- [ ] Running the CLI against a single CSV produces a markdown report containing all 5 fixed sections, with the Data Quality Scorecard populated from real schema-inference output and the remaining sections present as explicit placeholders
- [ ] The schema-inference Tool is callable and unit-testable directly with no LLM involved, hitting ≥95% type-inference accuracy on a benchmark fixture set
- [ ] Tool output contains only column-level aggregates (schema, types, cardinality, nullity counts) — never a raw row
- [ ] A local JSONL trace log is written alongside the report with one entry per Reason/Act/Observe step
- [ ] The orchestration layer accepts the LLM client as an injected dependency; an integration test drives the full pipeline with a fake/scripted LLM client while the schema-inference Tool and report renderer run for real
- [ ] CLI invocation completes end-to-end against a 1k–100k row fixture in a reasonable time (rough sanity check; tight performance tuning is not required yet)

## Blocked by

None - can start immediately.
