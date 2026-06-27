# PRD: V1 — EDA + Report Agent

## Problem Statement

Data analysts and data science teams spend a disproportionate amount of time on repetitive, manual exploratory data analysis — checking column types, profiling distributions, hunting for outliers, building correlation matrices, and writing up what they found — before they can get to the question they actually care about. The depth and rigor of this work varies by who's doing it and how much time they have, and ambiguous data (mixed types, high missingness, tiny samples) often gets silently glossed over rather than flagged. Both ad-hoc analysts and automated pipelines need a fast, trustworthy first pass at a new CSV — one that produces an actual report rather than a notebook full of half-finished cells, and that surfaces uncertainty instead of hiding it.

## Solution

An agent that takes a CSV (or several joinable CSVs) and autonomously runs schema inference, distribution analysis, outlier detection, correlation analysis, and missing-value analysis, then produces a narrative markdown report with embedded charts and a data-quality scorecard. The agent reasons step by step, deciding what to check next based on what it has already learned. When it hits a genuinely ambiguous decision, it pauses and asks if a person is watching (interactive session), or proceeds with a clearly flagged default and records the assumption in the report if no one is (automated pipeline). V1 ships as a CLI tool, designed so a future service wrapper is a thin addition rather than a rewrite; ML modeling and conversational Q&A are later versions.

## User Stories

1. As an ad-hoc analyst, I want to point the agent at a single CSV file, so that I can get an EDA report without writing any code.
2. As an ad-hoc analyst, I want to point the agent at multiple joinable CSVs, so that I can analyze data that's split across related files.
3. As a data science power user, I want to supply an optional schema/metadata file alongside my CSV, so that the agent doesn't have to guess at column semantics I already know.
4. As an automated pipeline operator, I want the agent invocable as a single CLI command with file paths as arguments, so that I can wire it into existing scripts or cron jobs.
5. As an ad-hoc analyst, I want the agent to infer join keys automatically across multiple CSVs, so that I don't have to manually specify foreign keys for a quick analysis.
6. As a data science power user, I want to be asked to confirm an inferred join key when the agent isn't confident, so that I don't get silently wrong results from a bad join guess.
7. As a pipeline operator, I want an ambiguous join key to fall back to a documented default with a clear flag in the report, so that my unattended run still completes.
8. As an ad-hoc analyst, I want the agent to infer each column's type (numeric, categorical, datetime, identifier, etc.), so that I don't have to manually annotate every column.
9. As a data science power user, I want to see why a column's type was ambiguous (e.g. numeric ID vs. measurement), so that I can correct it if the agent guessed wrong.
10. As an ad-hoc analyst, I want ambiguous column types to trigger a checkpoint when I'm running interactively, so that I can clarify intent before the rest of the EDA runs on a bad assumption.
11. As a pipeline operator, I want ambiguous column types to fall back to a flagged default when running unattended, so that the pipeline doesn't hang waiting for input that will never come.
12. As an ad-hoc analyst, I want histograms and summary statistics (skew, kurtosis) for numeric columns, so that I can understand the shape of my data at a glance.
13. As a non-technical report reader, I want distribution findings described in plain language, not just charts and numbers, so that I can understand what's notable without a statistics background.
14. As a data science power user, I want outliers flagged using both IQR and z-score methods, so that I can cross-check which points are genuinely anomalous.
15. As an ad-hoc analyst, I want outlier plots saved as chart artifacts, so that I can visually inspect what the agent flagged.
16. As a data science power user, I want a full correlation matrix plus a callout of the most notable pairwise relationships, so that I don't have to scan an NxN matrix myself for the columns that matter.
17. As an ad-hoc analyst, I want a missing-value breakdown per column with imputation recommendations, so that I know which columns need cleanup before I model on them.
18. As a pipeline operator, I want a missing-value rate above a configurable threshold to go through the same checkpoint/flagged-assumption mechanism as other uncertainty triggers, so that high-missingness columns don't get silently treated as fine.
19. As a data science power user, I want candidate columns for encoding, binning, scaling, or interaction terms surfaced as recommendations, so that I have a head start on feature engineering without the agent changing my data for me.
20. As an ad-hoc analyst, I want feature suggestions explained in terms of what the EDA just found (e.g. "this column is highly skewed, consider a log transform"), so that the recommendation is grounded in evidence I can see.
21. As an ad-hoc analyst running interactively, I want the agent to pause and ask me a direct question when it hits a genuinely ambiguous decision, so that I stay in control of consequential calls.
22. As a pipeline operator, I want every checkpoint to have a documented, safe default it falls back to when no one is watching, so that automated runs always complete.
23. As any user, I want every flagged assumption to be visible in the report's data-quality section, so that I know what to double-check even after the fact.
24. As a data science power user, I want the uncertainty thresholds (missing-value %, minimum dataset size, etc.) to be configurable, so that I can tune sensitivity to my own risk tolerance.
25. As a pipeline operator, I want to override specific thresholds via CLI flags for a one-off run, so that I don't have to edit a checked-in config file for a single exception.
26. As a non-technical stakeholder, I want a markdown report with a clear executive summary, so that I can get the headline findings without reading the whole document.
27. As any user, I want every report to follow the same fixed section structure, so that I know where to look for a given kind of finding regardless of dataset.
28. As a data science power user, I want a data-quality scorecard as its own section, so that I can quickly assess whether a dataset is trustworthy before investing more time in it.
29. As an ad-hoc analyst, I want charts saved as PNG files alongside the markdown report, so that I can drop them into a slide deck or other document directly.
30. As a data science power user, I want the report to reference its charts via relative paths, so that the report folder is portable as a single self-contained directory.
31. As a data science power user, I want a structured trace log of every reasoning step and tool call the agent made, so that I can debug a surprising or wrong-looking result.
32. As a pipeline operator, I want the trace log to never contain row-level data, so that I can retain it for debugging without taking on additional data-handling risk.
33. As any user, I want the agent to always produce either a complete report or a clear error, so that a failed run never looks like a silent success.
34. As a data science power user, I want assurance that the LLM never sees raw rows of my data, only computed aggregates, so that I can run this on sensitive datasets with lower risk.
35. As a pipeline operator, I want the agent's tool layer to be the only thing that touches raw data, so that I can audit exactly what computation ran without having to trust the LLM's behavior.
36. As a developer extending the agent, I want EDA capabilities exposed as individually callable, testable tool functions, so that I can validate correctness against fixture datasets without invoking the LLM.
37. As a developer testing the agent end-to-end, I want to substitute a fake LLM client while keeping every tool function and the report assembly real, so that I can write deterministic integration tests for the orchestration logic.
38. As an ad-hoc analyst, I want to run the whole thing as a single CLI command with sane defaults, so that I don't need to write a config file just to try it out.
39. As a power user, I want a checked-in config file for threshold defaults, so that repeated runs across a team are consistent and version-controlled.
40. As a future API integrator, I want the core agent logic decoupled from CLI I/O, so that wrapping it in a service later doesn't require rewriting the reasoning loop.
41. As an ad-hoc analyst, I want the full run to complete in under 3 minutes for datasets up to 100k rows, so that I get my report back fast enough to stay in flow.

## Implementation Decisions

- **Agent orchestration**: A LangGraph state machine implements the Reason → Act → Observe loop: Ingest & Validate → Schema Inference → EDA Loop → (Uncertainty trigger? → Human checkpoint or Flagged assumption) → Feature Suggestion Generation → Report & Chart Assembly → Output Delivery. Edges can loop back — e.g. re-running EDA after a checkpoint resolves a column-type ambiguity.
- **Execution model (ADR-0001)**: The agent calls a fixed set of predefined Tools rather than generating and executing arbitrary code. The V1 tool surface covers schema inference, distribution analysis, outlier detection (IQR + z-score), correlation analysis, missing-value analysis, feature-engineering suggestion generation, and join-key inference. Each tool is a pure function over a pandas DataFrame, callable directly without going through the LLM.
- **Data exposure (ADR-0002)**: Every tool's return value is restricted to column-level aggregates (schema, summary stats, value counts, correlation values, outlier counts/indices) — never a raw row. This is enforced at the tool-output layer, not left to LLM discretion.
- **Uncertainty handling**: One mechanism handles every Uncertainty trigger (ambiguous column type, missing-value rate over threshold, dataset too small, conflicting schema hints, ambiguous or low-confidence join key). Interactive session (TTY detected) → Human checkpoint: pause, surface the question, wait for an answer, resume (potentially re-entering an earlier graph node). Non-interactive → Flagged assumption: proceed using a documented per-trigger default, record it in the Data Quality Scorecard.
- **Threshold configuration**: Defaults for every uncertainty trigger (missing-value rate, minimum dataset size, join-confidence threshold, etc.) live in a checked-in config file, falling back to built-in defaults if absent. Individual thresholds can be overridden via CLI flags for a single run.
- **Join logic**: A join-key-inference tool scores candidate keys by column-name match, dtype compatibility, and cardinality/uniqueness overlap. High-confidence matches join automatically; low-confidence or multi-candidate cases route through the standard uncertainty-trigger mechanism.
- **Report assembly**: Fixed section template — Executive Summary, Data Quality Scorecard, Distributions, Correlations, Feature Engineering Recommendations — rendered as markdown with relative-path image references to chart PNGs saved in a `charts/` subdirectory alongside the report. The LLM writes narrative prose within each fixed section; it does not choose which sections exist.
- **Observability**: Every Reason/Act/Observe step is appended to a local structured trace log (JSONL) per run, saved alongside the report. Each entry captures the reasoning text, the tool name and arguments, and the tool's (aggregate-only) output. No third-party tracing service (e.g. LangSmith) in V1.
- **Interface**: V1 ships as a CLI only. Core orchestration (graph, tools, report assembly) is decoupled from CLI-specific I/O (argument parsing, stdin prompting), so a future service wrapper can reuse it directly — replacing the interactive-prompt checkpoint with a pause/resume state without touching the graph itself.
- **LLM client boundary**: The orchestration layer depends on the LLM client through an injectable interface. This boundary is also the test seam (see Testing Decisions).
- **Glossary terms in play** (defined in `CONTEXT.md`): Tool, Uncertainty trigger, Human checkpoint, Flagged assumption.

## Testing Decisions

- **What makes a good test here**: assert on external behavior — report structure/contents, chart files existing and being referenced, trace-log structure, and numeric correctness of tool outputs against known fixtures. Never assert on exact LLM-generated prose; narrative text is inherently non-deterministic even with every upstream input held fixed.
- **Tool-layer tests**: Each tool (schema inference, outlier detection, correlation, missing-value analysis, join-key inference, feature-suggestion generation) gets direct unit tests against fixture DataFrames with known, hand-verified properties (a deliberately ambiguous numeric-ID column, a column at exactly the default missingness threshold, etc.). These run with no LLM involved and are what the V1 success metrics (≥95% type-inference accuracy, ≥90% uncertainty-flag recall) get benchmarked against.
- **Orchestration/integration tests**: Drive the full agent entry point with a fake, scripted LLM client (the single confirmed seam) returning deterministic tool-call sequences, while every tool, the LangGraph graph, and report/chart assembly run for real against fixture CSVs. Assertions check structural properties: required sections present, referenced chart files exist on disk, one trace-log entry per tool call, and a deliberately ambiguous fixture produces the expected Human checkpoint (interactive fake) or Flagged assumption (non-interactive fake), surfaced in the Data Quality Scorecard.
- **Checkpoint/flagged-assumption coverage**: At least one fixture per uncertainty-trigger type, exercised through both the interactive and non-interactive path, to validate the fallback behavior this PRD specifies.
- **Prior art**: None yet — this is the first PRD for the repo. It establishes the pattern (fake only the LLM client; everything else real) that subsequent modules should follow.

## Out of Scope

- ML model training/selection (V2)
- Natural language Q&A / conversational follow-up (V2)
- Automated, agent-applied feature transformation — V1 only surfaces recommendations (V2)
- Non-CSV inputs: databases, Parquet, APIs (V3+)
- Jupyter notebook export (V3)
- Multi-run comparison / longitudinal dataset tracking (V4)
- A FastAPI (or other) service interface — V1 only designs for one, it doesn't build one
- LangSmith or other third-party observability integration
- Base64-embedded HTML report export
- Any code-generation or arbitrary-code-execution path for the agent (explicitly rejected for V1 per ADR-0001)

## Further Notes

- All open questions from the original concept doc (`ds_agent_concept.md`) were resolved in a grilling session prior to this PRD. Decisions are recorded in `CONTEXT.md` (glossary) and `docs/adr/0001-fixed-tool-calling-for-v1.md` / `docs/adr/0002-aggregates-only-tool-outputs.md`.
- The throughline across every resolved decision: ambiguity is handled by one reusable mechanism (Uncertainty trigger → Human checkpoint or Flagged assumption) rather than bespoke logic per feature, and every fork was resolved in favor of determinism/testability over day-one flexibility. Later PRDs should default to extending that mechanism rather than inventing a new one.
- The V1 success metrics from the concept doc (end-to-end runtime < 3 minutes for 1k–100k rows, ≥ 4/5 usefulness rating, ≥ 95% type-inference accuracy, ≥ 90% uncertainty-flag recall, zero silent failures) still apply and should inform acceptance criteria for implementation tickets that follow from this PRD.
