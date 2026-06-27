# Data Science Agent — Concept Document
> **Status:** Pre-PRD ideation | **Next step:** PRD grill session in Claude Code

---

## Vision

An agentic AI system that autonomously conducts end-to-end data science workflows — from raw CSV ingestion through EDA, feature engineering, ML modeling, and natural language Q&A — delivering narrative reports and visualizations to both technical and non-technical users. The agent reasons step-by-step, deciding what to do next based on what it just learned, while flagging uncertainty and deferring to the user when confidence is low.

---

## Core Design Principles

1. **Reactive reasoning over rigid pipelines** — the agent decides its next action based on the result of the last one, not a fixed script.
2. **Hybrid autonomy** — runs autonomously by default; surfaces decision points when data is ambiguous, quality is poor, or multiple valid paths exist.
3. **General-purpose** — serves ad-hoc analysts, automated pipelines, and data science power users equally.
4. **Output-first thinking** — every agent action should be traceable to a user-facing output (report, chart, or answer).

---

## V1 Scope: EDA + Report

V1 is deliberately constrained to prove the reasoning loop and output quality before adding ML and Q&A complexity.

### V1 Capabilities
- **CSV ingestion** — single file, multiple joinable CSVs, or CSV + schema/metadata file
- **Automated EDA**
  - Schema inference (column types, cardinality, nullity)
  - Distribution analysis (histograms, skew, kurtosis)
  - Outlier detection (IQR, z-score)
  - Correlation matrix and notable pairwise relationships
  - Missing value analysis and imputation recommendations
- **Feature engineering suggestions** — identify candidates for encoding, binning, scaling, or interaction terms (surfaced as recommendations, not automatically applied in V1)
- **Narrative report generation** — markdown output with embedded chart references, plain-language insight summaries, and a data quality scorecard
- **Visualization artifacts** — static charts (distributions, correlations, outlier plots) saved alongside the report

### V1 Out of Scope
- ML model training and selection (V2)
- Natural language Q&A / conversational follow-up (V2)
- Automated feature transformation (V2)
- Non-CSV inputs (V3+)

---

## Post-V1 Roadmap

| Version | Key Addition |
|---------|-------------|
| V2 | ML model selection & training; auto feature engineering |
| V2 | Natural language Q&A over the dataset and report |
| V3 | Jupyter notebook export as a reference artifact |
| V3 | Non-CSV inputs (databases, Parquet, APIs) |
| V4 | Multi-run comparison; longitudinal dataset tracking |

---

## Agent Architecture

### Reasoning Pattern: ReAct Loop (Reason → Act → Observe)

The agent follows a step-by-step reasoning cycle rather than planning the full workflow upfront:

```
User Input (CSV + optional goal/context)
        ↓
  [REASON] What do I know so far? What should I do next?
        ↓
  [ACT]    Execute a tool (load data, run EDA step, generate chart...)
        ↓
  [OBSERVE] What did I learn? Did anything unexpected happen?
        ↓
  [REASON] Do I have enough to move on, or do I need to flag something?
        ↓
  ... repeat until report is complete
```

**Uncertainty flagging triggers:**
- Missing value rate > configurable threshold (default 20%)
- Ambiguous column types (e.g. numeric IDs, mixed-type columns)
- Dataset too small for statistical conclusions
- Conflicting schema hints between file and metadata

### LangGraph State Machine (Conceptual)

```
[Ingest & Validate] → [Schema Inference] → [EDA Loop]
                                                ↓
                                     [Flag? → Human checkpoint]
                                                ↓
                               [Feature Suggestion Generation]
                                                ↓
                                    [Report & Chart Assembly]
                                                ↓
                                          [Output Delivery]
```

Each node in the graph is a LangGraph state; edges can loop back (e.g. re-run EDA after user clarifies a column type).

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Agent framework | LangGraph (LangChain ecosystem) | Native support for stateful, cyclic reasoning graphs |
| LLM | Claude (Anthropic API) | Strong reasoning, instruction-following, and long-context for large CSVs |
| Data manipulation | pandas + numpy | Standard; well-understood by LLM code generation |
| Visualization | matplotlib / seaborn | Static chart generation; no runtime JS dependency |
| Report generation | Markdown → optional PDF via `weasyprint` or `pandoc` | Portable, easy to extend |
| Notebook export | `nbformat` | Programmatic Jupyter notebook generation |
| Packaging | Python 3.11+, `uv` or `poetry` | Modern dependency management |

### Interface (TBD — candidates for PRD decision)
- **Local CLI** — lowest friction for V1, good for developer testing
- **FastAPI service** — enables future UI, webhook triggers, pipeline integration
- **Recommendation:** start with CLI for V1, design with FastAPI in mind so the transition is a thin wrapper

---

## Key Open Questions for PRD

These are the questions the PRD grill session should resolve:

1. **Uncertainty threshold configuration** — hardcoded defaults or user-configurable per run? Stored in a config file?
2. **Human checkpoint UX** — in CLI: interactive prompt; in API: pause and return a `requires_input` state. How does this affect async pipeline use cases?
3. **Chart format and storage** — PNG alongside the report, base64-embedded in HTML, or both?
4. **LLM code execution model** — does the agent *generate* Python and execute it (code-gen loop), or does it *call* predefined Python tools? Hybrid?
5. **Input join logic** — for multiple CSVs, does the agent infer join keys automatically or require the user to specify?
6. **Report structure** — fixed sections (executive summary → data quality → distributions → correlations → recommendations) or LLM-determined structure per dataset?
7. **Observability** — how do we log the agent's reasoning steps for debugging? LangSmith integration?
8. **Security / data privacy** — are CSVs sent to the LLM in full, or only metadata + samples? What's the PII handling story?

---

## Success Metrics for V1

| Metric | Target |
|--------|--------|
| End-to-end run time (1k–100k row CSV) | < 3 minutes |
| Report usefulness (user rating) | ≥ 4/5 in internal testing |
| Correct column type inference | ≥ 95% on benchmark datasets |
| Uncertainty flags correctly triggered | ≥ 90% recall on known edge cases |
| Zero silent failures | Agent must always produce output or a clear error |

---

## Glossary

- **ReAct loop** — Reason, Act, Observe cycle; the agent's core reasoning pattern
- **EDA** — Exploratory Data Analysis
- **Human checkpoint** — a point where the agent pauses and surfaces a question or decision to the user
- **State machine** — LangGraph's model of the agent as a directed graph of states and transitions
- **Feature suggestion** — a recommendation to transform a column (e.g. one-hot encode, log-scale); not automatically applied in V1
