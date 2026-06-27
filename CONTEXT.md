# Data Science Agent

An agentic system that runs end-to-end data science workflows over CSV inputs — EDA, modeling, and reporting — reasoning step-by-step rather than executing a fixed pipeline.

## Language

**Tool**:
A predefined Python function with structured arguments that the agent invokes during the Act phase of its reasoning loop to perform a concrete operation (e.g. compute a correlation matrix, detect outliers). The agent selects which tool to call and with what arguments; it does not generate or execute arbitrary code in V1. Tool outputs are restricted to column-level aggregates (schema, stats, value counts) — never row-level data that ties multiple fields together (see ADR-0002).
_Avoid_: Action (that's the loop phase, not the unit of execution), code-gen, free-form code execution

**Uncertainty trigger**:
A condition detected during EDA (e.g. missing-value rate above threshold, ambiguous column type, dataset too small for statistical conclusions) that requires either a human checkpoint or a flagged assumption before the agent proceeds.

**Human checkpoint**:
An interactive pause where the agent surfaces an uncertainty trigger as a question and waits for the user's answer before proceeding. Only occurs in interactive (TTY) sessions.
_Avoid_: using "checkpoint" for the non-interactive case — that's a flagged assumption, not a checkpoint.

**Flagged assumption**:
The default decision the agent makes when an uncertainty trigger fires during a non-interactive run. The agent proceeds using this default and records it in the report's data-quality/uncertainty section for later human review, rather than pausing.
_Avoid_: silent default, auto-resolution
