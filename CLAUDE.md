## Architecture rules (follow exactly)

- ADR-0001: Agent calls predefined Tool functions only — never generates or executes code
- ADR-0002: Tool output is aggregate-only (column stats, counts, schema) — no raw rows ever
- LLM client is always injected (never constructed inside the graph) — this is the test seam
- Integration tests use FakeLLMClient only, no real API calls
- Internal message format is Anthropic-style (tool_use / tool_result blocks)
- All 5 report sections (Executive Summary, Data Quality Scorecard, Distributions, Correlations, Feature Engineering Recommendations) must always be present in output
- The LLM system prompt in `agent/graph.py` is **auto-generated** from the `tools` list passed to `build_graph` via `_build_system_prompt(tools)`. Do not hard-code tool names in the prompt string. When adding a new Tool, register it in `_DEFAULT_TOOLS` in `cli.py` — the prompt sequence updates automatically.
- New tools go in `src/ds_agent/tools/` following the Tool ABC in `tools/base.py`
- New graph nodes get wired into `agent/graph.py` via `build_graph()`
- Report section population goes in `report/renderer.py`
- The uncertainty mechanism (`uncertainty.py`) is generic, keyed by trigger type + documented default — new uncertainty triggers plug into it rather than inventing their own pause/flag logic
- Chart files go in a `charts/` subdirectory of output_dir; reference them in the report via relative markdown image links

## Issue implementation workflow

When implementing issues, proceed autonomously all the way to completion — do not pause to ask permission or confirm next steps unless something genuinely requires manual approval (e.g. a destructive action or an ambiguous external dependency). Use default recommendations for any design decisions not specified in the issue. Check in only at the end with a summary of what was done.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (`KyleNaluan/data-science-agent`) via the `gh` CLI. External PRs are not treated as a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) with no remapping. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
