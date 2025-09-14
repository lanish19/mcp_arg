# TODO2.md — Holistic Fix Plan (Post-Stepwise + Envelope)

Objective: Close remaining accuracy, traceability, and orchestration gaps so AIs can reliably chain tools, get high-signal artifacts, and render exportable graphs with clear weaknesses, assumptions, and probe plans.

Status snapshot (already done):
- Response envelopes (v1.1.0) with metadata + basic logging
- Stepwise orchestration with envelope unwrapping and next_tools
- Compact nodes/links/patterns; ontology sanitization + synonyms; slug tools
- Three new endpoints: map_assumptions_to_nodes, export_graph, analyze_and_probe
- Validation extras; quality rubric (sub-scores)

Priority Phases

## Phase A — Signal Fidelity & Traceability (High ROI)

1) Always-populated pattern spans
- Files: `MCP_example_template/arg_mcp/patterns.py`, `MCP_example_template/arg_mcp/engine.py`
- Edits:
  - In `EnhancedPatternDetector.detect`, when no explicit trigger phrase is found:
    - Compute a fallback span using highest-weight token/bi-gram from TF-IDF vector for the top row.
    - Set `span=(start,end)` for every pattern.
  - In `AnalysisEngine.stage2_patterns`, preserve `span` as `source_text_span` on each pattern dict (array form `[start,end]`).
- Acceptance:
  - 100% of returned patterns have non-null `source_text_span` aligned with input text.
  - Off-by-one free in tests for multiple positions.

2) Link weaknesses to nodes (span-overlap mapping)
- Files: `MCP_example_template/argument_mcp.py`
- Edits:
  - In `identify_reasoning_weaknesses`, for each pattern, map to nodes whose `source_text_span` overlaps the pattern span.
  - Emit weaknesses objects: `{label, why, how, node_ids:[...], span:[start,end]}`.
  - Keep `sensitivity` filtering; default cutoff unchanged.
- Acceptance:
  - Each weakness includes `node_ids` (non-empty when spans available).
  - Weakness count stable vs prior; more informative payload.

3) Rich assumption objects
- Files: `MCP_example_template/arg_mcp/engine.py`, `MCP_example_template/argument_mcp.py`
- Edits:
  - In `stage3_infer`, emit richer assumptions:
    - `{text, linked_patterns:[pattern_id], impact:"high|med|low", confidence:float, tests:[...]}` (deterministic heuristics)
  - In `generate_missing_assumptions`, return the richer objects; keep envelope and warnings.
- Acceptance:
  - Non-empty `assumptions` for the geopolitical sample; objects include `linked_patterns` and at least one `tests` suggestion.

4) Validation coverage upgrades
- Files: `MCP_example_template/argument_mcp.py`, `MCP_example_template/arg_mcp/engine.py`
- Edits:
  - Add duplicate-edge detection and graph cycle detection (simple DFS) in `_ENGINE._validate_graph` or wrapper.
  - Add checks: `no_evidence_types`, `no_assumptions`, `missing_spans` (pattern or nodes), `no_links`.
  - Include concrete `next_steps` (e.g., `map_assumptions_to_nodes`, `export_graph`).
- Acceptance:
  - For sample, validator reports at least one of the above when applicable; suggestions recommend next actions.

## Phase B — Probe Planning Depth

5) Thematic probe rulelets
- Files: `MCP_example_template/arg_mcp/probes.py`, `MCP_example_template/argument_mcp.py`
- Edits:
  - If limited-options cues found ("three ways", "either/or"), include dilemma-focused probes.
  - If phrases like "frozen conflict" or "leadership vacuum" detected, add stability/conflict probes.
  - Attach `why`, `how`, and `targets` (top-5 node_ids) to each step.
  - Keep min 3 / max 5 probes.
- Acceptance:
  - For sample, probe plan contains at least 3, context-relevant probes with `why/how/targets` fields populated.

6) Tool catalog tagging (phase/theme/requires)
- Files: `MCP_example_template/arg_mcp/ontology.py` (ToolCatalog), `argument_tools.csv`
- Edits:
  - Add derived tags (phase, theme) during load (lightweight heuristics); extend `tools_search` to filter by tags any/all via query syntax: `tags:any:causal,assumption`.
- Acceptance:
  - `tools_search` can filter by tags; returned objects include `tags` array.

## Phase C — API & Data Hygiene

7) JSON Schemas coverage (req/resp) for all endpoints
- Files: `schemas/v1/*`
- Edits:
  - Add response schemas for remaining endpoints: `detect_argument_patterns`, `ontology_*`, `construct_argument_graph`, `validate_argument_graph`, `assess_argument_quality`, `identify_reasoning_weaknesses`, `generate_counter_analysis`, `map_assumptions_to_nodes`, `analyze_and_probe`.
  - Add request schemas mirrors where relevant.
  - Factor common types: `Link`, `Pattern`, `Weakness`, `Assumption`, `ProbeStep` in `schemas/v1/common/`.
- Acceptance:
  - All endpoints reference a schema_url; schemas validate golden outputs in tests.

8) Error model consistency
- Files: `MCP_example_template/argument_mcp.py`
- Edits:
  - Ensure envelope error codes across all tools: `INVALID_INPUT_SHAPE`, `MISSING_ARGUMENT_TEXT`, `TOOL_NOT_FOUND`, `UNSUPPORTED_FORMAT`, `INTERNAL_ERROR`.
  - Example: `tools_get` returns envelope error with `TOOL_NOT_FOUND` when missing.
- Acceptance:
  - Negative-path tests assert error.code and hint/where fields.

9) Ontology hygiene & list endpoints
- Files: `MCP_example_template/arg_mcp/ontology.py`, `argument_mcp.py`
- Edits:
  - Expand `SYNONYMS`; return `metadata.applied_synonyms` consistently.
  - `ontology_list_*` include counts and parent references where applicable.
- Acceptance:
  - Searching "false trilemma" reliably maps to False Dilemma; list endpoints render objects with counts.

## Phase D — Exports & Mapping

10) Assumption mapping strategy options
- Files: `MCP_example_template/argument_mcp.py`
- Edits:
  - `map_assumptions_to_nodes` add `strategy:"best-match|strict"`; strict requires score ≥ threshold; include `unmapped` rationale.
- Acceptance:
  - Both strategies covered in tests; strict leaves low-similarity items in `unmapped` with reason.

11) Graph export polish
- Files: `MCP_example_template/argument_mcp.py`
- Edits:
  - Mermaid/Graphviz: color nodes by `primary_subtype`; truncate labels safely; escape characters robustly.
  - JSON-LD: add `@type` on nodes/links; include minimal context.
- Acceptance:
  - Visuals render in common viewers; sample produces readable diagram with claim/premise contrasts.

## Phase E — Tests, Goldens, Perf

12) Golden integration for geopolitical sample
- Files: `tests/test_integration_geopolitics.py`
- Assertions:
  - Non-empty `assumptions` (include dilemma/causal necessities);
  - ≥3 weaknesses with `node_ids` and spans; 
  - ≥3 probes with `why/how/targets`.
  - `export_graph(mermaid)` includes main claim node id.

13) Fuzz/error tests
- Files: `tests/test_errors.py`
- Cases: missing text; wrong types; unsupported export format; unknown tool name; ensure structured errors.

14) Performance guardrails
- Files: `argument_mcp.py`, config constants
- Edits:
  - Centralize thresholds (pattern top-N, ontology threshold); log `input_size` and `result_size`; ensure typical run under budget.
- Acceptance:
  - Logs show stable elapsed_ms and capped payload sizes.

Migration & Docs

15) Docs update
- Files: `README.md`, `docs/CONTRACTS.md` (new)
- Edits:
  - Document envelope, schemas, new endpoints, error codes; add walkthrough (text → decompose → assumptions → map → weaknesses → probe → export).

16) Migration notes
- Files: `docs/MIGRATION.md` (new)
- Edits:
  - Note envelopes as default; legacy clients must unwrap `data` and check `error`.

Implementation Notes
- Deterministic only; no randomness.
- Keep payloads compact; truncate strings ~200 chars where needed.
- Do not break existing tool names; add fields additively.

Ownership Map (primary modules)
- Spans/Patterns/Inference: `arg_mcp/patterns.py`, `arg_mcp/engine.py`
- Weaknesses/Validation/Exports/Stepwise: `argument_mcp.py`
- Probes/Tools: `arg_mcp/probes.py`, `arg_mcp/ontology.py`
- Schemas/Docs/Tests: `schemas/v1/*`, `README.md`, `tests/*`
