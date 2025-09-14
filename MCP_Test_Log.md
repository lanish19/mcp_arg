# MCP Test Log

Date: 2025-09-14

## Goals
- Validate functionality, robustness, and error model of the Argument Reasoning MCP.
- Stress test inputs, confirm data-driven tool selection, and verify exports.
- Log issues and proposed fixes as we go.

## Test Plan
- Functional coverage across endpoints: ontology, tools, analysis, probes, mapping, export.
- Negative-path and error model checks.
- Stress tests with long inputs, unicode, and edge cases.
- Performance and payload sanity (envelope metadata present; sizes reasonable).

## Checklist
- [ ] Ontology list/search + synonyms mapping
- [ ] Tool catalog list/search + semantic ranking
- [ ] Stepwise analysis end-to-end on geopolitical sample
- [ ] Probe orchestration with candidates (freedom to choose)
- [ ] Mapping strategies best-match vs strict
- [ ] Graph export (mermaid, jsonld) and validation
- [ ] Error model: missing text, unknown tool, unsupported export format
- [ ] Stress: long input, unicode, empty

## Findings (incremental)
- Good:
  -
- Issues:
  -
- Fix ideas:
  -

### Findings Update (Initial MCP-in-Cursor checks)
- Good:
  - Endpoints return envelopes with `version`, `data`, and `metadata.schema_url` as designed.
  - Tool and ontology schemas exist and are referenced via `metadata.schema_url`.
- Issues:
  - Cursor MCP wrapper expects array outputs for certain tools (e.g., `ontology_list_dimensions`, `tools_list`), but server returns envelopes → validation error: output is not of type 'array'.
  - Client wrapper rejected unexpected arguments: `count` for `ontology_semantic_search` and `tools_semantic_search` (Pydantic validation).
  - Result: cannot fully exercise endpoints from Cursor until wrapper and server agree on payload shape.
- Fix ideas:
  - Add compatibility endpoints returning raw arrays (no envelope), e.g., `ontology_list_dimensions_raw`, `tools_list_raw`, `ontology_semantic_search_raw`, `tools_semantic_search_raw`.
  - Alternatively, add a `compat=true` param to return bare `data`.
  - Accept common aliases in signatures (e.g., `count` → `max_results`) to match client expectations.
  - Update Cursor tool definitions (mcp.json) to reflect envelope responses or to unwrap `data`.
