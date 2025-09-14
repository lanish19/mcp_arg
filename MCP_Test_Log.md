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
