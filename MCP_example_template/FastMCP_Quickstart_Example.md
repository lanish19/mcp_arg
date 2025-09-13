# FastMCP Quickstart Example

This is a quickstart example for FastMCP, automatically set up for you!

## Getting Started

This repository contains two FastMCP servers:

- `echo.py` — minimal echo example
- `argument_mcp.py` — Sequential Argument Reasoning MCP that loads ontology CSVs and exposes query + analysis tools

### Files
- `echo.py` — A simple MCP server that echoes back your messages
- `argument_mcp.py` — Ontology-aware argument reasoning server (multi-stage engine)

#### argument_mcp.py tools
- `ontology_list_dimensions()` — list ontology dimensions
- `ontology_list_categories(dimension)` — list categories for a dimension
- `ontology_list_buckets(dimension?, category?)` — list buckets (e.g., specific schemes, fallacies)
- `ontology_search(query, dimension?, category?, bucket?)` — full-text ontology search
- `ontology_bucket_detail(bucket_name)` — exact-match bucket details
- `tools_list()` — list diagnostic/probe tools from `argument_tools.csv`
- `tools_search(query)` — search probe tool catalog
- `tools_get(name)` — get a probe tool by name
- `analyze_argument_comprehensive(argument_text, forum?, audience?, goal?, analysis_depth?)` — multi-stage analysis returning structured graph, patterns, assumptions, and probe plan
- `decompose_argument_structure(argument_text, include_implicit?)` — structural breakdown with nodes/links and detected patterns
- `detect_argument_patterns(argument_text, pattern_types?)` — detect causal/authority/analogical/etc. patterns with confidence
- `generate_missing_assumptions(argument_components, prioritization?)` — template-based assumption generation
- `orchestrate_probe_analysis(analysis_results, forum?, audience?, goal?)` — dynamic probe selection and chaining
- `ontology_semantic_search(query, dimensions?, similarity_threshold?, max_results?)` — semantic search in ontology
- `ontology_pattern_match(argument_patterns, match_type?)` — map detected patterns to ontology categories
- `construct_argument_graph(analysis_results)` — build graph with adjacency
- `validate_argument_graph(graph, validation_level?)` — structural validation and suggestions
- `compare_arguments(argument_graphs, comparison_dimensions?)` — structural/pattern comparison across graphs
- `assess_argument_quality(argument_graph, assessment_framework?)` — qualitative strengths/weaknesses
- `identify_reasoning_weaknesses(argument_analysis, weakness_categories?)` — fallacy/bias/gap flags with rationales
- `generate_counter_analysis(argument_graph, analysis_type?)` — counter-brief scaffolds

#### argument_mcp.py resources & prompts
- `argument://dimensions` — newline-delimited dimensions
- `argument://buckets/{dimension}` — newline-delimited buckets for a dimension
- `argument://tools` — newline-delimited probe tools
- Prompt `analyze` — returns a Master-Brief-aligned instruction scaffold for LLMs

### Deployment

This repository is ready to be deployed!

- Create a new [FastMCP Cloud account](http://fastmcp.cloud/signup)
- Connect your GitHub account
- Select `Clone our template` and a deployment will be created for you!

### Learn More

- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)

### Notes
- The argument MCP loads `new_argumentation_database_buckets_fixed.csv` and `argument_tools.csv` at startup using UTF‑8 with BOM handling.
- The analyzer uses deterministic multi-stage heuristics (no network calls) with regex-based semantic detection, template gap analysis, and context-aware probe orchestration.

---
*This repository was automatically created from the FastMCP quickstart template.*
