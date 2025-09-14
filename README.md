# FastMCP Argumentation Analysis Tool

A comprehensive Model Context Protocol (MCP) server for advanced argumentation analysis, reasoning pattern detection, and structured argument evaluation.

## 🚀 Overview

This repository contains a sophisticated FastMCP server implementation that provides powerful tools for analyzing arguments, detecting reasoning patterns, identifying logical gaps, and generating comprehensive argument assessments. The system combines ontology-driven analysis with multi-stage reasoning engines to deliver structured insights into argument quality and structure.

## 📁 Project Structure

```
├── MCP_example_template/          # Main MCP server implementation
│   ├── argument_mcp.py           # Core argumentation analysis server
│   ├── echo.py                   # Simple echo server example
│   ├── arg_mcp/                  # Modular argumentation components
│   │   ├── engine.py            # Multi-stage analysis engine
│   │   ├── ontology.py          # Ontology management and search
│   │   ├── patterns.py          # Pattern detection algorithms
│   │   ├── structures.py        # Argument structure analysis
│   │   ├── probes.py            # Diagnostic probe tools
│   │   ├── gap.py               # Dynamic inference engine for gaps
│   │   ├── domain_profiles.py   # Domain adaptation profiles
│   │   └── validation.py        # Factual, sentiment, plausibility checks
│   └── FastMCP_Quickstart_Example.md  # Detailed documentation
├── argument_tools.csv            # Diagnostic probe tools catalog
├── new_argumentation_database_buckets_fixed.csv  # Ontology database
└── README.md                     # This file
```

## 🛠️ Core Features

### Argumentation Analysis Tools

- **Comprehensive Analysis**: Multi-stage argument breakdown with structured graph generation
- **Pattern Detection**: Advanced detection of causal, authority, analogical, and other reasoning patterns
- **Gap Analysis**: Dynamic inference rule checks generating scheme-specific missing premises
- **Quality Assessment**: Framework-based evaluation of argument strengths and weaknesses
- **Domain Adaptation**: Legal, scientific, policy profiles that adjust requirement weighting
- **Validation**: Contradiction, sentiment, and plausibility checks for claims and assumptions
- **Counter-Analysis**: Generation of counter-argument scaffolds and alternative perspectives

### Ontology Management

- **Semantic Search**: Advanced search capabilities across argumentation dimensions and categories
- **Pattern Matching**: Automatic mapping of detected patterns to ontology categories
- **Structured Data**: Organized knowledge base covering reasoning schemes, fallacies, and argument types

### Diagnostic Tools

- **Probe Orchestration**: Dynamic selection and chaining of diagnostic tools
- **Tool Catalog**: Extensive library of argument analysis probes and diagnostic instruments
- **Context-Aware Analysis**: Forum, audience, and goal-specific analysis customization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- FastMCP framework

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lanish19/mcp_arg.git
cd mcp_arg
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # If available
```

3. Run the MCP server:
```bash
python MCP_example_template/argument_mcp.py
```

## 📋 Available Tools

### Ontology Tools
- `ontology_list_dimensions()` — List all ontology dimensions
- `ontology_list_categories(dimension)` — List categories for a dimension
- `ontology_list_buckets(dimension?, category?)` — List specific schemes and fallacies
- `ontology_search(query, dimension?, category?, bucket?)` — Full-text ontology search
- `ontology_semantic_search(query, dimensions?, similarity_threshold?, max_results?)` — Semantic search
- `ontology_pattern_match(argument_patterns, match_type?)` — Pattern-to-ontology mapping

### Analysis Tools
- `analyze_argument_stepwise(argument_text, steps?, forum?, audience?, goal?, max_steps?)` — Recommended stage-by-stage pipeline with chaining hints
- `analyze_argument_comprehensive(argument_text, forum?, audience?, goal?, analysis_depth?)` — Complete argument analysis (prefer stepwise for chaining)
- `analyze_and_probe(argument_text, analysis_depth?, audience?, goal?)` — One-shot combined analysis + probes
- `decompose_argument_structure(argument_text, include_implicit?)` — Structural breakdown
- `detect_argument_patterns(argument_text, pattern_types?)` — Pattern detection with confidence scores
- `generate_missing_assumptions(argument_components, prioritization?)` — Assumption generation
- `identify_reasoning_weaknesses(argument_analysis, weakness_categories?)` — Weakness identification
- `generate_counter_analysis(argument_graph, analysis_type?)` — Counter-argument generation

### Diagnostic Tools
- `tools_list()` — List available diagnostic probe tools
- `tools_search(query)` — Search probe tool catalog
- `tools_get(name)` — Get specific probe tool details
- `orchestrate_probe_analysis(analysis_results, forum?, audience?, goal?)` — Dynamic probe selection
- `map_assumptions_to_nodes(analysis_results, assumptions[], strategy?)` — Map assumptions to nodes
- `export_graph(graph, format)` — Export as Mermaid, Graphviz, or JSON-LD
### Response Envelope (v1.1.0)

All tool responses are wrapped:
```json
{
  "version": "v1.1.0",
  "data": { /* payload */ },
  "metadata": { "schema_url": "schemas/v1/<endpoint>.response.json", "warnings": [], "next_steps": [] },
  "error": null
}
```

Errors use structured codes (e.g., `INVALID_INPUT_SHAPE`, `UNSUPPORTED_FORMAT`).

### Graph Analysis
- `construct_argument_graph(analysis_results)` — Build argument graphs with adjacency
- `validate_argument_graph(graph, validation_level?)` — Graph validation and suggestions
- `compare_arguments(argument_graphs, comparison_dimensions?)` — Cross-argument comparison
- `assess_argument_quality(argument_graph, assessment_framework?)` — Quality assessment

## 📊 Data Sources

### Ontology Database
The system uses `new_argumentation_database_buckets_fixed.csv` containing:
- Argumentation schemes and patterns
- Logical fallacies and biases
- Reasoning categories and dimensions
- Structured argument types

### Diagnostic Tools
The `argument_tools.csv` file provides:
- Probe tool definitions and descriptions
- Diagnostic methodologies
- Analysis frameworks and templates

## 🔧 Configuration

The MCP server automatically loads CSV files at startup with UTF-8 BOM handling for compatibility. The analyzer uses deterministic multi-stage heuristics without network calls, employing:
- Regex-based semantic detection
- Template-based gap analysis
- Context-aware probe orchestration

## 🌐 Deployment

### FastMCP Cloud Deployment

1. Create a [FastMCP Cloud account](http://fastmcp.cloud/signup)
2. Connect your GitHub account
3. Select "Clone our template" for automatic deployment

### Local Development

Run the server locally for development and testing:
```bash
python MCP_example_template/argument_mcp.py
```

## 📚 Resources & Prompts

### Resource Endpoints
- `argument://dimensions` — Newline-delimited dimensions
- `argument://buckets/{dimension}` — Dimension-specific buckets
- `argument://tools` — Available probe tools

### Analysis Prompts
- `analyze` — Master-Brief-aligned instruction scaffold for LLMs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the argumentation analysis capabilities.

## 📖 Learn More

- [FastMCP Documentation](https://gofastmcp.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Argumentation Theory Resources](https://modelcontextprotocol.io/)

## 📄 License

This project is open source and available under the MIT License.

---

*This repository provides a comprehensive toolkit for argumentation analysis using the Model Context Protocol framework.*

### Recommended multi-step pipeline

Call `analyze_argument_stepwise(argument_text)` to obtain stage-by-stage outputs and chaining hints.

Default steps:
1) decompose_argument_structure
2) detect_argument_patterns
3) ontology_pattern_match
4) generate_missing_assumptions
5) orchestrate_probe_analysis
6) construct_argument_graph
7) validate_argument_graph
8) assess_argument_quality
9) identify_reasoning_weaknesses
10) generate_counter_analysis

Minimal example:
```python
from MCP_example_template.argument_mcp import analyze_argument_stepwise

res = analyze_argument_stepwise("Experts say we should ban X because it causes harm.")
print(len(res["stages"]))
```

Example of a stage object (`stages[0]`):
```json
{
  "name": "decompose_argument_structure",
  "inputs_summary": {"argument_text": "Experts say we should ban X because it causes harm."},
  "key_outputs": {"nodes": 3, "links": 2},
  "next_tools": ["detect_argument_patterns", "ontology_pattern_match"]
}
```

## API Envelope and Schemas (v1.1.0)

All tools return a versioned envelope:

- version: API version string
- data: tool-specific payload (or null on error)
- metadata: includes schema_url, warnings, next_steps, and optional meta
- error: structured error object {code, message, hint?, where?}

Responses refer to JSON Schemas in `schemas/v1/` for validation.

## Error Codes

- INVALID_INPUT_SHAPE: inputs wrong type/shape
- MISSING_ARGUMENT_TEXT: empty or missing argument_text
- TOOL_NOT_FOUND: tool not present in catalog
- UNSUPPORTED_FORMAT: export format not supported
- INTERNAL_ERROR: unexpected runtime error

## New/Updated Endpoints

- decompose_argument_structure: returns structure + inline patterns and next_tools
- detect_argument_patterns: returns compact patterns with source_text_span
- generate_missing_assumptions: returns enriched assumptions (impact, confidence, tests)
- validate_argument_graph: adds duplicate/cycle checks and suggestions
- identify_reasoning_weaknesses: links weaknesses to node_ids via span overlap
- map_assumptions_to_nodes: supports strategy=best-match|strict and unmapped reasons
- export_graph: Mermaid/Graphviz colored by subtype; JSON-LD with @type
- tools_search: supports tag filters via `tags:any:x,y` or `tags:all:x,y`
