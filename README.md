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
- `analyze_argument_comprehensive(argument_text, forum?, audience?, goal?, analysis_depth?)` — Complete argument analysis
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
