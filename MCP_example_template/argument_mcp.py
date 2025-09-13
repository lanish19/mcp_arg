"""
Sequential Argument Reasoning MCP — ontology-aware argument tooling.

Enhanced, multi-stage analysis engine with argument structure decomposition, pattern detection, gap analysis, probe orchestration, and integration. Backward compatible with basic ontology browsing tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os

from fastmcp import FastMCP

# Internal modules
from .arg_mcp.ontology import Ontology, ToolCatalog, load_ontology, load_tool_catalog
from .arg_mcp.engine import AnalysisEngine, AnalysisContext
from .arg_mcp.structures import ArgumentNode, ArgumentLink


# ---- Data loading ----


WORKDIR = os.path.dirname(os.path.dirname(__file__))

ONTOLOGY_CSV = os.path.join(WORKDIR, "new_argumentation_database_buckets_fixed.csv")
TOOLS_CSV = os.path.join(WORKDIR, "argument_tools.csv")


# Remove old lightweight classes in favor of arg_mcp modules


# ---- Lightweight analyzer ----


SCHEME_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Argument from Expert Opinion", ["expert", "scientist", "doctor", "professor", "specialist", "authority"]),
    ("Argument from Analogy", ["like", "similar to", "analogous", "as if"]),
    ("Argument from Example", ["for example", "for instance", "e.g.", "such as"]),
    ("Practical Reasoning", ["should", "ought", "best way", "in order to", "so that"]),
    ("Argument from Cause to Effect", ["causes", "leads to", "results in", "because of", "due to"]),
    ("Argument from Effect to Cause", ["therefore the cause", "explains", "due to this cause", "must have caused"]),
    ("Argument from Consequences", ["consequences", "harmful", "beneficial", "bad outcome", "good outcome"]),
    ("Dilemmatic Reasoning", ["either or", "either/or", "two options", "no other choice"]),
]


FALLACY_CUES: List[Tuple[str, List[str]]] = [
    ("False Dilemma", ["either or", "either/or", "only two", "no alternative"]),
    ("Hasty Generalization", ["always", "never", "everyone", "no one"]),
    ("Appeal to Popularity", ["everyone agrees", "most people think", "the majority"]),
    ("Appeal to Authority", ["experts say", "scientists say", "authorities say"]),
]


def detect_keywords(text: str, patterns: List[Tuple[str, List[str]]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    L = text.lower()
    for label, keys in patterns:
        for k in keys:
            if k in L:
                out.append({"label": label, "trigger": k})
                break
    return out


def choose_probes(scheme_hits: List[str], fallacy_hits: List[str], tools: ToolCatalog) -> List[Dict[str, str]]:
    picks: List[str] = []
    # Heuristic mapping from patterns to probes
    for s in scheme_hits:
        if s in ("Argument from Cause to Effect", "Argument from Effect to Cause", "Argument from Consequences"):
            picks.append("The Causal Probe Playbook™")
        if s in ("Practical Reasoning", "Value‑Based Practical Reasoning"):
            picks.append("The Toulmin Scaffold Mapper™")
        if s in ("Argument from Expert Opinion",):
            picks.append("The Contextual Relevance Check™")
    if fallacy_hits:
        picks.append("Devil’s Advocacy Generator™")
        picks.append("Perturbation Test (Adversarial Thinking)™")
    # Always useful baselines
    picks.append("Assumption Audit™ and Key Assumptions Scrubber™")
    picks.append("Triangulation Method Fusion™")
    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for p in picks:
        if p not in seen and tools.get(p):
            ordered.append(p)
            seen.add(p)
    return [tools.get(p) for p in ordered if tools.get(p)]


def build_outline_map(claim: str, findings: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Claim: {claim}")
    lines.append("")
    lines.append("Supports (convergent/linked):")
    for i, s in enumerate(findings.get("candidate_schemes", []), 1):
        lines.append(f"- Scheme {i}: {s['name']} — {s['rationale']}")
    lines.append("")
    lines.append("Potential Attacks:")
    for i, f in enumerate(findings.get("vulnerabilities", []), 1):
        lines.append(f"- {f['label']} — triggered by “{f['trigger']}”")
    lines.append("")
    lines.append("Assumptions (linchpin candidates):")
    for a in findings.get("assumptions", []):
        lines.append(f"- {a}")
    lines.append("")
    lines.append("Probes to apply:")
    for t in findings.get("probes", []):
        lines.append(f"- {t['name']} — {t['purpose']}")
    return "\n".join(lines)


# ---- MCP server ----


mcp = FastMCP("Sequential Argument Reasoning MCP")


# Load data once at import
_ONTO = Ontology(load_ontology(ONTOLOGY_CSV))
_TOOLS = ToolCatalog(load_tool_catalog(TOOLS_CSV))
_ENGINE = AnalysisEngine(_ONTO, _TOOLS)


@mcp.tool
def ontology_list_dimensions() -> List[str]:
    """List ontology dimensions (e.g., Argument Scheme, Fallacy, Cognitive Bias)."""
    return _ONTO.list_dimensions()


@mcp.tool
def ontology_list_categories(dimension: str) -> List[str]:
    """List categories for a given dimension."""
    return _ONTO.list_categories(dimension)


@mcp.tool
def ontology_list_buckets(dimension: Optional[str] = None, category: Optional[str] = None) -> List[str]:
    """List buckets (e.g., specific schemes/fallacies) optionally filtered by dimension/category."""
    return _ONTO.list_buckets(dimension, category)


@mcp.tool
def ontology_search(query: str, dimension: Optional[str] = None, category: Optional[str] = None, bucket: Optional[str] = None) -> List[Dict[str, str]]:
    """Full-text search over the ontology; optional filters for dimension/category/bucket."""
    return _ONTO.search(query, dimension, category, bucket)


@mcp.tool
def ontology_bucket_detail(bucket_name: str) -> List[Dict[str, str]]:
    """Return ontology rows exactly matching a bucket name (case-insensitive)."""
    return _ONTO.bucket_detail(bucket_name)


@mcp.tool
def tools_list() -> List[str]:
    """List available diagnostic/probe tools from the playbook."""
    return _TOOLS.list()


@mcp.tool
def tools_search(query: str) -> List[Dict[str, str]]:
    """Search the probe tool catalog by keywords."""
    return _TOOLS.search(query)


@mcp.tool
def tools_get(name: str) -> Optional[Dict[str, str]]:
    """Get a probe tool by exact name."""
    return _TOOLS.get(name)


@mcp.tool
def analyze_argument_comprehensive(
    argument_text: str,
    forum: Optional[str] = None,
    audience: Optional[str] = None,
    goal: Optional[str] = None,
    analysis_depth: str = "standard",
) -> Dict[str, Any]:
    """Multi-stage comprehensive argument analysis returning rich structure and probe plan."""
    ctx = AnalysisContext(forum=forum, audience=audience, goal=goal, depth=analysis_depth)
    return _ENGINE.analyze_comprehensive(argument_text, ctx)


@mcp.tool
def decompose_argument_structure(argument_text: str, include_implicit: bool = True) -> Dict[str, Any]:
    """Systematic breakdown into components and relationships with confidence scoring."""
    s1 = _ENGINE.stage1_decompose(argument_text)
    s2 = _ENGINE.stage2_patterns(argument_text, s1["nodes"])  # annotate nodes
    return {"structure": {"nodes": s2["nodes"], "links": s1["links"]}, "patterns": s2["patterns"]}


@mcp.tool
def detect_argument_patterns(argument_text: str, pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Identify argument schemes, reasoning patterns, and logical structures with confidence scores."""
    s2 = _ENGINE.stage2_patterns(argument_text, [])
    pats = s2["patterns"]
    if pattern_types:
        pats = [p for p in pats if p.get("pattern_type") in pattern_types]
    return {"patterns": pats}


@mcp.tool
def generate_missing_assumptions(argument_components: Dict[str, Any], prioritization: str = "critical") -> Dict[str, Any]:
    """Assumption generation using template-based gap analysis."""
    patterns = argument_components.get("patterns") or []
    # Also allow direct text
    if not patterns and argument_components.get("text"):
        s2 = _ENGINE.stage2_patterns(argument_components["text"], [])
        patterns = s2["patterns"]
    missing = _ENGINE.gap.analyze(patterns)
    if prioritization != "all":
        missing = [m for m in missing if m.priority == prioritization]
    return {"assumptions": [m.__dict__ for m in missing]}


@mcp.tool
def orchestrate_probe_analysis(analysis_results: Dict[str, Any], forum: Optional[str] = None, audience: Optional[str] = None, goal: Optional[str] = None) -> Dict[str, Any]:
    """Dynamic probe selection and chaining based on detected patterns and context."""
    ctx = AnalysisContext(forum=forum, audience=audience, goal=goal, depth="standard")
    patterns = analysis_results.get("patterns", [])
    plan = _ENGINE.stage4_probes(ctx, patterns)
    return plan


@mcp.tool
def ontology_semantic_search(query: str, dimensions: Optional[List[str]] = None, similarity_threshold: float = 0.2, max_results: int = 10) -> List[Dict[str, Any]]:
    """Semantic search across ontology using token-similarity; returns scored results."""
    return _ONTO.semantic_search(query, dimensions=dimensions, threshold=similarity_threshold, max_results=max_results)


@mcp.tool
def ontology_pattern_match(argument_patterns: List[Dict[str, Any]], match_type: str = "similar") -> List[Dict[str, Any]]:
    """Match detected patterns against ontology categories with coarse similarity mapping."""
    queries: List[str] = []
    for p in argument_patterns:
        t = (p.get("pattern_type") or "").lower()
        if t == "causal":
            queries.append("Argument from Cause to Effect")
            queries.append("Reasoning Pattern: Causal")
        elif t == "authority":
            queries.append("Argument from Expert Opinion")
            queries.append("Appeal to Authority")
        elif t == "analogical":
            queries.append("Argument from Analogy")
        elif t == "normative":
            queries.append("Practical Reasoning")
        elif t == "quantifier":
            queries.append("Hasty Generalization")
    results: List[Dict[str, Any]] = []
    for q in queries:
        results += _ONTO.semantic_search(q, dimensions=None, threshold=0.0, max_results=3)
    # Deduplicate by (dimension, bucket)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("dimension"), r.get("bucket"))
        if key not in seen:
            uniq.append(r)
            seen.add(key)
    return uniq


@mcp.tool
def construct_argument_graph(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build argument graph from analysis results, with adjacency lists."""
    structure = analysis_results.get("structure") or {}
    nodes = structure.get("nodes") or []
    links = structure.get("links") or []
    # Build adjacency
    adjacency: Dict[str, Dict[str, List[str]]] = {}
    for n in nodes:
        adjacency[n["node_id"]] = {"out": [], "in": []}
    for l in links:
        if l["source_node"] in adjacency:
            adjacency[l["source_node"]]["out"].append(l["target_node"])
        if l["target_node"] in adjacency:
            adjacency[l["target_node"]]["in"].append(l["source_node"])
    return {"nodes": nodes, "links": links, "adjacency": adjacency}


@mcp.tool
def validate_argument_graph(graph: Dict[str, Any], validation_level: str = "structural") -> Dict[str, Any]:
    """Check graph consistency and return identified issues with suggestions."""
    nodes = graph.get("nodes") or []
    links = graph.get("links") or []
    issues = _ENGINE._validate_graph(nodes, links)  # structural checks
    suggestions = []
    if issues.get("isolated_nodes"):
        suggestions.append("Connect isolated nodes to claims or remove as out-of-scope.")
    if issues.get("dangling_links"):
        suggestions.append("Ensure all links reference existing nodes.")
    return {"issues": issues, "suggestions": suggestions}


@mcp.tool
def compare_arguments(argument_graphs: List[Dict[str, Any]], comparison_dimensions: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compare multiple argument graphs by structure size and pattern coverage."""
    comps: List[Dict[str, Any]] = []
    for i, g in enumerate(argument_graphs):
        nodes = g.get("nodes") or []
        links = g.get("links") or []
        main_claims = [n for n in nodes if n.get("primary_subtype") == "Main Claim"]
        schemes = [n.get("argument_scheme") for n in nodes if n.get("argument_scheme")]
        comps.append({
            "index": i,
            "node_count": len(nodes),
            "link_count": len(links),
            "main_claim": main_claims[0].get("content") if main_claims else None,
            "schemes": sorted(set(schemes)),
        })
    return {"comparison": comps}


@mcp.tool
def assess_argument_quality(argument_graph: Dict[str, Any], assessment_framework: str = "comprehensive") -> Dict[str, Any]:
    """Qualitative strengths/weaknesses assessment from structure, patterns, and assumptions."""
    nodes = argument_graph.get("nodes") or []
    strengths: List[str] = []
    weaknesses: List[str] = []
    # Simple heuristics
    if any(n.get("argument_scheme") for n in nodes):
        strengths.append("Identified argument schemes provide explainable warrants.")
    if len([n for n in nodes if n.get("primary_subtype") == "Premise"]) >= 2:
        strengths.append("Multiple premises provide convergent support.")
    if any(n.get("assumptions") for n in nodes):
        weaknesses.append("Assumptions require validation; some may be linchpins.")
    if any("always" in (n.get("content","")).lower() for n in nodes):
        weaknesses.append("Quantified generality may risk hasty generalization.")
    return {"strengths": strengths, "weaknesses": weaknesses}


@mcp.tool
def identify_reasoning_weaknesses(argument_analysis: Dict[str, Any], weakness_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Flag plausible fallacies/biases/gaps with brief why/how rationales."""
    patterns = argument_analysis.get("patterns") or []
    findings: List[Dict[str, str]] = []
    for p in patterns:
        t = p.get("pattern_type")
        m = p.get("details", {}).get("match")
        if t == "authority":
            findings.append({"label": "Appeal to Authority", "why": f"Authority marker: {m}", "how": "Check expertise, bias, and consensus."})
        if t == "analogical":
            findings.append({"label": "Weak Analogy", "why": f"Analogical cue: {m}", "how": "Specify relevant similarity dimensions and scope."})
        if t == "quantifier":
            findings.append({"label": "Hasty Generalization", "why": f"Generalizing term: {m}", "how": "Provide representative sampling or constraints."})
        if t == "causal":
            findings.append({"label": "Post hoc / Confounding Risk", "why": f"Causal cue: {m}", "how": "Address time order, confounders, and mechanism."})
    return {"weaknesses": findings}


@mcp.tool
def generate_counter_analysis(argument_graph: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Generate structured counter-briefs: weaknesses, alternatives, or steel-man targets."""
    nodes = argument_graph.get("nodes") or []
    main = next((n for n in nodes if n.get("primary_subtype") == "Main Claim"), None)
    counters: List[str] = []
    if main:
        claim = main.get("content", "")
        counters.append(f"Alternative explanation for claim: challenge causal/mechanistic link in '{claim}'.")
        counters.append("Undercut warrant: show expert scope misalignment or COI.")
        counters.append("Counterexample: provide case where premises hold but conclusion fails.")
    return {"counter_points": counters}


@mcp.resource("argument://dimensions")
def resource_dimensions() -> str:
    return "\n".join(_ONTO.list_dimensions())


@mcp.resource("argument://buckets/{dimension}")
def resource_buckets_by_dim(dimension: str) -> str:
    buckets = _ONTO.list_buckets(dimension=dimension)
    return "\n".join(buckets)


@mcp.resource("argument://tools")
def resource_tools() -> str:
    return "\n".join(_TOOLS.list())


@mcp.prompt("analyze")
def prompt_analyze(
    claim: str,
    forum: str = "",
    audience: str = "",
    goal: str = "",
) -> str:
    """
    Returns a prompt template aligned with the Master Brief ethos to guide LLMs.
    Note: This is a textual scaffold; the `analyze_claim` tool is the rules-based alternative.
    """
    parts = [
        "You are an ontology-aware argument analyst.",
        "Fidelity-first: preserve the user’s meaning; label uncertainties.",
        "Decompose and synthesize: claims, reasons, evidence, assumptions, warrants, schemes.",
        "Dialectical rigor: develop both strengthening and weakening pathways.",
        "Use the following context (forum/audience/goal) to calibrate standards:",
        f"Forum: {forum}",
        f"Audience: {audience}",
        f"Goal: {goal}",
        "Claim:",
        claim,
        "Deliver:",
        "- Claims and sub-claims",
        "- Candidate schemes with scheme-specific critical questions (if known)",
        "- Evidence type(s) and suitability for the forum",
        "- Assumptions (explicit/implicit; mark linchpins)",
        "- Reasoning patterns (deductive/inductive/abductive/causal/etc.)",
        "- Linkage semantics (supports/linked; undercut/rebut; necessary/sufficient)",
        "- Vulnerabilities (fallacies, biases, ID threats) with brief why/how",
        "- Probe plan using diagnostic tools (Toulmin, Causal Probe, Assumption Audit, etc.)",
        "- Strengthening and weakening pathways",
        "- Clarifying questions if needed",
        "Style: concise, scannable lists; no scores; no invented facts.",
    ]
    return "\n".join(parts)


# Convenience: allow running as a script for local dev (optional)
if __name__ == "__main__":
    print("Loaded dimensions:", ", ".join(_ONTO.list_dimensions())[:200], "...")
    print("Probe tools count:", len(_TOOLS.list()))
