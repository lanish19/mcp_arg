"""
Sequential Argument Reasoning MCP — ontology-aware argument tooling.

Enhanced, multi-stage analysis engine with argument structure decomposition, pattern detection, gap analysis, probe orchestration, and integration. Backward compatible with basic ontology browsing tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import os
import hashlib
import json
import time

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
API_VERSION = "v1.1.0"
_start_time = time.time()

# ---- Performance & limits ----
MAX_INPUT_LENGTH = 50000
MAX_PATTERNS = 20
MAX_WEAKNESSES = 15
MAX_ASSUMPTIONS = 12
MAX_PROBES = 8

def _validate_input_size(text: str, max_length: int = MAX_INPUT_LENGTH) -> Optional[Dict[str, Any]]:
    if isinstance(text, str) and len(text) > max_length:
        return _envelope(
            data=None,
            schema=_schema("error"),
            error=_error(
                "INVALID_INPUT",
                f"Input text too long: {len(text)} characters (max: {max_length})",
                hint="Consider breaking into smaller chunks for analysis",
                where="size_guard",
            ),
        )
    return None

def _cap_response_size(data: Any, caps: Optional[Dict[str, int]] = None) -> Tuple[Any, bool]:
    caps = caps or {
        "patterns": MAX_PATTERNS,
        "weaknesses": MAX_WEAKNESSES,
        "assumptions": MAX_ASSUMPTIONS,
        "probe_plan": MAX_PROBES,
        "tools": 200,
    }
    truncated = False
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, list) and k in caps and len(v) > caps[k]:
                out[k] = v[: caps[k]]
                truncated = True
            elif isinstance(v, (dict, list)):
                vv, t = _cap_response_size(v, caps)
                out[k] = vv
                truncated = truncated or t
            else:
                out[k] = v
        return out, truncated
    if isinstance(data, list):
        new_list: List[Any] = []
        for item in data:
            if isinstance(item, (dict, list)):
                ii, t = _cap_response_size(item, caps)
                new_list.append(ii)
                truncated = truncated or t
            else:
                new_list.append(item)
        return new_list, truncated
    return data, truncated

def _envelope(data: Any = None, schema: str = "", warnings: Optional[List[str]] = None, error: Optional[Dict[str, Any]] = None, next_steps: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "version": API_VERSION,
        "data": data,
        "metadata": {
            "schema_url": schema or "",
            "warnings": warnings or [],
            "next_steps": next_steps or [],
            **(meta or {}),
        },
        "error": error,
    }

def _compat_enabled(value: Any = None, **kwargs) -> bool:
    """Return True if compat/raw mode is requested."""
    if isinstance(value, str) and value.lower() == "raw":
        return True
    if value is False:
        return True
    if kwargs.get("envelope") is False:
        return True
    return False

def _maybe_envelope(data: Any, schema_name: str, compat: Any = None, **kwargs) -> Any:
    return data if _compat_enabled(compat, **kwargs) else _envelope(data=data, schema=_schema(schema_name))

def _schema(name: str) -> str:
    return f"schemas/v1/{name}.response.json"
def _error(code: str, message: str, hint: Optional[str] = None, where: Optional[str] = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if hint:
        err["hint"] = hint
    if where:
        err["where"] = where
    return err

def _log_event(endpoint: str, start: float, input_size: int, result_size: int) -> None:
    try:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log = {"endpoint": endpoint, "elapsed_ms": elapsed_ms, "input_size": input_size, "result_size": result_size}
        # Minimal stdout logging to keep deterministic behavior
        print(json.dumps({"log": log}))
    except Exception:
        pass


# Load data once at import
_ONTO = Ontology(load_ontology(ONTOLOGY_CSV))
_TOOLS = ToolCatalog(load_tool_catalog(TOOLS_CSV))
_ENGINE = AnalysisEngine(_ONTO, _TOOLS)

# ---- Stable tool registry & resources ----
_TOOL_REGISTRY_CACHE: Optional[List[str]] = None

def _get_tool_registry() -> List[str]:
    global _TOOL_REGISTRY_CACHE
    if _TOOL_REGISTRY_CACHE is None:
        try:
            _TOOL_REGISTRY_CACHE = _TOOLS.list()
        except Exception:
            _TOOL_REGISTRY_CACHE = []
    return _TOOL_REGISTRY_CACHE

def _handle_service_unavailable(retry_after: int = 30) -> Dict[str, Any]:
    err = _error("SERVICE_UNAVAILABLE", "Service temporarily unavailable", hint="Retry after backend restart completes", where="service")
    env = _envelope(data=None, schema=_schema("error"), error=err, next_steps=["retry"])  # type: ignore[arg-type]
    env["metadata"]["retry_after"] = retry_after
    return env


@mcp.tool
def ontology_list_dimensions(compat: Optional[Any] = None) -> List[str]:
    """List ontology dimensions (e.g., Argument Scheme, Fallacy, Cognitive Bias)."""
    start = time.perf_counter()
    dims = _ONTO.list_dimensions()
    # include counts per dimension
    counts: Dict[str, int] = {}
    for d in dims:
        counts[d] = len(_ONTO.list_buckets(dimension=d))
    data = {"dimensions": [{"name": d, "count_buckets": counts.get(d, 0)} for d in dims]}
    out = _maybe_envelope(data, "ontology_list_dimensions", compat=compat)
    _log_event("ontology_list_dimensions", start, 0, len(json.dumps(out)))
    return out


@mcp.tool
def ontology_list_categories(dimension: str, compat: Optional[Any] = None) -> List[str]:
    """List categories for a given dimension."""
    start = time.perf_counter()
    cats = _ONTO.list_categories(dimension)
    data = {"dimension": dimension, "categories": [{"name": c, "count_buckets": len(_ONTO.list_buckets(dimension, c))} for c in cats]}
    out = _maybe_envelope(data, "ontology_list_categories", compat=compat)
    _log_event("ontology_list_categories", start, len(dimension), len(json.dumps(out)))
    return out


@mcp.tool
def ontology_list_buckets(dimension: Optional[str] = None, category: Optional[str] = None, compat: Optional[Any] = None) -> List[str]:
    """List buckets (e.g., specific schemes/fallacies) optionally filtered by dimension/category."""
    start = time.perf_counter()
    buckets = _ONTO.list_buckets(dimension, category)
    data = {"dimension": dimension, "category": category, "buckets": [{"name": b, "parent": {"dimension": dimension, "category": category}} for b in buckets]}
    out = _maybe_envelope(data, "ontology_list_buckets", compat=compat)
    _log_event("ontology_list_buckets", start, len(json.dumps({"dimension":dimension,"category":category})), len(json.dumps(out)))
    return out


@mcp.tool
def ontology_search(query: str, dimension: Optional[str] = None, category: Optional[str] = None, bucket: Optional[str] = None, compat: Optional[Any] = None) -> List[Dict[str, str]]:
    """Full-text search over the ontology; optional filters for dimension/category/bucket."""
    start = time.perf_counter()
    results = _ONTO.search(query, dimension, category, bucket)
    data = {"results": results, "applied_synonyms": _ONTO.last_applied_synonyms()}
    out = _maybe_envelope(data, "ontology_search", compat=compat)
    _log_event("ontology_search", start, len(query), len(json.dumps(out)))
    return out


@mcp.tool
def ontology_bucket_detail(bucket_name: str, compat: Optional[Any] = None) -> List[Dict[str, str]]:
    """Return ontology rows exactly matching a bucket name (case-insensitive)."""
    start = time.perf_counter()
    rows = _ONTO.bucket_detail(bucket_name)
    out = _maybe_envelope({"rows": rows}, "ontology_bucket_detail", compat=compat)
    _log_event("ontology_bucket_detail", start, len(bucket_name), len(json.dumps(out)))
    return out


@mcp.tool
def tools_list(compat: Optional[Any] = None) -> List[str]:
    """List available diagnostic/probe tools from the playbook."""
    start = time.perf_counter()
    tools = _get_tool_registry()
    out = _maybe_envelope({"tools": tools}, "tools_list", compat=compat)
    _log_event("tools_list", start, 0, len(json.dumps(out)))
    return out


@mcp.tool
def tools_search(query: str, compat: Optional[Any] = None) -> List[Dict[str, str]]:
    """Search the probe tool catalog by keywords."""
    start = time.perf_counter()
    res = _TOOLS.search(query)
    out = _maybe_envelope({"results": res}, "tools_search", compat=compat)
    _log_event("tools_search", start, len(query), len(json.dumps(out)))
    return out


@mcp.tool
def tools_semantic_search(query: str, max_results: int = 10, threshold: float = 0.1, compat: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Semantic search over tool catalog (name+purpose+how)."""
    start = time.perf_counter()
    res = _TOOLS.semantic_search_tools(query, threshold=threshold, max_results=max_results)
    out = _maybe_envelope({"results": res}, "tools_semantic_search", compat=compat)
    _log_event("tools_semantic_search", start, len(query), len(json.dumps(out)))
    return out


@mcp.tool
def tools_dump(page: int = 1, per_page: int = 50, fields: Optional[List[str]] = None, compat: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Page through tool catalog for client-side selection."""
    start = time.perf_counter()
    res = _TOOLS.dump(page=page, per_page=per_page, fields=fields)
    out = _maybe_envelope({"page": page, "per_page": per_page, "tools": res}, "tools_dump", compat=compat)
    _log_event("tools_dump", start, len(json.dumps({"page":page,"per_page":per_page})), len(json.dumps(out)))
    return out


@mcp.tool
def recommend_tools(analysis_results: Dict[str, Any], forum: Optional[str] = None, audience: Optional[str] = None, goal: Optional[str] = None, max_results: int = 10, compat: Optional[Any] = None) -> Dict[str, Any]:
    """Rank tools by semantic similarity to patterns, weaknesses, and validation issues. Deterministic."""
    start = time.perf_counter()
    pats = (analysis_results.get("patterns") or []) if isinstance(analysis_results, dict) else []
    weaks = (analysis_results.get("weaknesses") or []) if isinstance(analysis_results, dict) else []
    val = (analysis_results.get("validation", {}) or {}) if isinstance(analysis_results, dict) else {}
    queries: List[str] = []
    for p in pats:
        queries.append(" ".join([p.get("pattern_type",""), (p.get("details") or {}).get("scheme",""), (p.get("label") or "")]))
    for w in weaks:
        queries.append(" ".join([w.get("label",""), w.get("why",""), w.get("how","")]))
    for k in (val.get("issues", {}) or {}).keys():
        queries.append(k)
    # Fallback to forum/audience/goal keywords to bias selection
    for x in [forum, audience, goal]:
        if x:
            queries.append(str(x))
    scored: Dict[str, Dict[str, Any]] = {}
    for q in queries:
        for r in _TOOLS.semantic_search_tools(q, threshold=0.12, max_results=max_results):
            key = r["slug"]
            best = scored.get(key)
            if not best or r["score"] > best["score"]:
                scored[key] = r
    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:max_results]
    out = _maybe_envelope({"candidates": ranked}, "recommend_tools", compat=compat)
    _log_event("recommend_tools", start, len(json.dumps(analysis_results)), len(json.dumps(out)))
    return out

@mcp.tool
def tools_get(name: str, compat: Optional[Any] = None) -> Optional[Dict[str, str]]:
    """Get a probe tool by exact name."""
    start = time.perf_counter()
    res = _TOOLS.get(name)
    if not res:
        err = _error("TOOL_NOT_FOUND", f"Tool '{name}' not found", hint="Use tools_list or tools_search to discover available tools.", where="tools_get")
        out = _maybe_envelope(None, "tools_get", compat=compat)
        out["error"] = err
        _log_event("tools_get", start, len(name), len(json.dumps(out)))
        return out
    out = _maybe_envelope({"tool": res}, "tools_get", compat=compat)
    _log_event("tools_get", start, len(name), len(json.dumps(out)))
    return out


@mcp.tool
def analyze_argument_comprehensive(
    argument_text: str,
    forum: Optional[str] = None,
    audience: Optional[str] = None,
    goal: Optional[str] = None,
    analysis_depth: str = "standard",
    compat: Optional[Any] = None,
) -> Dict[str, Any]:
    """Multi-stage comprehensive argument analysis returning rich structure and probe plan.

    Prefer `analyze_argument_stepwise` for multi-step chaining and tool-by-tool artifacts.
    """
    start = time.perf_counter()
    guard = _validate_input_size(argument_text)
    if guard is not None:
        return guard
    ctx = AnalysisContext(forum=forum, audience=audience, goal=goal, depth=analysis_depth)
    res = _ENGINE.analyze_comprehensive(argument_text, ctx)
    # Cap payload sizes for safety
    capped, truncated = _cap_response_size(res)
    meta = {"truncated": truncated} if truncated else None
    out = _maybe_envelope(capped, "analyze_argument_comprehensive", compat=compat)
    if truncated and isinstance(out, dict):
        try:
            out.setdefault("metadata", {})
            out["metadata"]["warnings"] = (out["metadata"].get("warnings") or []) + ["Response truncated by caps"]
        except Exception:
            pass
    _log_event("analyze_argument_comprehensive", start, len(argument_text), len(json.dumps(out)))
    return out


@mcp.tool
def decompose_argument_structure(argument_text: str, include_implicit: bool = True, compat: Optional[Any] = None) -> Dict[str, Any]:
    """Systematic breakdown into components and relationships with confidence scoring."""
    if not isinstance(argument_text, str) or not argument_text.strip():
        err = _error("MISSING_ARGUMENT_TEXT", "argument_text is required", where="decompose_argument_structure")
        out = _envelope(data=None, schema=_schema("decompose_argument_structure"), error=err)
        _log_event("decompose_argument_structure", time.perf_counter(), 0, len(json.dumps(out)))
        return out
    guard = _validate_input_size(argument_text)
    if guard is not None:
        return guard
    s1 = _ENGINE.stage1_decompose(argument_text)
    s2 = _ENGINE.stage2_patterns(argument_text, s1["nodes"])  # annotate nodes
    # Compact links and nodes are already pruned by to_dict; keep only necessary fields
    resp = {
        "structure": {"nodes": s2["nodes"], "links": s1["links"]},
        "patterns": s2["patterns"],
        "next_tools": ["detect_argument_patterns", "ontology_pattern_match"],
    }
    # Optional tracing metadata
    inputs_summary = {
        "argument_text": (argument_text[:200] + ("…" if len(argument_text) > 200 else "")),
        "include_implicit": include_implicit,
    }
    resp["stage_id"] = "decompose_argument_structure"
    resp["inputs_digest"] = hashlib.sha1(json.dumps(inputs_summary, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    out = _maybe_envelope(resp, "decompose_argument_structure", compat=compat)
    _log_event("decompose_argument_structure", time.perf_counter(), len(argument_text), len(json.dumps(out)))
    return out


# Health check endpoint (stable tool registry & discovery)
@mcp.tool
def health_status(compat: Optional[Any] = None) -> Dict[str, Any]:
    """Health check endpoint returning system status."""
    uptime_ms = int((time.time() - _start_time) * 1000)
    try:
        dims = _ONTO.list_dimensions()
        tool_count = len(_get_tool_registry())
        ont_counts = {d: len(_ONTO.list_buckets(dimension=d)) for d in dims}
    except Exception:
        dims = []
        tool_count = 0
        ont_counts = {}
    data = {
        "version": API_VERSION,
        "uptime_ms": uptime_ms,
        "dimensions": dims,
        "tool_count": tool_count,
        "ontology_bucket_counts": ont_counts,
        "status": "healthy" if dims and tool_count >= 0 else "degraded",
        "commit": os.environ.get("GIT_COMMIT", "unknown"),
    }
    return _maybe_envelope(data, "tools_list", compat=compat)


# Raw variant for compatibility with clients expecting arrays
@mcp.tool
def ontology_list_dimensions_raw() -> Any:
    # Raw array version for maximum compatibility
    try:
        dims = _ONTO.list_dimensions()
        return dims
    except Exception:
        return []


@mcp.tool
def detect_argument_patterns(argument_text: str, pattern_types: Optional[List[str]] = None, compat: Optional[Any] = None) -> Dict[str, Any]:
    """Identify argument schemes, reasoning patterns, and logical structures with confidence scores."""
    if not isinstance(argument_text, str) or not argument_text.strip():
        err = _error("MISSING_ARGUMENT_TEXT", "argument_text is required", where="detect_argument_patterns")
        out = _envelope(data=None, schema=_schema("detect_argument_patterns"), error=err)
        _log_event("detect_argument_patterns", time.perf_counter(), 0, len(json.dumps(out)))
        return out
    s2 = _ENGINE.stage2_patterns(argument_text, [])
    pats = s2["patterns"]
    if pattern_types:
        pats = [p for p in pats if p.get("pattern_type") in pattern_types]
    # Patterns already trimmed in detector; optionally filter by type
    resp = {"patterns": pats, "next_tools": ["ontology_pattern_match", "generate_missing_assumptions", "orchestrate_probe_analysis"]}
    inputs_summary = {
        "argument_text": (argument_text[:200] + ("…" if len(argument_text) > 200 else "")),
        "pattern_types": pattern_types or [],
    }
    resp["stage_id"] = "detect_argument_patterns"
    resp["inputs_digest"] = hashlib.sha1(json.dumps(inputs_summary, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    out = _maybe_envelope(resp, "detect_argument_patterns", compat=compat)
    _log_event("detect_argument_patterns", time.perf_counter(), len(argument_text), len(json.dumps(out)))
    return out


@mcp.tool
def generate_missing_assumptions(argument_components: Dict[str, Any], prioritization: str = "critical", compat: Optional[Any] = None) -> Dict[str, Any]:
    """Assumption generation using template-based gap analysis."""
    if not isinstance(argument_components, dict):
        err = _error("INVALID_INPUT_SHAPE", "argument_components must be an object", where="generate_missing_assumptions")
        out = _envelope(data=None, schema=_schema("generate_missing_assumptions"), error=err)
        _log_event("generate_missing_assumptions", time.perf_counter(), 0, len(json.dumps(out)))
        return out
    text = argument_components.get("text", "")
    patterns = argument_components.get("patterns") or []
    # If no patterns provided but text is, detect minimally
    if not patterns and text:
        s2 = _ENGINE.stage2_patterns(text, [])
        patterns = s2["patterns"]
    warnings: List[str] = []
    if not text and not patterns:
        # Missing inputs; return empty set with warning
        resp = {"assumptions": [], "next_tools": ["construct_argument_graph", "validate_argument_graph"]}
        out = _envelope(data=resp, schema=_schema("generate_missing_assumptions"), warnings=["No text or patterns provided; unable to infer assumptions."])
        _log_event("generate_missing_assumptions", time.perf_counter(), len(json.dumps(argument_components)), len(json.dumps(out)))
        return out
    # Use deterministic stage3 inference to generate assumption strings
    infer = _ENGINE.stage3_infer(text, patterns)
    assumptions = infer.get("assumptions", [])
    # Already enriched by stage3; coerce to dicts if strings for backward compatibility
    assumption_dicts: List[Dict[str, Any]] = []
    for a in assumptions:
        if isinstance(a, dict):
            # Keep fields but truncate text for hygiene
            t = (a.get("text", "") or "")
            a2 = dict(a)
            a2["text"] = t[:200]
            # Ensure required fields per schema
            if not a2.get("category"):
                ptype = (a.get("pattern_type") or "")
                a2["category"] = {
                    "authority": "epistemic",
                    "causal": "mechanistic",
                    "analogical": "bridging",
                    "normative": "contentious",
                }.get(ptype, "epistemic")
            imp = a2.get("impact")
            if imp == "med":
                a2["impact"] = "medium"
            if not a2.get("tests"):
                a2["tests"] = ["evidence_request"]
            if "confidence" in a2 and isinstance(a2["confidence"], float):
                a2["confidence"] = round(float(a2["confidence"]), 3)
            assumption_dicts.append(a2)
        else:
            assumption_dicts.append({"text": str(a)[:200], "category": "epistemic", "impact": "medium", "confidence": 0.5, "tests": ["evidence_request"]})
    # Prioritize critical/high
    if prioritization == "critical":
        assumption_dicts = [a for a in assumption_dicts if a.get("impact") in ["critical", "high"]]
    resp = {"assumptions": assumption_dicts[:MAX_ASSUMPTIONS], "next_tools": ["construct_argument_graph", "validate_argument_graph"]}
    inputs_summary = {
        "has_patterns": bool(argument_components.get("patterns")),
        "has_text": bool(text),
        "prioritization": prioritization,
    }
    resp["stage_id"] = "generate_missing_assumptions"
    resp["inputs_digest"] = hashlib.sha1(json.dumps(inputs_summary, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    out = _maybe_envelope(resp, "generate_missing_assumptions", compat=compat)
    _log_event("generate_missing_assumptions", time.perf_counter(), len(json.dumps(argument_components)), len(json.dumps(out)))
    return out


@mcp.tool
def orchestrate_probe_analysis(analysis_results: Dict[str, Any], forum: Optional[str] = None, audience: Optional[str] = None, goal: Optional[str] = None, compat: Optional[Any] = None) -> Dict[str, Any]:
    """Dynamic probe selection and chaining based on detected patterns and context."""
    ctx = AnalysisContext(forum=forum, audience=audience, goal=goal, depth="standard")
    patterns = analysis_results.get("patterns", [])
    structure = analysis_results.get("structure", {})
    weaknesses = analysis_results.get("weaknesses", [])
    plan = _ENGINE.stage4_probes(ctx, patterns, structure=structure, weaknesses=weaknesses)
    # Also include ranked candidates so clients can choose
    queries: List[str] = []
    for p in patterns:
        queries.append(" ".join([p.get("pattern_type",""), (p.get("details") or {}).get("scheme",""), (p.get("label") or "")]))
    if analysis_results.get("structure"):
        # Use node subtypes to bias
        for n in (analysis_results.get("structure") or {}).get("nodes", []):
            if n.get("primary_subtype"):
                queries.append(n.get("primary_subtype"))
    candidates = _ENGINE.probes.ranked_candidates(queries, max_results=10)
    plan_data = plan.get("probe_plan", [])[:MAX_PROBES]
    data = {"probe_plan": plan_data, "candidates": candidates}
    out = _maybe_envelope(data, "orchestrate_probe_analysis", compat=compat)
    _log_event("orchestrate_probe_analysis", time.perf_counter(), len(json.dumps(analysis_results)), len(json.dumps(out)))
    return out


@mcp.tool
def ontology_semantic_search(query: str, dimensions: Optional[List[str]] = None, similarity_threshold: float = 0.2, max_results: int = 10, compat: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Semantic search across ontology using token-similarity; returns scored results."""
    start = time.perf_counter()
    data = _ONTO.semantic_search(query, dimensions=dimensions, threshold=similarity_threshold, max_results=max_results)
    out = _maybe_envelope({"results": data, "applied_synonyms": _ONTO.last_applied_synonyms()}, "ontology_semantic_search", compat=compat)
    _log_event("ontology_semantic_search", start, len(query), len(json.dumps(out)))
    return out


@mcp.tool
def ontology_pattern_match(argument_patterns: List[Dict[str, Any]], match_type: str = "similar", compat: Optional[Any] = None) -> List[Dict[str, Any]]:
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
    out = _maybe_envelope({"matches": uniq}, "ontology_pattern_match", compat=compat)
    _log_event("ontology_pattern_match", time.perf_counter(), len(json.dumps(argument_patterns)), len(json.dumps(out)))
    return out


@mcp.tool
def construct_argument_graph(analysis_results: Dict[str, Any], compat: Optional[Any] = None) -> Dict[str, Any]:
    """Build argument graph from analysis results, with adjacency lists."""
    # Accept either {"structure": {...}} or direct {"nodes", "links"}
    if not analysis_results:
        err = _error("INVALID_INPUT_SHAPE", "Missing analysis_results", where="construct_argument_graph")
        out = _maybe_envelope(None, "construct_argument_graph", compat=compat)
        out["error"] = err
        _log_event("construct_argument_graph", time.perf_counter(), 0, len(json.dumps(out)))
        return out
    structure = analysis_results.get("structure") or analysis_results
    nodes = structure.get("nodes") or []
    links = structure.get("links") or []
    if not isinstance(nodes, list) or not isinstance(links, list):
        err = _error("INVALID_INPUT_SHAPE", "analysis_results must contain 'structure' or 'nodes'", hint="Call decompose_argument_structure first to generate structure", where="construct_argument_graph")
        out = _maybe_envelope({"nodes": [], "links": []}, "construct_argument_graph", compat=compat)
        out["error"] = err
        _log_event("construct_argument_graph", time.perf_counter(), len(json.dumps(analysis_results)), len(json.dumps(out)))
        return out
    # Build adjacency
    adjacency: Dict[str, Dict[str, List[str]]] = {}
    for n in nodes:
        adjacency[n["node_id"]] = {"outgoing": [], "incoming": []}
    for l in links:
        if l["source_node"] in adjacency:
            adjacency[l["source_node"]]["outgoing"].append(l["target_node"])
        if l["target_node"] in adjacency:
            adjacency[l["target_node"]]["incoming"].append(l["source_node"])
    resp = {"nodes": nodes, "links": links, "adjacency": adjacency, "next_tools": ["validate_argument_graph", "assess_argument_quality", "generate_counter_analysis"]}
    inputs_summary = {
        "nodes": len(nodes),
        "links": len(links),
    }
    resp["stage_id"] = "construct_argument_graph"
    resp["inputs_digest"] = hashlib.sha1(json.dumps(inputs_summary, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    out = _maybe_envelope(resp, "construct_argument_graph", compat=compat)
    _log_event("construct_argument_graph", time.perf_counter(), len(json.dumps(analysis_results)), len(json.dumps(out)))
    return out


@mcp.tool
def validate_argument_graph(graph: Dict[str, Any], validation_level: str = "structural", compat: Optional[Any] = None) -> Dict[str, Any]:
    """Enhanced validation with structured categories and guidance."""
    nodes = graph.get("nodes") or []
    links = graph.get("links") or []
    eng_issues = _ENGINE._validate_graph(nodes, links)  # base structural checks
    issues: Dict[str, List[Any]] = {"structural": [], "logical": [], "completeness": [], "consistency": []}
    # Structural
    if eng_issues.get("dangling_links"):
        for lid in eng_issues["dangling_links"]:
            issues["structural"].append(f"Dangling link {lid}")
    if eng_issues.get("duplicate_edges"):
        for lid in eng_issues["duplicate_edges"]:
            issues["structural"].append(f"Duplicate link {lid}")
    if eng_issues.get("cycles"):
        issues["structural"].append(f"Circular reasoning detected in {len(eng_issues['cycles'])} cycle(s)")
    # Completeness
    if not nodes:
        issues["completeness"].append("Graph contains no nodes")
    if len(nodes) > 1 and not links:
        issues["completeness"].append("Multi-node graph has no relationships")
    if any(n.get("source_text_span") is None for n in nodes if isinstance(n, dict)):
        issues["completeness"].append("Some nodes missing source_text_span")
    # Logical
    mains = [n for n in nodes if n.get("primary_subtype") == "Main Claim"]
    if len(mains) == 0:
        issues["logical"].append("Missing main claim")
    if len(mains) > 1:
        issues["logical"].append("Multiple main claims present")
    # Consistency
    # Detect contradictory relationships (SUPPORT vs ATTACK) between same nodes
    seen_pairs: Dict[Tuple[str, str], set] = {}
    for l in links:
        key = (l.get("source_node"), l.get("target_node"))
        seen_pairs.setdefault(key, set()).add(l.get("relationship_type"))
    for key, kinds in seen_pairs.items():
        if "SUPPORT" in kinds and "ATTACK" in kinds:
            issues["consistency"].append(f"Conflicting relationships between {key[0]} and {key[1]}")

    suggestions: List[str] = []
    if issues["structural"]:
        suggestions.append("Fix dangling/duplicate links and resolve cycles")
    if issues["completeness"]:
        suggestions.append("Add relationships and ensure nodes include source spans")
    if issues["logical"]:
        suggestions.append("Clarify main claim and link premises appropriately")
    if issues["consistency"]:
        suggestions.append("Resolve conflicting support/attack links")

    total_issues = sum(len(v) for v in issues.values())
    summary = {
        "total_issues": total_issues,
        "severity": "high" if issues["structural"] else ("medium" if total_issues else "none"),
        "graph_health": "unhealthy" if issues["structural"] else ("needs_attention" if total_issues else "healthy"),
    }
    resp = {"issues": issues, "suggestions": suggestions, "next_tools": ["assess_argument_quality", "identify_reasoning_weaknesses"], "next_steps": ["export_graph"], "summary": summary}
    inputs_summary = {"nodes": len(nodes), "links": len(links), "validation_level": validation_level}
    resp["stage_id"] = "validate_argument_graph"
    resp["inputs_digest"] = hashlib.sha1(json.dumps(inputs_summary, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    out = _maybe_envelope(resp, "validate_argument_graph", compat=compat)
    _log_event("validate_argument_graph", time.perf_counter(), len(json.dumps(graph)), len(json.dumps(out)))
    return out


@mcp.tool
def analyze_argument_stepwise(
    argument_text: str,
    steps: Optional[List[str]] = None,
    forum: Optional[str] = None,
    audience: Optional[str] = None,
    goal: Optional[str] = None,
    max_steps: Optional[int] = None,
    compat: Optional[Any] = None
) -> Dict[str, Any]:
    """Execute a recommended multi-step pipeline and return stage-by-stage artifacts and chaining hints."""
    default_steps: List[str] = [
        "decompose_argument_structure",
        "detect_argument_patterns",
        "ontology_pattern_match",
        "generate_missing_assumptions",
        "orchestrate_probe_analysis",
        "construct_argument_graph",
        "validate_argument_graph",
        "assess_argument_quality",
        "identify_reasoning_weaknesses",
        "generate_counter_analysis",
    ]
    allowed: List[str] = list(default_steps)
    sequence: List[str] = default_steps if steps is None else steps

    # Validate steps
    stages: List[Dict[str, Any]] = []
    validated_steps: List[str] = []
    for name in sequence:
        if name not in allowed:
            err = _error("INVALID_INPUT_SHAPE", f"Unknown step '{name}'", hint="See 'allowed' for valid steps", where="analyze_argument_stepwise")
            data = {"step": name, "allowed": allowed, "stages_so_far": stages}
            out = _envelope(data=data, schema=_schema("analyze_argument_stepwise"), error=err)
            _log_event("analyze_argument_stepwise", time.perf_counter(), len(json.dumps(sequence)), len(json.dumps(out)))
            return out
        validated_steps.append(name)

    if isinstance(max_steps, int) and max_steps >= 0:
        validated_steps = validated_steps[:max_steps]

    # Orchestration state
    ctx = AnalysisContext(forum=forum, audience=audience, goal=goal, depth="standard")
    current_structure: Dict[str, Any] = {}
    current_patterns: List[Dict[str, Any]] = []
    current_assumptions: List[Any] = []
    current_graph: Dict[str, Any] = {}
    current_assessments: Dict[str, Any] = {}
    current_probes: Dict[str, Any] = {}

    # next_tools mapping (explicit per requirements; others are sensible defaults)
    next_map: Dict[str, List[str]] = {
        "decompose_argument_structure": ["detect_argument_patterns", "ontology_pattern_match"],
        "detect_argument_patterns": ["ontology_pattern_match", "generate_missing_assumptions", "orchestrate_probe_analysis"],
        "generate_missing_assumptions": ["construct_argument_graph", "validate_argument_graph"],
        "construct_argument_graph": ["validate_argument_graph", "assess_argument_quality", "generate_counter_analysis"],
        "validate_argument_graph": ["assess_argument_quality", "identify_reasoning_weaknesses"],
        "ontology_pattern_match": ["generate_missing_assumptions", "orchestrate_probe_analysis"],
        "orchestrate_probe_analysis": ["construct_argument_graph", "assess_argument_quality"],
        "assess_argument_quality": ["identify_reasoning_weaknesses", "generate_counter_analysis"],
        "identify_reasoning_weaknesses": ["generate_counter_analysis"],
        "generate_counter_analysis": [],
    }

    def trunc(t: str) -> str:
        return t[:200] + ("…" if len(t) > 200 else "")

    for step in validated_steps:
        if step == "decompose_argument_structure":
            env = decompose_argument_structure(argument_text)
            r = env.get("data", {}) if isinstance(env, dict) else env
            current_structure = r.get("structure", {})
            current_patterns = r.get("patterns", current_patterns)
            stages.append({
                "name": step,
                "inputs_summary": {"argument_text": trunc(argument_text)},
                "key_outputs": {"nodes": len(current_structure.get("nodes", [])), "links": len(current_structure.get("links", []))},
                "next_tools": next_map.get(step, []),
            })
        elif step == "detect_argument_patterns":
            env = detect_argument_patterns(argument_text)
            r = env.get("data", {}) if isinstance(env, dict) else env
            current_patterns = r.get("patterns", [])
            top = [p.get("pattern_type") for p in current_patterns[:3]]
            stages.append({
                "name": step,
                "inputs_summary": {"argument_text": trunc(argument_text)},
                "key_outputs": {"count": len(current_patterns), "top": top},
                "next_tools": next_map.get(step, []),
            })
        elif step == "ontology_pattern_match":
            env = ontology_pattern_match(current_patterns)
            r = env.get("data", {}) if isinstance(env, dict) else env
            matches = r.get("matches", []) if isinstance(r, dict) else (r or [])
            top_buckets = []
            seen = set()
            for m in matches:
                b = m.get("bucket")
                if b and b not in seen:
                    top_buckets.append(b)
                    seen.add(b)
                if len(top_buckets) >= 3:
                    break
            stages.append({
                "name": step,
                "inputs_summary": {"patterns": len(current_patterns)},
                "key_outputs": {"matches": len(matches), "top_buckets": top_buckets},
                "next_tools": next_map.get(step, []),
            })
        elif step == "generate_missing_assumptions":
            env = generate_missing_assumptions({"patterns": current_patterns, "text": argument_text})
            r = env.get("data", {}) if isinstance(env, dict) else env
            current_assumptions = r.get("assumptions", [])
            stages.append({
                "name": step,
                "inputs_summary": {"patterns": len(current_patterns)},
                "key_outputs": {"assumptions": len(current_assumptions)},
                "next_tools": next_map.get(step, []),
            })
        elif step == "orchestrate_probe_analysis":
            env = orchestrate_probe_analysis({"patterns": current_patterns, "structure": current_structure}, forum=forum, audience=audience, goal=goal)
            r = env.get("data", {}) if isinstance(env, dict) else env
            current_probes = r
            probes_list = r.get("probe_plan", []) if isinstance(r, dict) else []
            stages.append({
                "name": step,
                "inputs_summary": {"patterns": len(current_patterns), "forum": forum, "audience": audience, "goal": goal},
                "key_outputs": {"probes": len(probes_list)},
                "next_tools": next_map.get(step, []),
            })
        elif step == "construct_argument_graph":
            env = construct_argument_graph({"structure": current_structure})
            r = env.get("data", {}) if isinstance(env, dict) else env
            current_graph = r
            stages.append({
                "name": step,
                "inputs_summary": {"nodes": len(current_structure.get("nodes", [])), "links": len(current_structure.get("links", []))},
                "key_outputs": {"nodes": len(r.get("nodes", [])), "links": len(r.get("links", []))},
                "next_tools": next_map.get(step, []),
            })
        elif step == "validate_argument_graph":
            env = validate_argument_graph(current_graph)
            r = env.get("data", {}) if isinstance(env, dict) else env
            issues = r.get("issues", {}) if isinstance(r, dict) else {}
            issue_keys = list(issues.keys()) if issues else []
            stages.append({
                "name": step,
                "inputs_summary": {"nodes": len(current_graph.get("nodes", [])), "links": len(current_graph.get("links", []))},
                "key_outputs": {"issues": issue_keys},
                "next_tools": next_map.get(step, []),
            })
        elif step == "assess_argument_quality":
            env = assess_argument_quality(current_graph)
            r = env.get("data", {}) if isinstance(env, dict) else env
            current_assessments = r
            strengths = r.get("strengths", [])
            weaknesses = r.get("weaknesses", [])
            stages.append({
                "name": step,
                "inputs_summary": {"nodes": len(current_graph.get("nodes", []))},
                "key_outputs": {"strengths": len(strengths), "weaknesses": len(weaknesses)},
                "next_tools": next_map.get(step, []),
            })
        elif step == "identify_reasoning_weaknesses":
            env = identify_reasoning_weaknesses({"patterns": current_patterns, "text": argument_text})
            r = env.get("data", {}) if isinstance(env, dict) else env
            findings = r.get("weaknesses", [])
            stages.append({
                "name": step,
                "inputs_summary": {"patterns": len(current_patterns)},
                "key_outputs": {"weaknesses": len(findings)},
                "next_tools": next_map.get(step, []),
            })
        elif step == "generate_counter_analysis":
            env = generate_counter_analysis(current_graph)
            r = env.get("data", {}) if isinstance(env, dict) else env
            counters = r.get("counter_points", [])
            stages.append({
                "name": step,
                "inputs_summary": {"nodes": len(current_graph.get("nodes", []))},
                "key_outputs": {"counter_points": len(counters)},
                "next_tools": next_map.get(step, []),
            })

    response: Dict[str, Any] = {
        "stages": stages,
        "final_artifacts": {
            "structure": current_structure or {},
            "patterns": current_patterns or [],
            "assumptions": current_assumptions or [],
            "graph": current_graph or {},
            "assessments": current_assessments or {},
            "probes": current_probes or {},
        },
    }

    if isinstance(max_steps, int) and max_steps >= 0 and len(stages) < len(sequence):
        response["truncated"] = True

    out = _maybe_envelope(response, "analyze_argument_stepwise", compat=compat)
    _log_event("analyze_argument_stepwise", time.perf_counter(), len(argument_text), len(json.dumps(out)))
    return out


@mcp.tool
def compare_arguments(argument_graphs: List[Dict[str, Any]], comparison_dimensions: Optional[List[str]] = None, compat: Optional[Any] = None) -> Dict[str, Any]:
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
    out = _maybe_envelope({"comparison": comps}, "compare_arguments", compat=compat)
    _log_event("compare_arguments", time.perf_counter(), len(json.dumps(argument_graphs)), len(json.dumps(out)))
    return out


@mcp.tool
def assess_argument_quality(argument_graph: Dict[str, Any], assessment_framework: str = "comprehensive", compat: Optional[Any] = None) -> Dict[str, Any]:
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
    # Sub-scores
    coverage = min(1.0, (len([n for n in nodes if n.get("primary_subtype") == "Main Claim"])>0) + (len([n for n in nodes if n.get("primary_subtype") == "Premise"])>0) + (any(n.get("assumptions") for n in nodes)))/3.0
    balance = 1.0 if any("Counterexample" in s for s in strengths) or any("rebut" in (n.get("content","")) for n in nodes) else 0.4
    clarity = 1.0 if all(len((n.get("content") or "")) <= 280 for n in nodes) else 0.6
    support_rigor = 1.0 if any(n.get("evidence_type") for n in nodes) else 0.5
    rubric = {"coverage": coverage, "balance": balance, "clarity": clarity, "support_rigor": support_rigor}
    meta = {"rubric": rubric, "weights": {"coverage": 0.35, "balance": 0.25, "clarity": 0.2, "support_rigor": 0.2}}
    out = _maybe_envelope({"strengths": strengths, "weaknesses": weaknesses}, "assess_argument_quality", compat=compat, )
    # attach meta if enveloped
    if isinstance(out, dict) and "metadata" in out:
        out["metadata"]["rubric"] = meta.get("rubric")
        out["metadata"]["weights"] = meta.get("weights")
    _log_event("assess_argument_quality", time.perf_counter(), len(json.dumps(argument_graph)), len(json.dumps(out)))
    return out


class OntologyGroundedWeaknessDetector:
    def __init__(self, ontology) -> None:
        self.ontology = ontology
        self.pattern_to_ontology = {
            "authority": {"bucket": "Appeals / Irrelevant Persuasion"},
            "causal": {"bucket": "Causal / Statistical Confusions"},
            "analogical": {"bucket": "Other / Check"},
        }

    def _find_overlapping_nodes(self, span: List[int], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(span, list) or len(span) != 2:
            return []
        a = (int(span[0]), int(span[1]))
        out: List[Dict[str, Any]] = []
        for n in nodes:
            sp = n.get("source_text_span")
            if isinstance(sp, list) and len(sp) == 2:
                b = (int(sp[0]), int(sp[1]))
                if a[0] < b[1] and b[0] < a[1]:
                    out.append(n)
        return out

    def detect(self, argument_analysis: Dict[str, Any], sensitivity: str = "default") -> List[Dict[str, Any]]:
        patterns = argument_analysis.get("patterns", [])
        nodes = (argument_analysis.get("structure") or {}).get("nodes", [])
        cutoff = {"low": 0.1, "default": 0.2, "high": 0.3}.get(sensitivity, 0.2)
        findings: List[Dict[str, Any]] = []
        for p in patterns:
            if p.get("confidence", 0.0) < cutoff:
                continue
            ptype = p.get("pattern_type")
            span = p.get("source_text_span")
            node_ids = [n.get("node_id") for n in self._find_overlapping_nodes(span, nodes)]
            if ptype == "authority":
                findings.append({
                    "label": "Appeal to Authority",
                    "type": "epistemic",
                    "severity": "medium",
                    "why": "Relies on expert opinion without establishing credibility or consensus",
                    "how": "Verify credentials, check bias, examine peer consensus",
                    "node_ids": node_ids,
                    "span": span,
                    "confidence": 0.8,
                    "ontology_bucket": self.pattern_to_ontology["authority"]["bucket"],
                })
            elif ptype == "causal":
                findings.append({
                    "label": "Post Hoc Ergo Propter Hoc",
                    "type": "logical",
                    "severity": "high",
                    "why": "Assumes temporal sequence implies causation",
                    "how": "Verify mechanism, test alternative explanations, check for confounders",
                    "node_ids": node_ids,
                    "span": span,
                    "confidence": 0.7,
                    "ontology_bucket": self.pattern_to_ontology["causal"]["bucket"],
                })
            elif ptype == "analogical":
                findings.append({
                    "label": "Weak Analogy",
                    "type": "logical",
                    "severity": "medium",
                    "why": "Comparison may not hold for relevant aspects",
                    "how": "Identify differences and test analogy boundaries",
                    "node_ids": node_ids,
                    "span": span,
                    "confidence": 0.6,
                    "ontology_bucket": self.pattern_to_ontology["analogical"]["bucket"],
                })
        return findings[:MAX_WEAKNESSES]

@mcp.tool
def identify_reasoning_weaknesses(argument_analysis: Dict[str, Any], weakness_categories: Optional[List[str]] = None, sensitivity: str = "default", compat: Optional[Any] = None) -> Dict[str, Any]:
    if not (argument_analysis.get("patterns") or argument_analysis.get("text")):
        err = _error("INVALID_INPUT_SHAPE", "Missing argument_analysis", where="identify_reasoning_weaknesses")
        out = _maybe_envelope(None, "identify_reasoning_weaknesses", compat=compat)
        out["error"] = err
        _log_event("identify_reasoning_weaknesses", time.perf_counter(), 0, len(json.dumps(out)))
        return out
    patterns = argument_analysis.get("patterns") or []
    if not patterns and argument_analysis.get("text"):
        s2 = _ENGINE.stage2_patterns(argument_analysis["text"], [])
        patterns = s2["patterns"]
        argument_analysis = dict(argument_analysis)
        argument_analysis["patterns"] = patterns
    detector = OntologyGroundedWeaknessDetector(_ONTO)
    weaknesses = detector.detect(argument_analysis, sensitivity=sensitivity)
    if weakness_categories:
        weaknesses = [w for w in weaknesses if w.get("type") in weakness_categories]
    out = _maybe_envelope({"weaknesses": weaknesses}, "identify_reasoning_weaknesses", compat=compat)
    _log_event("identify_reasoning_weaknesses", time.perf_counter(), len(json.dumps(argument_analysis)), len(json.dumps(out)))
    return out


@mcp.tool
def generate_counter_analysis(argument_graph: Dict[str, Any], analysis_type: str = "comprehensive", compat: Optional[Any] = None) -> Dict[str, Any]:
    """Generate structured counter-briefs: weaknesses, alternatives, or steel-man targets."""
    nodes = argument_graph.get("nodes") or []
    main = next((n for n in nodes if n.get("primary_subtype") == "Main Claim"), None)
    counters: List[str] = []
    if main:
        claim = main.get("content", "")
        counters.append(f"Alternative explanation for claim: challenge causal/mechanistic link in '{claim}'.")
        counters.append("Undercut warrant: show expert scope misalignment or COI.")
        counters.append("Counterexample: provide case where premises hold but conclusion fails.")
    out = _maybe_envelope({"counter_points": counters}, "generate_counter_analysis", compat=compat)
    _log_event("generate_counter_analysis", time.perf_counter(), len(json.dumps(argument_graph)), len(json.dumps(out)))
    return out


@mcp.tool
def map_assumptions_to_nodes(analysis_results: Dict[str, Any], assumptions: List[Dict[str, str]], strategy: str = "best-match", compat: Optional[Any] = None) -> Dict[str, Any]:
    """Map assumptions to nearest nodes by token overlap; returns mappings and unmapped.

    strategy: "best-match" (default) or "strict" where a minimum score threshold is required.
    """
    start = time.perf_counter()
    # Unwrap envelope if provided
    ar = analysis_results.get("data") if isinstance(analysis_results, dict) and "data" in analysis_results else analysis_results
    structure = (ar.get("structure") or ar) if isinstance(ar, dict) else {}
    nodes = structure.get("nodes", [])
    mappings: List[Dict[str, Any]] = []
    unmapped: List[str] = []
    threshold = 0.2 if strategy == "best-match" else 0.35
    for a in assumptions or []:
        atext = a.get("text", "") if isinstance(a, dict) else str(a)
        if not atext:
            continue
        best = None
        best_score = 0.0
        a_tokens = set(atext.lower().split())
        for n in nodes:
            content = (n.get("content") or "").lower()
            if not content:
                continue
            n_tokens = set(content.split())
            inter = len(a_tokens & n_tokens)
            union = len(a_tokens | n_tokens) or 1
            score = inter / union
            if score > best_score:
                best_score = score
                best = n.get("node_id")
        if best is not None and best_score >= threshold:
            mappings.append({"assumption": atext, "node_id": best, "score": round(best_score, 3)})
        else:
            reason = "below_threshold" if best_score < threshold else "no_match"
            unmapped.append({"assumption": atext, "reason": reason, "score": round(best_score, 3)})
    out = _maybe_envelope({"mappings": mappings, "unmapped": unmapped, "strategy": strategy}, "map_assumptions_to_nodes", compat=compat)
    _log_event("map_assumptions_to_nodes", start, len(json.dumps(analysis_results)), len(json.dumps(out)))
    return out


@mcp.tool
def export_graph(graph: Dict[str, Any], format: str = "mermaid", analysis_id: Optional[str] = None, compat: Optional[Any] = None) -> Dict[str, Any]:
    """Export graph in mermaid|graphviz|jsonld formats."""
    start = time.perf_counter()
    gr = graph.get("data") if isinstance(graph, dict) and "data" in graph else graph
    nodes = (gr or {}).get("nodes", [])
    links = (gr or {}).get("links", [])
    fmt = (format or "mermaid").lower()
    allowed = {"mermaid", "graphviz", "jsonld"}
    if fmt not in allowed:
        err = _error("UNSUPPORTED_FORMAT", f"Unsupported format '{format}'", hint="Use mermaid|graphviz|jsonld", where="export_graph")
        out = _maybe_envelope(None, "export_graph", compat=compat)
        out["error"] = err
        _log_event("export_graph", start, len(json.dumps(graph)), len(json.dumps(out)))
        return out
    content = ""
    if fmt == "mermaid":
        lines = ["graph TD;"]
        for n in nodes:
            nid = n.get("node_id")
            label = ((n.get("primary_subtype") or "") + ": " + (n.get("content") or "")).strip()
            subtype = (n.get("primary_subtype") or "").lower()
            # color by subtype
            style = "fill:#e3f2fd,stroke:#1e88e5" if "main claim" in subtype else ("fill:#e8f5e9,stroke:#43a047" if "premise" in subtype else "fill:#f3e5f5,stroke:#8e24aa")
            safe = label[:80].replace("\\", "").replace("\"", "\\\"")
            lines.append(f"  {nid}[\"{safe}\"]; ")
            lines.append(f"  style {nid} {style}")
        for l in links:
            lines.append(f"  {l.get('source_node')} --> {l.get('target_node')};")
        content = "\n".join(lines)
    elif fmt == "jsonld":
        # add minimal typing
        jnodes = []
        for n in nodes:
            nd = dict(n)
            nd["@type"] = "ArgumentNode"
            jnodes.append(nd)
        jlinks = []
        for l in links:
            ld = dict(l)
            ld["@type"] = "ArgumentLink"
            jlinks.append(ld)
        content = json.dumps({"@context": {"@vocab": "https://example.org/arg#"}, "@graph": {"nodes": jnodes, "links": jlinks}}, ensure_ascii=False)
    else:  # graphviz
        lines = ["digraph G {"]
        for n in nodes:
            nid = n.get("node_id")
            label = ((n.get("primary_subtype") or "") + ": " + (n.get("content") or "")).strip()
            subtype = (n.get("primary_subtype") or "").lower()
            color = "#1e88e5" if "main claim" in subtype else ("#43a047" if "premise" in subtype else "#8e24aa")
            safe = label[:80].replace("\\", "").replace("\"", "\\\"")
            lines.append(f"  {nid} [label=\"{safe}\", color=\"{color}\", style=\"filled\"]; ")
        for l in links:
            lines.append(f"  {l.get('source_node')} -> {l.get('target_node')};")
        lines.append("}")
        content = "\n".join(lines)
    out = _maybe_envelope({"format": fmt, "content": content}, "export_graph", compat=compat)
    _log_event("export_graph", start, len(json.dumps(graph)), len(json.dumps(out)))
    return out


@mcp.tool
def analyze_and_probe(argument_text: str, analysis_depth: str = "comprehensive", audience: Optional[str] = None, goal: Optional[str] = None, confidence_threshold: float = 0.2, compat: Optional[Any] = None) -> Dict[str, Any]:
    """One-shot: analyze, infer assumptions, weaknesses, and probe plan."""
    start = time.perf_counter()
    ctx = AnalysisContext(forum=None, audience=audience, goal=goal, depth=analysis_depth)
    base = _ENGINE.analyze_comprehensive(argument_text, ctx)
    weak_env = identify_reasoning_weaknesses({"patterns": base.get("patterns", []), "text": argument_text})
    weaknesses = (weak_env.get("data") or {}).get("weaknesses", []) if isinstance(weak_env, dict) else []
    probes_env = orchestrate_probe_analysis({"patterns": base.get("patterns", []), "structure": base.get("structure", {})}, forum=None, audience=audience, goal=goal)
    probe_plan = (probes_env.get("data") or {}).get("probe_plan", []) if isinstance(probes_env, dict) else []
    data = {
        "structure": base.get("structure", {}),
        "patterns": base.get("patterns", []),
        "assumptions": base.get("assumptions", []),
        "weaknesses": weaknesses,
        "probe_plan": probe_plan,
    }
    out = _maybe_envelope(data, "analyze_and_probe", compat=compat)
    _log_event("analyze_and_probe", start, len(argument_text), len(json.dumps(out)))
    return out


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

@mcp.resource("argument://tools_index")
def tools_index() -> Dict[str, Any]:
    registry = []
    for name in _get_tool_registry():
        registry.append({
            "name": name,
            "canonical_id": name.lower().replace(" ", "_").replace("™", ""),
            "uri": f"argument://tools/{name}",
            "schema_version": API_VERSION,
        })
    return {"tools": registry}


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
