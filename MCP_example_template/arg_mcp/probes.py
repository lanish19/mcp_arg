from __future__ import annotations

from typing import List, Dict, Any, Optional

from .ontology import ToolCatalog


class ProbeOrchestrator:
    def __init__(self, tools: ToolCatalog):
        self.tools = tools

    def select_probes_by_pattern(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        kinds = {p.get("pattern_type") for p in patterns}
        picks: List[str] = []
        if "causal" in kinds:
            picks += [
                "The Causal Probe Playbook™",
                "Correlation–Causation Auditor™",
                "Quasi-experimental diagnostics (DiD/RDD/ITS)",
            ]
        if "authority" in kinds:
            picks += [
                "The Contextual Relevance Check™",
                "Multi-Source Validation Triangulation™",
            ]
        if "analogical" in kinds:
            picks += [
                "Toulmin Scaffold Mapper™",
                "Perturbation Test (Adversarial Thinking)™",
            ]
        if "normative" in kinds:
            picks += [
                "Occam’s Razor Slicer™ + Inversion Failure Forecaster™",
            ]
        # Always
        picks += [
            "Assumption Audit™ and Key Assumptions Scrubber™",
            "Triangulation Method Fusion™",
        ]
        out: List[Dict[str, str]] = []
        seen = set()
        for name in picks:
            meta = self.tools.get(name)
            if meta and meta["name"] not in seen:
                out.append(meta)
                seen.add(meta["name"])
        return out

    def prioritize_by_context(self, probes: List[Dict[str, str]], forum: Optional[str], audience: Optional[str], goal: Optional[str]) -> List[Dict[str, str]]:
        # Simple contextual weighting heuristic
        if not forum:
            return probes
        forum_l = forum.lower()
        order: List[Dict[str, str]] = []
        rest: List[Dict[str, str]] = []
        for p in probes:
            name_l = p["name"].lower()
            if "legal" in forum_l and any(k in name_l for k in ["toulmin", "assumption", "triangulation"]):
                order.append(p)
            elif "scientific" in forum_l and any(k in name_l for k in ["causal", "quasi-experimental", "replic", "triangulation"]):
                order.append(p)
            elif "policy" in forum_l and any(k in name_l for k in ["triangulation", "assumption", "toulmin"]):
                order.append(p)
            else:
                rest.append(p)
        return order + rest

    def chain_probes_conditionally(self, initial_findings: Dict[str, Any], profile=None) -> List[Dict[str, Any]]:
        # Build a conditional chain of probe suggestions
        patterns = initial_findings.get("patterns", [])
        probes = self.select_probes_by_pattern(patterns)
        # Thematic rulelets based on textual cues
        text_blob = " ".join([initial_findings.get("forum") or "", initial_findings.get("audience") or "", initial_findings.get("goal") or ""]).lower()
        pattern_texts = []
        for p in patterns:
            # harvest any available snippets from roles
            r = p.get("roles") or {}
            pattern_texts.append(" ".join([str(v) for v in r.values()]))
        text_blob += (" " + " ".join(pattern_texts)).lower()
        def add_tool(name: str):
            meta = self.tools.get(name)
            if meta:
                probes.append(meta)
        if any(k in text_blob for k in ["either or", "either/or", "two options", "no other choice", "three ways"]):
            add_tool("Dilemma Disassembler™")
        if any(k in text_blob for k in ["frozen conflict", "stalemate", "leadership vacuum", "power vacuum", "instability"]):
            add_tool("Stability & Conflict Stress Test™")
        if profile is not None:
            for name in profile.preferred_probes:
                meta = self.tools.get(name)
                if meta:
                    probes.insert(0, meta)

        ranked = self.prioritize_by_context(
            probes,
            initial_findings.get("forum"),
            initial_findings.get("audience"),
            initial_findings.get("goal"),
        )
        sequence: List[Dict[str, Any]] = []
        prev: Optional[str] = None
        for p in ranked:
            step = {
                "tool": p,
                "when": "after " + prev if prev else "initial",
                "rationale": "Selected by detected pattern, thematic cues, and forum weighting.",
            }
            sequence.append(step)
            prev = p["name"]
        # Ensure non-empty defaults and cap length
        if not sequence:
            fallback = [
                self.tools.get("Assumption Audit™ and Key Assumptions Scrubber™"),
                self.tools.get("Triangulation Method Fusion™"),
                self.tools.get("The Contextual Relevance Check™"),
            ]
            sequence = [{"tool": p, "when": "initial", "rationale": "Default probe"} for p in fallback if p]
        # Attach targets if nodes present
        nodes = (initial_findings.get("structure") or {}).get("nodes") or []
        target_ids = [n.get("node_id") for n in nodes if n.get("primary_subtype") in ("Main Claim", "Premise")]
        for s in sequence:
            s.setdefault("targets", target_ids[:5])
            # Attach why/how based on the selected tool metadata
            tmeta = s.get("tool") or {}
            s.setdefault("why", tmeta.get("purpose"))
            s.setdefault("how", tmeta.get("how"))
        # Cap to max 5
        return sequence[:5]

    def ranked_candidates(self, queries: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        # Delegate to ToolCatalog semantic search and merge results
        scored: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            res = self.tools.semantic_search_tools(q, threshold=0.12, max_results=max_results)
            for r in res:
                key = r["slug"]
                if key not in scored or r["score"] > scored[key]["score"]:
                    scored[key] = r
        return sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:max_results]


class ContextAwareProbeOrchestrator:
    def __init__(self, tool_catalog: ToolCatalog) -> None:
        self.tool_catalog = tool_catalog
        self.pattern_probe_map = {
            "authority": ["The Authority Credential Checker™", "The Consensus Vote Optimizer™"],
            "causal": ["The Causal Probe Playbook™", "The Mill's Methods Card Player™"],
            "analogical": ["The Analogy Fidelity Filter™", "The Structured Analogy Generator™"],
            "dilemmatic": ["The False Dilemma Detector™", "The Creative Synthesis Generator™"],
        }

    def _get_tool(self, name: str) -> Dict[str, Any]:
        return self.tool_catalog.get(name) or {}

    def _find_overlapping_nodes(self, span: list, nodes: list) -> list:
        if not isinstance(span, list) or len(span) != 2:
            return []
        a = (int(span[0]), int(span[1]))
        out = []
        for n in nodes or []:
            sp = n.get("source_text_span")
            if isinstance(sp, list) and len(sp) == 2:
                b = (int(sp[0]), int(sp[1]))
                if a[0] < b[1] and b[0] < a[1]:
                    out.append(n)
        return out

    def _select_pattern_probes(self, pattern: Dict[str, Any], structure: Dict[str, Any]) -> list:
        ptype = pattern.get("pattern_type", "other")
        nodes = (structure or {}).get("nodes", [])
        targets = [n.get("node_id") for n in self._find_overlapping_nodes(pattern.get("source_text_span"), nodes)]
        picks: list = []
        if ptype == "authority":
            picks.append({
                "tool": self._get_tool("The Authority Credential Checker™"),
                "when": "immediate",
                "rationale": "Authority-based argument detected - verify credibility",
                "targets": targets,
                "priority": "high",
            })
        elif ptype == "causal":
            picks.append({
                "tool": self._get_tool("The Causal Probe Playbook™"),
                "when": "immediate",
                "rationale": "Causal claim detected - test mechanism and alternatives",
                "targets": targets,
                "priority": "high",
            })
        elif ptype == "analogical":
            picks.append({
                "tool": self._get_tool("The Analogy Fidelity Filter™"),
                "when": "immediate",
                "rationale": "Analogy detected - test similarity boundaries",
                "targets": targets,
                "priority": "medium",
            })
        return [p for p in picks if p.get("tool")]

    def _has_limited_options_cues(self, text: str) -> bool:
        cues = ["three ways", "two options", "either/or", "only choice", "must choose", "no alternative", "either we", "only way"]
        return any(cue in (text or "").lower() for cue in cues)

    def _has_geopolitical_cues(self, text: str) -> bool:
        cues = ["leadership vacuum", "power vacuum", "geopolitical", "rogue actors", "global order", "instability", "crisis"]
        L = (text or "").lower()
        return any(c in L for c in cues)

    def _has_quantitative_cues(self, text: str) -> bool:
        import re
        pats = [r"\d+%", r"\d+\.\d+", r"\$\d+", r"\d+ times", r"increase.*\d+", r"decrease.*\d+", r"statistics show"]
        return any(re.search(p, text or "", re.I) for p in pats)

    def _select_context_probes(self, analysis_results: Dict[str, Any], forum: Optional[str], audience: Optional[str], goal: Optional[str]) -> list:
        probes: list = []
        text = analysis_results.get("text", "")
        if self._has_limited_options_cues(text):
            tool = self._get_tool("The False Dilemma Detector™")
            if tool:
                probes.append({"tool": tool, "when": "immediate", "rationale": "Limited options presented", "targets": [], "priority": "high"})
        if self._has_geopolitical_cues(text):
            tool = self._get_tool("The Second-Order Consequence Mapper™")
            if tool:
                probes.append({"tool": tool, "when": "followup", "rationale": "Geopolitical context - analyze downstream consequences", "targets": [], "priority": "medium"})
        if self._has_quantitative_cues(text):
            tool = self._get_tool("The Base Rate Reality Anchor™")
            if tool:
                probes.append({"tool": tool, "when": "immediate", "rationale": "Quantitative claims - ground in baselines", "targets": [], "priority": "medium"})
        return probes

    def _dedupe_rank(self, items: list) -> list:
        seen = set()
        out = []
        for it in items:
            key = (it.get("tool", {}).get("slug"), it.get("rationale"))
            if key not in seen and it.get("tool"):
                out.append(it)
                seen.add(key)
        return out

    def generate_probe_plan(self, analysis_results: Dict[str, Any], forum: Optional[str] = None, audience: Optional[str] = None, goal: Optional[str] = None) -> Dict[str, Any]:
        patterns = analysis_results.get("patterns", [])
        structure = analysis_results.get("structure", {})
        weaknesses = analysis_results.get("weaknesses", [])
        plan: list = []
        for p in patterns:
            plan.extend(self._select_pattern_probes(p, structure))
        plan.extend(self._select_context_probes(analysis_results, forum, audience, goal))
        plan = self._dedupe_rank(plan)
        # Add top semantic candidates as reference
        queries: list = []
        for p in patterns:
            queries.append(" ".join([p.get("pattern_type", ""), (p.get("details") or {}).get("scheme", ""), p.get("label", "")]))
        candidates = self.tool_catalog.semantic_search_tools(" ".join(queries), threshold=0.12, max_results=10)
        return {"probe_plan": plan[:5], "candidates": candidates}

