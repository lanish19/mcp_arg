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

