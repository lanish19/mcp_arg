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

    def chain_probes_conditionally(self, initial_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Build a conditional chain of probe suggestions
        patterns = initial_findings.get("patterns", [])
        probes = self.select_probes_by_pattern(patterns)
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
                "rationale": "Selected by detected pattern and forum weighting.",
            }
            sequence.append(step)
            prev = p["name"]
        return sequence

