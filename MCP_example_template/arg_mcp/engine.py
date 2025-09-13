from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re

from .structures import ArgumentNode, ArgumentLink, NodePropertyAssigner
from .patterns import PatternDetector, Pattern
from .gap import GapAnalyzer
from .ontology import Ontology, ToolCatalog
from .probes import ProbeOrchestrator


@dataclass
class AnalysisContext:
    forum: Optional[str] = None
    audience: Optional[str] = None
    goal: Optional[str] = None
    depth: str = "standard"  # minimal, standard, thorough, exhaustive


class AnalysisEngine:
    def __init__(self, ontology: Ontology, tools: ToolCatalog) -> None:
        self.ontology = ontology
        self.tools = tools
        self.detector = PatternDetector(ontology)
        self.gap = GapAnalyzer()
        self.probes = ProbeOrchestrator(tools)
        self.assigner = NodePropertyAssigner(ontology)

    # Stage 1: Structural Decomposition
    def stage1_decompose(self, text: str) -> Dict[str, Any]:
        # Simple segmentation heuristics
        sentences = self._split_sentences(text)
        nodes: List[ArgumentNode] = []
        links: List[ArgumentLink] = []

        # Identify conclusion-like sentences
        concl_ix: Optional[int] = None
        for i, s in enumerate(sentences):
            if re.search(r"\b(therefore|thus|so|hence|conclude|in conclusion)\b", s, re.I):
                concl_ix = i
                break
        # Fallback: last sentence as main claim when none detected
        if concl_ix is None and sentences:
            concl_ix = len(sentences) - 1

        # Create nodes
        for i, s in enumerate(sentences):
            if i == concl_ix:
                n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Main Claim")
                n.confidence = 0.7
            else:
                # Mark premise if causal/normative keywords
                if re.search(r"\b(because|since|should|ought|due to)\b", s, re.I):
                    n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Premise")
                    n.confidence = 0.6
                else:
                    n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Statement")
                    n.confidence = 0.5
            nodes.append(n)

        # Link premises to conclusion as SUPPORT (linked if they contain connective cues)
        if concl_ix is not None:
            target_id = nodes[concl_ix].node_id
            for i, n in enumerate(nodes):
                if i == concl_ix:
                    continue
                link = ArgumentLink.make(n.node_id, target_id, "SUPPORT", relationship_subtype="convergent")
                links.append(link)

        return {"sentences": sentences, "nodes": [n.to_dict() for n in nodes], "links": [l.to_dict() for l in links]}

    # Stage 2: Multi-Dimensional Pattern Recognition
    def stage2_patterns(self, text: str, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        patterns = [asdict(p) for p in self.detector.detect(text)]
        for n in nodes:
            self.assigner.assign_comprehensive_properties(n, text, patterns)
        return {"patterns": patterns, "nodes": nodes}

    # Stage 3: Systematic Gap Analysis
    def stage3_gaps(self, patterns: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        missing = self.gap.analyze(patterns)
        # Attach assumptions to main claim if present, else to all nodes
        main_candidates = [n for n in nodes if n.get("primary_subtype") == "Main Claim"] or nodes
        for m in missing:
            main_candidates[0].setdefault("assumptions", []).append(m.text)
        return {"assumptions": [m.__dict__ for m in missing], "nodes": nodes}

    # Stage 4: Dynamic Probe Orchestration
    def stage4_probes(self, context: AnalysisContext, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        initial = {"patterns": patterns, "forum": context.forum, "audience": context.audience, "goal": context.goal}
        chain = self.probes.chain_probes_conditionally(initial)
        return {"probe_plan": chain}

    # Stage 5: Integration and Validation
    def stage5_integrate(self, text: str, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]], patterns: List[Dict[str, Any]], assumptions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build light validation and narrative
        issues = self._validate_graph(nodes, links)
        narrative = self._build_narrative(text, nodes, patterns, assumptions)
        return {"validation": issues, "narrative": narrative}

    # Public API
    def _run_once(self, text: str, context: AnalysisContext) -> Dict[str, Any]:
        s1 = self.stage1_decompose(text)
        s2 = self.stage2_patterns(text, s1["nodes"])
        s3 = self.stage3_gaps(s2["patterns"], s2["nodes"])
        s4 = self.stage4_probes(context, s2["patterns"])
        s5 = self.stage5_integrate(text, s3["nodes"], s1["links"], s2["patterns"], s3["assumptions"])
        return {
            "context": asdict(context),
            "structure": {"nodes": s3["nodes"], "links": s1["links"]},
            "patterns": s2["patterns"],
            "assumptions": s3["assumptions"],
            "probes": s4["probe_plan"],
            "integration": s5,
        }

    def cross_validate_stages(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        pattern_types = {p.get("pattern_type") for p in analysis.get("patterns", [])}
        nodes = analysis.get("structure", {}).get("nodes", [])
        unmatched = []
        for pt in pattern_types:
            if not any(pt.lower() in ((n.get("reasoning_pattern") or "").lower()) for n in nodes):
                unmatched.append(pt)
        return {"needs_refinement": bool(unmatched), "unmatched_patterns": unmatched}

    def refine_analysis(self, analysis: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        nodes = analysis.get("structure", {}).get("nodes", [])
        main = next((n for n in nodes if n.get("primary_subtype") == "Main Claim"), None)
        if main:
            for pt in validation.get("unmatched_patterns", []):
                if not main.get("reasoning_pattern"):
                    main["reasoning_pattern"] = pt.title()
        return analysis

    def analyze_comprehensive(self, text: str, context: AnalysisContext, max_iterations: int = 2) -> Dict[str, Any]:
        analysis = self._run_once(text, context)
        iteration = 0
        while iteration < max_iterations:
            validation = self.cross_validate_stages(analysis)
            if not validation.get("needs_refinement"):
                break
            analysis = self.refine_analysis(analysis, validation)
            iteration += 1
        return analysis

    # Utilities
    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _span_in_text(self, pat: Dict[str, Any], full_text: str, snippet: str) -> bool:
        # coarse: if snippet contains the matched token, say it applies
        return pat.get("details", {}).get("match", "").lower() in snippet.lower()

    def _validate_graph(self, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> Dict[str, Any]:
        node_ids = {n["node_id"] for n in nodes}
        issues = {"dangling_links": [], "isolated_nodes": []}
        for l in links:
            if l["source_node"] not in node_ids or l["target_node"] not in node_ids:
                issues["dangling_links"].append(l["link_id"])
        # simple isolation check
        connected = set()
        for l in links:
            connected.add(l["source_node"]); connected.add(l["target_node"])
        for n in nodes:
            if n["node_id"] not in connected:
                issues["isolated_nodes"].append(n["node_id"])
        return issues

    def _build_narrative(self, text: str, nodes: List[Dict[str, Any]], patterns: List[Dict[str, Any]], assumptions: List[Dict[str, Any]]) -> str:
        lines = []
        lines.append("Argument Synopsis")
        lines.append(f"Text: {text[:200]}{'…' if len(text)>200 else ''}")
        main = next((n for n in nodes if n.get("primary_subtype") == "Main Claim"), None)
        if main:
            lines.append(f"Main Claim: {main.get('content')}")
        supports = [n for n in nodes if n.get("primary_subtype") == "Premise"]
        if supports:
            lines.append("Premises:")
            for i, p in enumerate(supports, 1):
                lines.append(f"- P{i}: {p.get('content')}")
        if patterns:
            lines.append("Detected Patterns:")
            for p in patterns[:8]:
                lines.append(f"- {p['pattern_type']} — {p['details'].get('match')} (conf {p['confidence']:.2f})")
        if assumptions:
            lines.append("Key Assumptions:")
            for a in assumptions[:6]:
                lines.append(f"- {a['text']} ({a['category']})")
        return "\n".join(lines)

