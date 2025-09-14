from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import logging

from .structures import ArgumentNode, ArgumentLink, NodePropertyAssigner, ArgumentGraph

from .patterns import PatternDetector, Pattern
from .gap import InferenceEngine, AssumptionGenerator
from .ontology import Ontology, ToolCatalog
from .probes import ProbeOrchestrator, ContextAwareProbeOrchestrator
from .domain_profiles import PROFILES, DomainProfile
from .validation import check_factual_consistency, check_sentiment_coherence, check_plausibility


@dataclass
class AnalysisContext:
    forum: Optional[str] = None
    audience: Optional[str] = None
    goal: Optional[str] = None
    depth: str = "standard"  # minimal, standard, thorough, exhaustive


class AnalysisEngine:
    def __init__(self, ontology: Ontology, tools: ToolCatalog, pattern_top_n: int = 5) -> None:
        self.ontology = ontology
        self.tools = tools
        self.detector = PatternDetector(ontology, top_n=pattern_top_n)
        self.assigner = NodePropertyAssigner(ontology)
        self.probes = ProbeOrchestrator(tools)
        self.context_probes = ContextAwareProbeOrchestrator(tools)
        self.profile: DomainProfile = PROFILES["general"]
        self.infer = InferenceEngine(ontology, self.profile)
        self.assumption_generator = AssumptionGenerator(ontology)
        # backward compatibility: expose inference engine as 'gap'
        self.gap = self.infer


    # Stage 1: Structural Decomposition
    def stage1_decompose(self, text: str) -> Dict[str, Any]:
        # Use EnhancedSpanTracker to get sentences with reliable spans from the start.
        span_tracker = EnhancedSpanTracker(text)
        sentences_with_spans = span_tracker._split_sentences_with_positions()
        sentences = [s["text"] for s in sentences_with_spans]

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
        for i, s_info in enumerate(sentences_with_spans):
            s = s_info["text"]
            span = (s_info["start"], s_info["end"])

            if i == concl_ix:
                n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Main Claim")
                n.confidence = 0.7
            else:
                # Mark premise or evidence based on keywords
                if re.search(r"\b(study|studies|data|report|research|shows|suggests|indicates|according to)\b", s, re.I):
                    n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Evidence")
                    n.confidence = 0.65
                elif re.search(r"\b(because|since|should|ought|due to)\b", s, re.I):
                    n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Premise")
                    n.confidence = 0.6
                else:
                    n = ArgumentNode.make("STATEMENT", content=s.strip(), primary_subtype="Statement")
                    n.confidence = 0.5
            # Attach source span directly. It is guaranteed to be meaningful.
            n.source_text_span = span
            nodes.append(n)

        # Link premises to conclusion as SUPPORT (linked if they contain connective cues)
        if concl_ix is not None:
            target_id = nodes[concl_ix].node_id
            for i, n in enumerate(nodes):
                if i == concl_ix:
                    continue
                link = ArgumentLink.make(n.node_id, target_id, "SUPPORT", relationship_subtype="convergent")
                links.append(link)

        # The span assignment is now robust and integrated, no need for the final check.
        return {"sentences": sentences, "nodes": [n.to_dict() for n in nodes], "links": [l.to_dict() for l in links]}

    # Stage 2: Multi-Dimensional Pattern Recognition
    def stage2_patterns(self, text: str, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw = self.detector.detect(text)
        patterns = []
        for p in raw:
            d = asdict(p)
            # Preserve source span
            if d.get("span") is not None:
                sp = d.pop("span")
                d["source_text_span"] = [sp[0], sp[1]]
            if not d.get("roles"):
                d.pop("roles", None)
            if isinstance(d.get("confidence"), float):
                d["confidence"] = round(float(d["confidence"]), 3)
            # Keep only minimal details
            details = d.get("details") or {}
            if isinstance(details, dict):
                d["details"] = {k: details[k] for k in ("scheme", "score") if k in details}
            patterns.append(d)
        for n in nodes:
            self.assigner.assign_comprehensive_properties(n, text, patterns)
        return {"patterns": patterns, "nodes": nodes}

    # Stage 3: Systematic Gap Analysis
    def stage3_infer(self, text: str, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        evaluations: List[Dict[str, Any]] = []
        enriched_assumptions: List[Dict[str, Any]] = []
        # Heuristic mapping for tests suggestions and impact
        def tests_for(scheme: str) -> List[str]:
            s = (scheme or "").lower()
            if "cause" in s:
                return ["timeline_check", "mechanism_specification", "confound_audit"]
            if "expert" in s or "authority" in s:
                return ["credential_verification", "bias_disclosure", "consensus_scan"]
            if "analogy" in s:
                return ["similarity_dimensions_spec", "scope_limitations"]
            if "practical" in s or "policy" in s:
                return ["goal_clarity", "cost_benefit_outline"]
            return ["evidence_request"]

        def impact_for(ptype: str) -> str:
            if ptype in ("causal", "authority"):
                return "high"
            if ptype in ("analogical", "normative"):
                return "med"
            return "low"

        for p in patterns[:5]:
            p = dict(p)
            p.setdefault("scheme", (p.get("details") or {}).get("scheme") or p.get("label"))
            p.setdefault("text", text)
            eval_res = self.infer.evaluate_scheme(p)
            evaluations.append(
                {
                    "scheme": eval_res.scheme,
                    "confidence": round(float(eval_res.confidence), 3),
                    "requirements": [
                        {
                            "name": r.requirement.name,
                            "satisfied": r.satisfied,
                            "score": round(float(r.score), 3),
                            "missing_premise": r.missing_premise,
                        }
                        for r in eval_res.requirements
                    ],
                }
            )
            # Build enriched assumptions based on evaluation plus pattern-driven generator
            ptype = p.get("pattern_type") or ""
            linked = [p.get("pattern_id")] if p.get("pattern_id") else []
            for atext in eval_res.generated_assumptions:
                if not atext:
                    continue
                enriched_assumptions.append(
                    {
                        "text": atext,
                        "linked_patterns": linked,
                        "impact": impact_for(ptype),
                        "confidence": round(float(eval_res.confidence), 3),
                        "tests": tests_for(p.get("scheme") or ""),
                    }
                )
        # Add assumption candidates directly from patterns for coverage
        enriched_assumptions.extend(self.assumption_generator.generate(text, patterns, None))
        # Deduplicate by text while preserving order
        seen_texts = set()
        deduped: List[Dict[str, Any]] = []
        for a in enriched_assumptions:
            t = a.get("text")
            if t and t not in seen_texts:
                deduped.append(a)
                seen_texts.add(t)
        return {"evaluations": evaluations, "assumptions": deduped}

    # Stage 4: Dynamic Probe Orchestration
    def stage4_probes(self, context: AnalysisContext, patterns: List[Dict[str, Any]], structure: Optional[Dict[str, Any]] = None, weaknesses: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        analysis_results = {"patterns": patterns, "structure": structure or {}, "weaknesses": weaknesses or [], "text": ""}
        plan = self.context_probes.generate_probe_plan(analysis_results, forum=context.forum, audience=context.audience, goal=context.goal)
        return plan

    # Stage 5: Integration and Validation
    def stage5_integrate(self, text: str, nodes: List[Dict[str, Any]], links: List[Dict[str, Any]], patterns: List[Dict[str, Any]], infer_res: Dict[str, Any]) -> Dict[str, Any]:
        graph = ArgumentGraph()
        for n in nodes:
            # Guard: ensure minimal required keys
            try:
                graph.add_node(ArgumentNode.from_dict(n))
            except Exception as e:
                logging.warning(f"Failed to add node from dict {n}: {e}", exc_info=True)
        for l in links:
            try:
                graph.add_link(ArgumentLink.from_dict(l))
            except Exception as e:
                logging.warning(f"Failed to add link from dict {l}: {e}", exc_info=True)
        for asm in infer_res["assumptions"]:
            a_text = asm if isinstance(asm, str) else (asm.get("text", "") if isinstance(asm, dict) else str(asm))
            if not a_text:
                continue
            a_node = ArgumentNode.make("STATEMENT", content=a_text, primary_subtype="Assumption")
            graph.add_node(a_node)
        structure_issues = graph.validate_structure()
        claims = [n.get("content", "") for n in nodes]
        conflicts = check_factual_consistency(claims)
        senti_issues = check_sentiment_coherence(claims)
        # Compute plausibility on assumption text, not raw dicts
        plaus = []
        for a in infer_res["assumptions"]:
            a_text = a if isinstance(a, str) else (a.get("text", "") if isinstance(a, dict) else str(a))
            if a_text:
                plaus.append(check_plausibility(a_text))
            else:
                plaus.append((0.5, "unknown"))
        # Normalize assumptions payload for narrative builder
        narr_assumptions: List[Dict[str, Any]] = []
        for a in infer_res["assumptions"]:
            if isinstance(a, dict):
                narr_assumptions.append({"text": a.get("text", ""), "category": a.get("impact", "")})
            else:
                narr_assumptions.append({"text": str(a)})
        narrative = self._build_narrative(text, nodes, patterns, narr_assumptions)
        return {
            "graph": graph.to_json(),
            "validation": {
                "structure": structure_issues,
                "contradictions": conflicts,
                "sentiment_mismatch": senti_issues,
                "plausibility": plaus,
            },
            "narrative": narrative,
        }

    # Public API
    def _run_once(self, text: str, context: AnalysisContext) -> Dict[str, Any]:
        self.profile = PROFILES.get(context.forum or "general", PROFILES["general"])
        self.infer.profile = self.profile
        s1 = self.stage1_decompose(text)
        s2 = self.stage2_patterns(text, s1["nodes"])
        s3 = self.stage3_infer(text, s2["patterns"])
        s4 = self.stage4_probes(context, s2["patterns"], structure={"nodes": s2["nodes"], "links": s1["links"]})
        s5 = self.stage5_integrate(text, s2["nodes"], s1["links"], s2["patterns"], s3)

        return {
            "context": asdict(context),
            "structure": {"nodes": s2["nodes"], "links": s1["links"]},
            "patterns": s2["patterns"],
            "evaluations": s3["evaluations"],
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
        issues: Dict[str, Any] = {"dangling_links": [], "isolated_nodes": [], "duplicate_edges": [], "cycles": []}
        seen_edges = set()
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for l in links:
            s = l.get("source_node"); t = l.get("target_node"); lid = l.get("link_id")
            if s not in node_ids or t not in node_ids:
                if lid:
                    issues["dangling_links"].append(lid)
                continue
            key = (s, t, l.get("relationship_type"))
            if key in seen_edges:
                if lid:
                    issues["duplicate_edges"].append(lid)
            else:
                seen_edges.add(key)
            adj.setdefault(s, []).append(t)
        # simple isolation check
        connected = set()
        for l in links:
            if l.get("source_node") in node_ids and l.get("target_node") in node_ids:
                connected.add(l["source_node"]); connected.add(l["target_node"])
        for n in nodes:
            if n["node_id"] not in connected:
                issues["isolated_nodes"].append(n["node_id"])
        # cycle detection (DFS)
        visited: Dict[str, int] = {}
        stack: Dict[str, bool] = {}
        cycle_found: List[List[str]] = []
        def dfs(u: str, path: List[str]):
            visited[u] = 1; stack[u] = True
            for v in adj.get(u, []):
                if v not in visited:
                    dfs(v, path + [v])
                elif stack.get(v):
                    cycle_found.append(path + [v])
            stack[u] = False
        for nid in node_ids:
            if nid not in visited:
                dfs(nid, [nid])
        if cycle_found:
            issues["cycles"] = cycle_found[:3]
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
                cat = a.get('category', '')
                lines.append(f"- {a['text']}" + (f" ({cat})" if cat else ""))
        return "\n".join(lines)


class EnhancedSpanTracker:
    def __init__(self, text: str) -> None:
        self.text = text or ""

    def _split_sentences_with_positions(self) -> List[Dict[str, Any]]:
        import re as _re
        sentences: List[Dict[str, Any]] = []
        current_pos = 0
        pattern = r"[.!?]+\s+"
        for m in _re.finditer(pattern, self.text):
            end_pos = m.start() + 1
            segment = self.text[current_pos:end_pos].strip()
            if segment:
                sentences.append({"text": segment, "start": current_pos, "end": end_pos})
            current_pos = m.end()
        if current_pos < len(self.text):
            tail = self.text[current_pos:].strip()
            if tail:
                sentences.append({"text": tail, "start": current_pos, "end": len(self.text)})
        return sentences

    def _find_content_span(self, content: str) -> Optional[Tuple[int, int]]:
        if not content:
            return None
        start = self.text.find(content)
        if start >= 0:
            return (start, start + len(content))
        start = self.text.lower().find(content.lower())
        if start >= 0:
            return (start, start + len(content))
        return self._fuzzy_span_match(content)

    def _fuzzy_span_match(self, content: str) -> Optional[Tuple[int, int]]:
        from .tfidf import TfidfVectorizer, cosine_similarity
        window_size = max(20, min(len(content) or 20, len(self.text)))
        windows: List[str] = []
        positions: List[Tuple[int, int]] = []
        step = max(5, window_size // 4)
        for i in range(0, max(1, len(self.text) - window_size + 1), step):
            win = self.text[i:i + window_size]
            windows.append(win)
            positions.append((i, i + window_size))
        if not windows:
            # Fallback for empty text or very short text
            return (0, len(self.text))
        try:
            vec = TfidfVectorizer().fit([content] + windows)
            mat = vec.transform([content] + windows)
            sims = cosine_similarity(mat[0], mat[1:]).flatten()
            best = int(sims.argmax()) if hasattr(sims, "argmax") else 0
            if float(sims[best]) > 0.3:
                return positions[best]
        except Exception:
            # If TF-IDF fails, we return None and let the caller decide on the ultimate fallback.
            pass
        return None
