from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

from .ontology import Ontology
import uuid


NodeType = str  # STATEMENT, ARGUMENT, EVIDENCE, STRUCTURAL, META, QUALITY
RelationshipType = str  # SUPPORT, ATTACK, LOGICAL, CONTEXTUAL, QUALIFIER, REBUT, UNDERCUT


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class ArgumentNode:
    # Core identification
    node_id: str
    node_type: NodeType  # STATEMENT, ARGUMENT, EVIDENCE, STRUCTURAL, META, QUALITY
    primary_subtype: Optional[str] = None  # e.g., a specific bucket name

    # Multi-dimensional properties
    claim_type: Optional[str] = None
    evidence_type: Optional[str] = None
    argument_scheme: Optional[str] = None
    reasoning_pattern: Optional[str] = None
    cognitive_biases: List[str] = field(default_factory=list)
    fallacy_indicators: List[str] = field(default_factory=list)
    uncertainty_markers: List[str] = field(default_factory=list)

    # Contextual properties
    rhetorical_appeal: Optional[str] = None
    dialogical_frame: Optional[str] = None
    standard_of_proof: Optional[str] = None

    # Content and metadata
    content: str = ""
    confidence: float = 0.5
    source_text_span: Optional[Tuple[int, int]] = None
    assumptions: List[str] = field(default_factory=list)

    # Quality indicators
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    robustness_score: Optional[float] = None

    @staticmethod
    def make(node_type: NodeType, content: str = "", primary_subtype: Optional[str] = None) -> "ArgumentNode":
        return ArgumentNode(node_id=_gen_id("N"), node_type=node_type, content=content, primary_subtype=primary_subtype)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.source_text_span is not None:
            d["source_text_span"] = list(self.source_text_span)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArgumentNode":
        span = d.get("source_text_span")
        if isinstance(span, list):
            d = dict(d)
            d["source_text_span"] = (span[0], span[1])
        return ArgumentNode(**d)

    def validate(self) -> List[str]:
        issues = []
        if not self.node_id:
            issues.append("node_id missing")
        if not self.node_type:
            issues.append("node_type missing")
        if not isinstance(self.content, str):
            issues.append("content must be string")
        if not (0.0 <= float(self.confidence) <= 1.0):
            issues.append("confidence must be in [0,1]")
        return issues


@dataclass
class ArgumentLink:
    link_id: str
    source_node: str
    target_node: str
    relationship_type: RelationshipType  # SUPPORT, ATTACK, LOGICAL, CONTEXTUAL, etc.
    relationship_subtype: Optional[str] = None  # from Relationship Type dimension

    # Inference properties
    argument_scheme: Optional[str] = None
    reasoning_pattern: Optional[str] = None
    inference_rule: Optional[str] = None

    # Quality indicators
    strength: float = 0.5
    validity: Optional[bool] = None
    fallacy_risks: List[str] = field(default_factory=list)
    confidence: float = 0.5

    # Contextual properties
    conditions: List[str] = field(default_factory=list)
    scope_limitations: List[str] = field(default_factory=list)
    temporal_aspects: Optional[str] = None

    @staticmethod
    def make(source_node: str, target_node: str, relationship_type: RelationshipType, relationship_subtype: Optional[str] = None) -> "ArgumentLink":
        return ArgumentLink(link_id=_gen_id("L"), source_node=source_node, target_node=target_node, relationship_type=relationship_type, relationship_subtype=relationship_subtype)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArgumentLink":
        return ArgumentLink(**d)

    def validate(self) -> List[str]:
        issues = []
        if not self.link_id:
            issues.append("link_id missing")
        if not self.source_node:
            issues.append("source_node missing")
        if not self.target_node:
            issues.append("target_node missing")
        if not self.relationship_type:
            issues.append("relationship_type missing")
        if not (0.0 <= float(self.strength) <= 1.0):
            issues.append("strength must be in [0,1]")
        if not (0.0 <= float(self.confidence) <= 1.0):
            issues.append("confidence must be in [0,1]")
        return issues


class NodePropertyAssigner:
    """Assign ontology-informed properties across multiple dimensions to nodes."""

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology

    def assign_comprehensive_properties(
        self, node: Dict[str, Any], text_context: str, detected_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        dims = {
            "claim_type": "Claim Type",
            "evidence_type": "Evidence Type",
            "reasoning_pattern": "Reasoning Pattern",
            "cognitive_biases": "Cognitive Bias",
            "fallacy_indicators": "Fallacy",
        }
        content = node.get("content", "")
        for field, dim in dims.items():
            res = self.ontology.semantic_search(content, dimensions=[dim], threshold=0.15, max_results=1)
            if not res:
                continue
            bucket = res[0]["bucket"]
            if field in ("cognitive_biases", "fallacy_indicators"):
                node.setdefault(field, [])
                if bucket not in node[field]:
                    node[field].append(bucket)
            else:
                node[field] = bucket

        # incorporate detected patterns as reasoning hints
        if detected_patterns:
            for p in detected_patterns:
                if p.get("pattern_type") == "normative":
                    node.setdefault("reasoning_pattern", "Practical Reasoning")
                if p.get("pattern_type") == "causal":
                    node.setdefault("reasoning_pattern", "Causal")

        return node



@dataclass
class ArgumentGraph:
    nodes: Dict[str, ArgumentNode] = field(default_factory=dict)
    edges: List[ArgumentLink] = field(default_factory=list)

    def add_node(self, node: ArgumentNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: ArgumentLink) -> None:
        self.edges.append(edge)

    def validate_structure(self) -> List[str]:
        issues: List[str] = []
        ids = set(self.nodes)
        for e in self.edges:
            if e.source_node not in ids or e.target_node not in ids:
                issues.append(f"dangling edge {e.link_id}")
        return issues

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

