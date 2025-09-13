from __future__ import annotations

"""Domain profiles that adapt requirement weights and probes."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DomainProfile:
    name: str
    requirement_weights: Dict[str, float]
    extra_requirements: Dict[str, List[str]]
    preferred_probes: List[str]


GENERAL_PROFILE = DomainProfile(
    name="general",
    requirement_weights={},
    extra_requirements={},
    preferred_probes=[],
)

LEGAL_PROFILE = DomainProfile(
    name="legal",
    requirement_weights={"precedent": 2.0, "burden_of_proof": 1.5},
    extra_requirements={"authority": ["chain_of_authority"], "causal": ["statutory_link"]},
    preferred_probes=["case_law_search"],
)

SCIENTIFIC_PROFILE = DomainProfile(
    name="scientific",
    requirement_weights={"methodology": 2.0, "replication": 1.5},
    extra_requirements={"causal": ["statistical_power"]},
    preferred_probes=["literature_search"],
)

POLICY_PROFILE = DomainProfile(
    name="policy",
    requirement_weights={"stakeholder": 1.5, "feasibility": 1.2},
    extra_requirements={"practical": ["cost_benefit", "implementation_risk"]},
    preferred_probes=["policy_database"],
)

PROFILES = {
    p.name: p
    for p in [GENERAL_PROFILE, LEGAL_PROFILE, SCIENTIFIC_PROFILE, POLICY_PROFILE]
}
