"""
Multi-dimensional stakeholder relationship management.
Handles complex corporate political dynamics and relationship cascading.
"""
from typing import Dict, List, Any
from .data_models import Action, RelationshipDelta


class StakeholderRelationshipMatrix:
    """
    Complex relationship tracking between corporate stakeholders.
    Models realistic corporate political dynamics and influence networks.
    """

    def __init__(self):
        # Direct relationships: stakeholder -> player perception
        self.direct_relationships: Dict[str, Dict[str, Any]] = {
            "ceo": {
                "trust": 0.7,
                "respect": 0.8,
                "influence_on_you": 0.9,
                "communication_style": "strategic",
                "priorities": [
                    "board_perception",
                    "quarterly_results",
                    "innovation",
                ],
            },
            "it_director": {
                "trust": 0.9,
                "respect": 0.7,
                "influence_on_you": 0.3,
                "communication_style": "technical",
                "priorities": [
                    "system_stability",
                    "team_morale",
                    "technical_debt",
                ],
            },
            "facilities_manager": {
                "trust": 0.5,
                "respect": 0.6,
                "influence_on_you": 0.1,
                "communication_style": "procedural",
                "priorities": [
                    "vendor_relations",
                    "cost_control",
                    "compliance",
                ],
            },
            "admin_team": {
                "trust": 0.8,
                "respect": 0.5,
                "influence_on_you": 0.4,
                "communication_style": "collaborative",
                "priorities": [
                    "office_harmony",
                    "productivity_tools",
                    "social_events",
                ],
            },
            "sales_director": {
                "trust": 0.6,
                "respect": 0.6,
                "influence_on_you": 0.5,
                "communication_style": "metrics_focused",
                "priorities": [
                    "revenue",
                    "competitive_advantage",
                    "customer_acquisition",
                ],
            },
            "it_team": {
                "trust": 0.7,
                "respect": 0.65,
                "influence_on_you": 0.4,
                "communication_style": "technical",
                "priorities": [
                    "code_quality",
                    "deep_work",
                    "flexibility",
                ],
            },
            "security_director": {
                "trust": 0.65,
                "respect": 0.8,
                "influence_on_you": 0.6,
                "communication_style": "risk_focused",
                "priorities": [
                    "security_compliance",
                    "risk_management",
                    "regulatory_adherence",
                ],
            },
            "senior_developer": {
                "trust": 0.8,
                "respect": 0.75,
                "influence_on_you": 0.3,
                "communication_style": "technical",
                "priorities": [
                    "technical_integrity",
                    "professional_reputation",
                    "engineering_process",
                ],
            },
            "hr_director": {
                "trust": 0.7,
                "respect": 0.75,
                "influence_on_you": 0.35,
                "communication_style": "policy_focused",
                "priorities": [
                    "employee_welfare",
                    "legal_compliance",
                    "sustainable_work_practices",
                ],
            },
            "cfo": {
                "trust": 0.65,
                "respect": 0.7,
                "influence_on_you": 0.45,
                "communication_style": "financial_focused",
                "priorities": [
                    "cost_reduction",
                    "roi_optimization",
                    "financial_sustainability",
                ],
            },
            "admin_team": {
                "trust": 0.75,
                "respect": 0.7,
                "influence_on_you": 0.25,
                "communication_style": "process_oriented",
                "priorities": [
                    "workflow_continuity",
                    "system_familiarity",
                    "efficiency",
                ],
            },
            "marketing_team": {
                "trust": 0.68,
                "respect": 0.72,
                "influence_on_you": 0.3,
                "communication_style": "trend_focused",
                "priorities": [
                    "modern_features",
                    "user_experience",
                    "innovation",
                ],
            },
        }

        # Inter-stakeholder alliances
        self.stakeholder_alliances: Dict[tuple, Dict[str, Any]] = {
            ("ceo", "it_director"): {
                "alignment": 0.8,
                "trust": 0.6,
                "communication_frequency": "high",
                "conflict_areas": ["budget_priorities", "timeline_expectations"],
            },
            ("facilities_manager", "admin_team"): {
                "alignment": 0.9,
                "trust": 0.8,
                "communication_frequency": "daily",
                "conflict_areas": [],
            },
            ("ceo", "facilities_manager"): {
                "alignment": 0.3,
                "trust": 0.4,
                "communication_frequency": "low",
                "conflict_areas": ["cost_vs_quality", "vendor_selection"],
            },
            ("it_director", "admin_team"): {
                "alignment": 0.7,
                "trust": 0.8,
                "communication_frequency": "medium",
                "conflict_areas": ["system_access", "update_timing"],
            },
        }

        # Influence network / power dynamics
        self.influence_network: Dict[str, Dict[str, Any]] = {
            "ceo": {
                "influences": ["it_director", "facilities_manager"],
                "influenced_by": [],
                "power_level": 1.0,
                "decision_authority": ["budget", "strategy", "personnel"],
            },
            "it_director": {
                "influences": ["admin_team"],
                "influenced_by": ["ceo"],
                "power_level": 0.7,
                "decision_authority": ["technology_choices", "system_access"],
            },
            "facilities_manager": {
                "influences": [],
                "influenced_by": ["ceo", "admin_team"],
                "power_level": 0.4,
                "decision_authority": ["vendor_contracts", "office_policies"],
            },
            "admin_team": {
                "influences": ["facilities_manager"],
                "influenced_by": ["it_director"],
                "power_level": 0.3,
                "decision_authority": ["daily_operations", "social_coordination"],
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_relationships_from_action(self, action: Action) -> Dict[str, RelationshipDelta]:
        """Calculate cascading effects from an action (exact algorithm from spec)."""
        # ------------------------------------------------------------------
        # Override heuristics if author supplied explicit deltas
        # ------------------------------------------------------------------
        if action.relationship_deltas:
            deltas: Dict[str, RelationshipDelta] = {
                sid: RelationshipDelta(
                    stakeholder_id=sid,
                    delta_type="author_override",
                    magnitude=delta,
                    source_action=action.id,
                    reasoning="Explicit delta from scenario specification",
                )
                for sid, delta in action.relationship_deltas.items()
            }
        else:
            deltas = {}
            # Step 1 — direct effects
            for stakeholder_id in action.directly_affects:
                direct_delta = self._calculate_direct_relationship_change(stakeholder_id, action)
                deltas[stakeholder_id] = direct_delta

        # Step 2 — indirect effects through influence network
        for stakeholder_id in self.direct_relationships.keys():
            if stakeholder_id in deltas:
                continue
            indirect_delta = self._calculate_indirect_relationship_change(stakeholder_id, action, deltas)
            if abs(indirect_delta.magnitude) > 0.05:
                deltas[stakeholder_id] = indirect_delta

        # Step 3 — alliance-based effects
        alliance_deltas = self._calculate_alliance_effects(action, deltas)
        for stakeholder_id, delta in alliance_deltas.items():
            if stakeholder_id in deltas:
                deltas[stakeholder_id] = self._combine_deltas(deltas[stakeholder_id], delta)
            else:
                deltas[stakeholder_id] = delta

        # ------------------------------------------------------------------
        # APPLY the computed deltas to persistent state
        # ------------------------------------------------------------------
        for stakeholder_id, delta in deltas.items():
            metrics = self.direct_relationships.get(stakeholder_id)
            if not metrics:
                continue
            # Update trust as primary demo metric; could be extended to respect.
            new_trust = max(0.0, min(1.0, metrics["trust"] + delta.magnitude))
            metrics["trust"] = new_trust

        return deltas

    def get_relationship(self, stakeholder_id: str) -> Dict[str, float]:
        """Return relationship metrics (empty dict if unknown)."""
        return self.direct_relationships.get(stakeholder_id, {})
        
    def get_stakeholder_status(self, stakeholder_id: str) -> Dict[str, Any]:
        """Get current status of a stakeholder for display purposes.
        
        Returns:
            Dict with trust, respect, and influence metrics
        """
        relationship = self.direct_relationships.get(stakeholder_id, {})
        if not relationship:
            return {}
            
        return {
            "trust": relationship.get("trust", 0.5),
            "respect": relationship.get("respect", 0.5),
            "influence": relationship.get("influence_on_you", 0.5)
        }
    
    def export_relationships(self) -> Dict[str, Any]:
        """Export relationship data for save files.
        
        Returns:
            Dict with serializable relationship data
        """
        return {
            "direct_relationships": self.direct_relationships,
            # Export only the serializable parts of other relationship structures
            "stakeholder_alliances": {str(k): v for k, v in self.stakeholder_alliances.items()},
            "influence_network": self.influence_network
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate(self) -> "ValidationResult":  # type: ignore
        from .data_models import ValidationResult  # local import to avoid cycle

        errors: List[str] = []

        # Ensure every alliance stakeholder exists
        for pair in self.stakeholder_alliances.keys():
            for sid in pair:
                if sid not in self.direct_relationships:
                    errors.append(f"Alliance references unknown stakeholder '{sid}'")

        # Ensure influence network stakeholders consistent
        for sid, data in self.influence_network.items():
            if sid not in self.direct_relationships:
                errors.append(f"Influence network references unknown stakeholder '{sid}'")
            for influenced in data.get("influences", []):
                if influenced not in self.direct_relationships:
                    errors.append(f"Influence target '{influenced}' missing")

        return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=[])

    # ------------------------------------------------------------------
    # Internal helper methods (copied/adapted from spec).
    # ------------------------------------------------------------------
    def _calculate_direct_relationship_change(self, stakeholder_id: str, action: Action) -> RelationshipDelta:
        stakeholder = self.direct_relationships[stakeholder_id]
        priority_alignment = self._calculate_priority_alignment(action, stakeholder["priorities"])
        competence_signal = self._calculate_competence_signal(action, stakeholder)
        personal_impact = self._calculate_personal_impact(action, stakeholder_id)

        style = stakeholder["communication_style"]
        if style == "strategic":
            magnitude = priority_alignment * 0.6 + competence_signal * 0.3 + personal_impact * 0.1
        elif style == "technical":
            magnitude = competence_signal * 0.6 + priority_alignment * 0.3 + personal_impact * 0.1
        elif style == "procedural":
            procedure_compliance = self._calculate_procedure_compliance(action)
            magnitude = procedure_compliance * 0.5 + priority_alignment * 0.3 + personal_impact * 0.2
        else:  # collaborative
            magnitude = personal_impact * 0.5 + priority_alignment * 0.3 + competence_signal * 0.2

        return RelationshipDelta(
            stakeholder_id=stakeholder_id,
            delta_type="direct",
            magnitude=magnitude,
            source_action=action.id,
            reasoning=f"Direct impact: {action.description}",
        )

    def _calculate_indirect_relationship_change(
        self,
        stakeholder_id: str,
        action: Action,
        direct_deltas: Dict[str, RelationshipDelta],
    ) -> RelationshipDelta:
        influence_score = 0.0
        reasoning_parts: List[str] = []

        for affected_id, delta in direct_deltas.items():
            influence_strength = self._get_influence_between(affected_id, stakeholder_id)
            if influence_strength <= 0:
                continue

            alliance_key = (affected_id, stakeholder_id)
            reverse_key = (stakeholder_id, affected_id)
            alliance = self.stakeholder_alliances.get(alliance_key) or self.stakeholder_alliances.get(reverse_key)
            alliance_multiplier = 1.0
            if alliance:
                alliance_multiplier = alliance["alignment"] * alliance["trust"]

            ripple = delta.magnitude * influence_strength * alliance_multiplier * 0.3
            if ripple != 0:
                influence_score += ripple
                reasoning_parts.append(f"{affected_id} influence via alliance")

        return RelationshipDelta(
            stakeholder_id=stakeholder_id,
            delta_type="indirect",
            magnitude=influence_score,
            source_action=action.id,
            reasoning="Indirect effects: " + "; ".join(reasoning_parts),
        )

    def _calculate_alliance_effects(
        self, action: Action, current_deltas: Dict[str, RelationshipDelta]
    ) -> Dict[str, RelationshipDelta]:
        results: Dict[str, RelationshipDelta] = {}
        for (sid_a, sid_b), alliance in self.stakeholder_alliances.items():
            if sid_a in current_deltas and sid_b not in current_deltas:
                magnitude = current_deltas[sid_a].magnitude * alliance["alignment"] * alliance["trust"] * 0.2
                results[sid_b] = RelationshipDelta(
                    stakeholder_id=sid_b,
                    delta_type="alliance",
                    magnitude=magnitude,
                    source_action=action.id,
                    reasoning=f"Alliance ripple from {sid_a}",
                )
            elif sid_b in current_deltas and sid_a not in current_deltas:
                magnitude = current_deltas[sid_b].magnitude * alliance["alignment"] * alliance["trust"] * 0.2
                results[sid_a] = RelationshipDelta(
                    stakeholder_id=sid_a,
                    delta_type="alliance",
                    magnitude=magnitude,
                    source_action=action.id,
                    reasoning=f"Alliance ripple from {sid_b}",
                )
        return results

    def _combine_deltas(self, a: RelationshipDelta, b: RelationshipDelta) -> RelationshipDelta:
        return RelationshipDelta(
            stakeholder_id=a.stakeholder_id,
            delta_type="combined",
            magnitude=a.magnitude + b.magnitude,
            source_action=a.source_action,
            reasoning=f"{a.reasoning}; {b.reasoning}",
        )

    # ------------------------------------------------------------------
    # Lower-level calculation helpers (simple heuristic implementations).
    # ------------------------------------------------------------------
    def _calculate_priority_alignment(self, action: Action, priorities: List[str]) -> float:
        # Simple heuristic: +0.2 if action_type matches any priority keyword
        match = any(p in action.description.lower() for p in priorities)
        return 0.2 if match else -0.1

    def _calculate_competence_signal(self, action: Action, stakeholder: Dict[str, Any]) -> float:
        # Heuristic: difficulty inversely indicates perceived competence cost
        return max(0.0, 1.0 - action.difficulty)

    def _calculate_personal_impact(self, action: Action, stakeholder_id: str) -> float:
        # If stakeholder directly affected, positive if resource_cost low
        cost = sum(action.resource_cost.values()) if action.resource_cost else 0.0
        return 0.3 if stakeholder_id in action.directly_affects and cost < 1 else -0.2

    def _calculate_procedure_compliance(self, action: Action) -> float:
        # Placeholder compliance heuristic: communication actions are compliant
        return 0.5 if action.action_type == "communication" else -0.1

    def _get_influence_between(self, source: str, target: str) -> float:
        if source == target:
            return 0.0
        src_info = self.influence_network.get(source)
        if not src_info:
            return 0.0
        if target in src_info["influences"]:
            return src_info["power_level"] * 0.5
        return 0.0
