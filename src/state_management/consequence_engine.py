"""Consequence calculation engine for cascading corporate dynamics."""
from typing import Dict, Any

from .data_models import Action, RelationshipDelta, ConsequenceSet
from .stakeholder_matrix import StakeholderRelationshipMatrix


class ConsequenceEngine:
    """Computes the downstream effects of actions on the game state.

    Initial implementation focuses on immediate relationship consequences only.
    Additional resource/scenario/behavior logic will be layered in future phases.
    """

    def __init__(self, relationship_matrix: StakeholderRelationshipMatrix):
        self.relationship_matrix = relationship_matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate_consequences(self, action: Action) -> ConsequenceSet:  # noqa: D401
        """Calculate consequences for a single action (layer 1 only)."""
        relationship_deltas = self.relationship_matrix.update_relationships_from_action(action)
        return ConsequenceSet(
            relationship_changes=relationship_deltas,
            resource_changes={},
            scenario_changes={},
            behavior_changes={},
            meta_changes={},
        )
