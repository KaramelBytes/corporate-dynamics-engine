"""Vendor Lock-in Dilemma – Scenario #7

Implements *The Vendor Lock-in Dilemma* specification from
`docs/scenario_07_vendor_lockin_dilemma.md`.
"""
from __future__ import annotations

from typing import List

from src.state_management.data_models import Action

from .base_scenario import (
    BaseScenario,
    CampaignPhase,
    ScenarioMetadata,
    ScenarioType,
    RelationshipUnlock,
    CampaignUnlock
)
from .scenario_factory import register_scenario


@register_scenario
class VendorLockinDilemma(BaseScenario):
    """Scenario addressing vendor management and cost optimization."""
    # Late-phase scenario
    campaign_phase = CampaignPhase.LATE
    
    metadata: ScenarioMetadata = ScenarioMetadata(
        scenario_id="vendor_lockin_dilemma",
        title="Vendor Lock-in Dilemma",
        description=(
            "TechCorp Solutions demands a 40% price increase for their enterprise software. "
            "Navigate the tension between cost optimization and operational stability while "
            "balancing stakeholder priorities across the organization."
        ),
        scenario_type=ScenarioType.VENDOR_MANAGEMENT,
        difficulty=0.7,
        themes=["vendor_management", "cost_optimization", "operational_stability"],
        stakeholders=[
            "ceo",
            "it_director",
            "sales_director",
            "it_team",
            "cfo",
            "admin_team",
            "marketing_team",
        ],
    )

    def __init__(self) -> None:
        # First initialize the parent class
        super().__init__()
        
        # Initialize unlock conditions with dynamic thresholds
        self.unlock_conditions = [
            # Option 1: Complete 5+ scenarios with good performance
            CampaignUnlock(
                required_completed_scenarios=[],  # Any 5 scenarios
                min_campaign_performance=0.0,  # Will be set dynamically
                required_campaign_phase=CampaignPhase.LATE,
                max_failures_allowed=2
            ),
            # Option 2: High CFO relationship and completed technical debt scenario
            CampaignUnlock(
                required_completed_scenarios=["technical_debt_reckoning"],
                min_campaign_performance=0.0,  # Will be set dynamically
                required_campaign_phase=CampaignPhase.LATE,
                required_relationships={"cfo": 0.65}  # Slightly reduced from 0.7
            ),
            # Option 3: High IT Director relationship and completed return to office scenario
            CampaignUnlock(
                required_completed_scenarios=["return_to_office_mandate"],
                min_campaign_performance=0.0,  # Will be set dynamically
                required_campaign_phase=CampaignPhase.LATE,
                required_relationships={"it_director": 0.65}  # Reduced from 0.7
            )
        ]
        self._actions = self._build_actions()
        
    def evaluate_unlock_conditions(self, game_state: 'GameState') -> bool:
        """Evaluate unlock conditions with dynamic thresholds."""
        # Get the current campaign state
        campaign_state = getattr(game_state, 'campaign_state', None)
        if not campaign_state:
            return False
            
        # Calculate dynamic threshold based on player progress
        completed_count = len(campaign_state.completed_scenarios)
        base_threshold = 0.5  # Base threshold of 50% for late-game scenarios
        
        # More gradual progression for late-game scenarios
        progress_adjustment = min(0.25, completed_count * 0.015)  # Up to 25% adjustment
        dynamic_threshold = max(0.4, base_threshold - progress_adjustment)  # 40-50% range
        
        # Update all campaign unlock conditions with dynamic threshold
        for condition in self.unlock_conditions:
            if isinstance(condition, CampaignUnlock):
                condition.min_campaign_performance = dynamic_threshold
        
        # Check all unlock conditions
        return any(condition.is_met(game_state, campaign_state) 
                 for condition in self.unlock_conditions)

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_actions(self) -> List[Action]:
        return [
            # Choice 1 – Migration Advocate
            Action(
                id="migration_advocate",
                description=(
                    "The 40% increase is unsustainable. We're migrating to NimbleStart - "
                    "it's 60% cheaper and has modern features. I'll dedicate a team to "
                    "handle data migration and rebuild integrations properly."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "cfo",
                    "admin_team",
                    "marketing_team",
                ],
                resource_cost={"time": 0.8, "budget": 0.6, "team_capacity": 0.7},
                difficulty=0.8,
                action_type="migration",
                relationship_deltas={
                    "ceo": 0.13,
                    "it_director": -0.03,
                    "sales_director": -0.10,
                    "it_team": 0.15,
                    "cfo": 0.35,
                    "admin_team": -0.43,
                    "marketing_team": 0.47,
                },
            ),
            # Choice 2 – Negotiation Strategist
            Action(
                id="negotiation_strategist",
                description=(
                    "We're not accepting 40% increases. I'm negotiating with TechCorp "
                    "using competitive alternatives as leverage. If they won't budge, "
                    "we have migration options ready."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "cfo",
                    "admin_team",
                    "marketing_team",
                ],
                resource_cost={"time": 0.4, "political_capital": 0.5},
                difficulty=0.7,
                action_type="negotiation",
                relationship_deltas={
                    "ceo": 0.33,
                    "it_director": 0.27,
                    "sales_director": 0.22,
                    "it_team": 0.07,
                    "cfo": 0.27,
                    "admin_team": 0.38,
                    "marketing_team": -0.03,
                },
            ),
            # Choice 3 – Stability Maintainer
            Action(
                id="stability_maintainer",
                description=(
                    "Migration risks outweigh cost savings during our growth phase. We'll "
                    "pay the increase but demand additional features and support. Better "
                    "to focus on revenue growth than operational disruption."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "cfo",
                    "admin_team",
                    "marketing_team",
                ],
                resource_cost={"budget": 0.4, "vendor_leverage": 0.6},
                difficulty=0.5,
                action_type="vendor_management",
                relationship_deltas={
                    "ceo": 0.10,
                    "it_director": 0.38,
                    "sales_director": 0.27,
                    "it_team": 0.13,
                    "cfo": -0.35,
                    "admin_team": 0.47,
                    "marketing_team": -0.33,
                },
            ),
            # Choice 4 – Hybrid Transition Planner
            Action(
                id="hybrid_transition_planner",
                description=(
                    "We'll do a phased migration over 18 months - start with new projects "
                    "on NimbleStart while gradually moving existing data. Reduces risk "
                    "while achieving cost savings."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "cfo",
                    "admin_team",
                    "marketing_team",
                ],
                resource_cost={"time": 0.9, "budget": 0.4, "team_capacity": 0.6},
                difficulty=0.85,
                action_type="transformation",
                relationship_deltas={
                    "ceo": 0.27,
                    "it_director": 0.33,
                    "sales_director": 0.18,
                    "it_team": 0.27,
                    "cfo": 0.13,
                    "admin_team": -0.06,
                    "marketing_team": 0.33,
                },
            ),
            # Choice 5 – Vendor Independence Builder
            Action(
                id="vendor_independence_builder",
                description=(
                    "This is our chance to reduce vendor dependence. We'll build internal "
                    "tools using OpenSource Plus - higher upfront investment but complete "
                    "control and minimal ongoing costs."
                ),
                directly_affects=[
                    "ceo",
                    "it_director",
                    "sales_director",
                    "it_team",
                    "cfo",
                    "admin_team",
                    "marketing_team",
                ],
                resource_cost={"time": 0.7, "budget": 0.8, "team_capacity": 0.9},
                difficulty=0.9,
                action_type="innovation",
                relationship_deltas={
                    "ceo": 0.03,
                    "it_director": -0.05,
                    "sales_director": -0.15,
                    "it_team": 0.35,
                    "cfo": 0.10,
                    "admin_team": -0.50,
                    "marketing_team": -0.03,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # BaseScenario implementation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._actions = self._build_actions()

    def get_available_actions(self) -> List[Action]:
        return self._actions

    def __repr__(self) -> str:  # noqa: D401
        return f"<VendorLockinDilemma id={self.scenario_id} choices={len(self._actions)}>"
