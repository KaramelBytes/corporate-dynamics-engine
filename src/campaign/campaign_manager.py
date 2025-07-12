"""Campaign management for the Corporate Dynamics Simulator.

This module handles campaign state, scenario progression, and performance tracking
across multiple scenarios in a campaign.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type, cast, Any
from collections import defaultdict
import random

from pydantic import BaseModel, Field, field_validator, computed_field

from src.scenarios.base_scenario import BaseScenario, UnlockCondition, RelationshipUnlock, CampaignUnlock
from src.scenarios.scenario_factory import ScenarioFactory
from src.state_management.game_state import GameState

logger = logging.getLogger(__name__)


class CampaignPhase(str, Enum):
    """Represents different phases of a campaign."""

    EARLY = "early"  # First 1-3 scenarios
    MID = "mid"  # Scenarios 4-6
    LATE = "late"  # Scenarios 7-9
    EXPERT = "expert"  # All scenarios completed


@dataclass
class ScenarioCompletion:
    """Tracks completion data for a single scenario within a campaign."""

    scenario_id: str
    completed_at: datetime
    performance_score: float  # 0.0-1.0 based on relationship gains
    choices_made: List[str]  # Action IDs chosen
    stakeholder_impact: Dict[str, float]  # Final relationship deltas


class CampaignState(BaseModel):
    """Tracks the state of an ongoing or completed campaign."""

    campaign_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    completed_scenarios: List[ScenarioCompletion] = Field(default_factory=list)
    current_phase: CampaignPhase = CampaignPhase.EARLY
    total_performance_score: float = 0.0
    unlock_progress: Dict[str, bool] = Field(default_factory=dict)  # scenario_id -> unlocked
    performance_score: float = 0.0  # Current overall campaign performance (0.0-1.0)

    @field_validator("performance_score")
    @classmethod
    def validate_performance_score(cls, v: float) -> float:
        """Ensure performance score is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Performance score must be between 0.0 and 1.0")
        return v


class CampaignManager:
    """Manages campaign progression and scenario unlocking logic."""

    def __init__(self, game_state: GameState, scenario_factory: ScenarioFactory):
        """Initialize the campaign manager with game state and scenario factory.
        
        Args:
            game_state: The current game state
            scenario_factory: Factory for creating scenario instances
        """
        self.game_state = game_state
        self.scenario_factory = scenario_factory
        
        # Initialize campaign state if not present
        if not hasattr(self.game_state, 'campaign_state') or self.game_state.campaign_state is None:
            self.start_new_campaign("default_campaign")

    def start_new_campaign(self, campaign_id: str) -> CampaignState:
        """Start a new campaign with the given ID.
        
        Args:
            campaign_id: Unique identifier for the campaign
            
        Returns:
            The newly created campaign state
        """
        logger.info("Starting new campaign: %s", campaign_id)
        self.game_state.campaign_state = CampaignState(
            campaign_id=campaign_id,
            start_time=datetime.utcnow(),
            current_phase=CampaignPhase.EARLY,
        )
        return self.game_state.campaign_state

    def complete_scenario(self, scenario_id: str, action_results: Dict[str, Any]) -> Dict[str, Any]:
        """Mark a scenario as completed and update campaign state.
        
        Args:
            scenario_id: ID of the completed scenario
            action_results: Dictionary containing action results including:
                - choices_made: List of action IDs chosen
                - relationship_changes: Dict of stakeholder ID to relationship delta
                
        Returns:
            Updated action results including the performance score
        """
        if not self.game_state.campaign_state:
            logger.warning("No active campaign, starting a new one")
            self.start_new_campaign("default_campaign")

        # Make a copy of action_results to avoid modifying the original
        results = dict(action_results)
        
        # Extract relationship changes from action results
        relationship_changes = results.get("relationship_changes", {})
        
        # If no explicit relationship changes, check if the action results themselves are relationship changes
        if not relationship_changes and all(isinstance(k, str) and isinstance(v, (int, float, float))
                                         for k, v in results.items() if k != 'choices_made'):
            relationship_changes = {k: v for k, v in results.items() if k != 'choices_made'}
            results["relationship_changes"] = relationship_changes
            
        # Create a clean results dict with just the relationship changes
        score_input = {"relationship_changes": relationship_changes}
        
        # Calculate performance score based on relationship changes
        performance_score = self.calculate_performance_score(score_input)
        
        # Add performance score to the results
        results["performance_score"] = performance_score

        # Record completion
        completion = ScenarioCompletion(
            scenario_id=scenario_id,
            completed_at=datetime.utcnow(),
            performance_score=performance_score,
            choices_made=results.get("choices_made", []),
            stakeholder_impact=relationship_changes,
        )

        self.game_state.campaign_state.completed_scenarios.append(completion)
        self.game_state.campaign_state.total_performance_score = (
            sum(s.performance_score for s in self.game_state.campaign_state.completed_scenarios)
            / len(self.game_state.campaign_state.completed_scenarios)
        )

        # Update unlock progress and performance score
        self.game_state.campaign_state.unlock_progress[scenario_id] = True
        self.game_state.campaign_state.performance_score = performance_score
        
        # Check for phase progression
        self._update_campaign_phase()
        
        logger.info(
            "Completed scenario %s with score %.2f. Campaign progress: %s",
            scenario_id,
            performance_score,
            self.game_state.campaign_state.current_phase.value,
        )
        
        return results

    def update_available_scenarios(self) -> None:
        """Update the list of available scenarios based on current campaign state.
        
        This is a no-op method maintained for backward compatibility.
        Scenarios are now evaluated on-demand in get_available_scenarios().
        """
        logger.debug("update_available_scenarios() called - scenarios are now evaluated on-demand")
        pass
        
    def get_available_scenarios(self) -> List[Type[BaseScenario]]:
        """Get all scenarios that are available in the current campaign.
        
        Returns:
            List of scenario classes that are available to play
        """
        if not self.game_state.campaign_state:
            return []
            
        all_scenarios = self.scenario_factory.available_scenarios(self.game_state)
        return [s for s in all_scenarios if self.evaluate_unlock_conditions(s)]
        
    def evaluate_unlock_conditions(self, scenario: Type[BaseScenario]) -> bool:
        """Check if a scenario meets all unlock requirements.
        
        Args:
            scenario: The scenario class to check
            
        Returns:
            bool: True if all unlock conditions are met, or if no conditions are specified
        """
        # Always available in tutorial mode or if no campaign state
        if scenario.is_tutorial or not self.game_state.campaign_state:
            return True
            
        # Check all unlock conditions
        campaign_state = self.game_state.campaign_state
        
        # Check if already completed (unless repeatable)
        completed_ids = {s.scenario_id for s in campaign_state.completed_scenarios}
        if scenario.metadata.scenario_id in completed_ids and not getattr(scenario, 'repeatable', False):
            return False
            
        # If no unlock conditions, check if we're in the right phase
        if not hasattr(scenario, 'unlock_conditions') or not scenario.unlock_conditions:
            # Only check phase if the scenario has a specific phase requirement
            if hasattr(scenario, 'campaign_phase') and scenario.campaign_phase:
                return self._get_scenario_phase(scenario.metadata.scenario_id) == scenario.campaign_phase
            return True
            
        # Check all unlock conditions - all must be met
        for condition in scenario.unlock_conditions:
            if not condition.is_met(self.game_state, campaign_state):
                return False
                
        return True
        
    def recommend_next_scenario(self) -> Optional[Type[BaseScenario]]:
        """Intelligently recommend the next scenario to play.
        
        Prioritizes:
        1. Tutorial scenarios if any are available
        2. Scenarios that will unlock new paths
        3. Scenarios that improve weak areas
        4. Random selection from available scenarios
        
        Returns:
            Recommended scenario class or None if none available
        """
        available = self.get_available_scenarios()
        if not available:
            return None
            
        # 1. Check for tutorial scenarios first
        tutorials = [s for s in available if s.is_tutorial]
        if tutorials:
            return random.choice(tutorials)
            
        # 2. Find scenarios that unlock the most new scenarios
        unlock_scores = self._calculate_unlock_scores(available)
        if unlock_scores:
            max_score = max(unlock_scores.values())
            if max_score > 0:
                best_candidates = [s for s, score in unlock_scores.items() 
                                 if score == max_score]
                return random.choice(best_candidates)
        
        # 3. Fall back to random selection from available
        return random.choice(available)
        
    def _calculate_unlock_scores(self, available_scenarios: List[Type[BaseScenario]]) -> Dict[Type[BaseScenario], int]:
        """Calculate how many new scenarios each available scenario would unlock."""
        if not self.game_state.campaign_state:
            return {}
            
        # Get all scenarios that could be unlocked
        all_scenarios = self.scenario_factory.available_scenarios(self.game_state)
        locked = [s for s in all_scenarios if not self.evaluate_unlock_conditions(s)]
        
        # For each available scenario, see how many locked scenarios it would unlock
        scores = {s: 0 for s in available_scenarios}
        
        for scenario in available_scenarios:
            # Create a hypothetical completion
            temp_state = self.game_state.model_copy(deep=True)
            temp_campaign = self.game_state.campaign_state.model_copy(deep=True)
            
            # Add completion with average performance
            completion = ScenarioCompletion(
                scenario_id=scenario.metadata.scenario_id,
                completed_at=datetime.utcnow(),
                performance_score=0.7,  # Assume average performance
                choices_made=[],
                stakeholder_impact={}
            )
            temp_campaign.completed_scenarios.append(completion)
            temp_state.campaign_state = temp_campaign
            
            # Count how many new scenarios would be unlocked
            for locked_scenario in locked:
                temp_manager = CampaignManager(temp_state, self.scenario_factory)
                if temp_manager.evaluate_unlock_conditions(locked_scenario):
                    scores[scenario] += 1
                    
        return scores
        
    def calculate_unlock_readiness(self) -> Dict[str, float]:
        """Calculate unlock progress for all scenarios (0.0-1.0).
        
        Returns:
            Dict mapping scenario_id to unlock progress (0.0 = locked, 1.0 = unlocked)
        """
        result = {}
        all_scenarios = self.scenario_factory.available_scenarios(self.game_state)
        
        for scenario in all_scenarios:
            if self.evaluate_unlock_conditions(scenario):
                result[scenario.metadata.scenario_id] = 1.0
                continue
                
            # Calculate partial progress for locked scenarios
            progress = 0.0
            total_conditions = len(scenario.unlock_conditions) or 1
            met_conditions = 0
            
            for condition in scenario.unlock_conditions:
                if isinstance(condition, RelationshipUnlock):
                    # Progress based on relationship level (0-1)
                    current = self.game_state.player_reputation.get(condition.stakeholder_id, 0.0)
                    progress += min(1.0, (current + 1.0) / 2.0)  # Convert -1..1 to 0..1
                    met_conditions += 1
                elif isinstance(condition, CampaignUnlock):
                    # Progress based on campaign state
                    campaign = self.game_state.campaign_state
                    if not campaign:
                        continue
                        
                    # Check completed scenarios
                    if condition.required_completed_scenarios:
                        completed = {s.scenario_id for s in campaign.completed_scenarios}
                        met = sum(1 for sid in condition.required_completed_scenarios 
                                if sid in completed)
                        progress += met / len(condition.required_completed_scenarios)
                        met_conditions += 1
                        
                    # Check campaign performance
                    if condition.min_campaign_performance > 0:
                        progress += min(1.0, campaign.total_performance_score / condition.min_campaign_performance)
                        met_conditions += 1
            
            # Average progress across all conditions
            if met_conditions > 0:
                result[scenario.metadata.scenario_id] = min(1.0, progress / met_conditions)
            else:
                result[scenario.metadata.scenario_id] = 0.0
                
        return result
    # FUTURE: Performance calculations may be too punishing for early scenarios - consider adjusting scoring algorithms for better progression balance
    # Class-level configuration for scoring
    SCORING_WEIGHTS = {
        'relationship': 0.4,   # Overall relationship improvement
        'balance': 0.25,       # Balance across stakeholders
        'risk': 0.2,           # Risk management
        'influence': 0.15      # Focus on high-influence stakeholders
    }
    
    # Risk thresholds (in standard deviations from mean relationship change)
    LOW_RISK_THRESHOLD = 0.3   # Below this is considered low risk
    HIGH_RISK_THRESHOLD = 0.7  # Above this is considered high risk

    def _calculate_relationship_score(self, relationship_changes: Dict[str, float]) -> float:
        """Calculate the base relationship improvement score (0-1).
        
        Args:
            relationship_changes: Dictionary of stakeholder ID to relationship delta (-1 to 1)
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not relationship_changes:
            return 0.5  # Neutral score if no changes
            
        # Convert all changes to 0-1 range where 0 is worst, 1 is best
        normalized_changes = [(delta + 1) / 2 for delta in relationship_changes.values()]
        return sum(normalized_changes) / len(normalized_changes)
    
    def _calculate_balance_score(self, relationship_changes: Dict[str, float]) -> float:
        """Calculate how balanced the relationship changes are across stakeholders.
        
        Args:
            relationship_changes: Dictionary of stakeholder ID to relationship delta
            
        Returns:
            Score from 0.0 (very unbalanced) to 1.0 (perfectly balanced)
        """
        if len(relationship_changes) < 2:
            return 1.0  # Perfect balance with 0 or 1 stakeholder
            
        # Calculate Gini coefficient (measure of inequality)
        values = sorted([(delta + 1) / 2 for delta in relationship_changes.values()])  # Normalize to 0-1
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 1.0  # Perfect balance if no changes
            
        # Gini coefficient calculation
        gini = 1 - (2 * sum((i + 1) * v for i, v in enumerate(values)) / (n * sum(values)) - (n + 1) / n)
        return 1.0 - gini  # Convert to balance score (1.0 = perfect balance)
    
    def _calculate_risk_score(self, relationship_changes: Dict[str, float]) -> float:
        """Calculate risk based on variance in relationship changes.
        
        Args:
            relationship_changes: Dictionary of stakeholder ID to relationship delta
            
        Returns:
            Risk score from 0.0 (low risk) to 1.0 (high risk)
        """
        if len(relationship_changes) < 2:
            # Return neutral risk based on dynamic threshold with 0 or 1 stakeholder
            return self._calculate_dynamic_unlock_threshold()
            
        changes = list(relationship_changes.values())
        mean_change = sum(changes) / len(changes)
        variance = sum((x - mean_change) ** 2 for x in changes) / len(changes)
        std_dev = variance ** 0.5
        
        # Convert standard deviation to risk score (0-1)
        # Lower variance = lower risk, higher variance = higher risk
        risk_score = min(1.0, std_dev * 2)  # Scale to 0-1 range
        
        # Invert so higher score is better (we want moderate risk)
        optimal_risk = 0.4  # Slightly below medium risk is optimal
        return 1.0 - abs(risk_score - optimal_risk) / optimal_risk
    
    def _calculate_influence_score(self, relationship_changes: Dict[str, float]) -> float:
        """Calculate score based on satisfying high-influence stakeholders.
        
        Args:
            relationship_changes: Dictionary of stakeholder ID to relationship delta
            
        Returns:
            Score from 0.0 to 1.0 based on satisfying key stakeholders
        """
        if not relationship_changes:
            return 0.5  # Neutral score if no changes
            
        # Get current relationship levels from game state
        current_relationships = self.game_state.player_reputation
        
        # Calculate influence scores (1.0 for best relationships, 0.0 for worst)
        influence_scores = []
        
        for stakeholder_id, delta in relationship_changes.items():
            current = current_relationships.get(stakeholder_id, 0.0)
            # Calculate influence (0-1) based on current relationship (higher = more influence)
            influence = (current + 1) / 2  # Convert from -1..1 to 0..1
            
            # Calculate satisfaction (0-1) based on relationship change
            satisfaction = (delta + 1) / 2  # Convert from -1..1 to 0..1
            
            # Weighted score gives more importance to high-influence stakeholders
            influence_scores.append(satisfaction * influence)
        
        # Return neutral influence score based on dynamic threshold if no scores
        return sum(influence_scores) / len(influence_scores) if influence_scores else self._calculate_dynamic_unlock_threshold()

    def _flatten_relationship_changes(self, relationship_changes: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested relationship changes to a simple stakeholder -> delta mapping.
        
        Args:
            relationship_changes: Nested dictionary of stakeholder -> attribute -> delta
            
        Returns:
            Flattened dictionary of stakeholder -> average delta
        """
        flat_changes = {}
        
        for stakeholder, changes in relationship_changes.items():
            if isinstance(changes, dict):
                # Calculate average of all numeric attribute changes
                numeric_changes = [v for v in changes.values() if isinstance(v, (int, float))]
                if numeric_changes:
                    flat_changes[stakeholder] = sum(numeric_changes) / len(numeric_changes)
            elif isinstance(changes, (int, float)):
                flat_changes[stakeholder] = changes
                
        return flat_changes
    
    def calculate_performance_score(self, action_results: Dict[str, Any]) -> float:
        """Calculate a 0.0-1.0 performance score based on relationship changes.
        
        The score uses a simplified algorithm that:
        - Starts with a 60% base score (acceptable corporate leadership baseline)
        - Adds up to 30% for positive relationship building
        - Adds up to 10% for average positive change
        - Subtracts up to 20% for severe relationship damage
        - Final score is bounded between 30% and 95%
        
        Args:
            action_results: Dictionary containing action results including:
                - relationship_changes: Dict[str, Dict[str, float]] - Stakeholder -> attribute -> delta
                
        Returns:
            Performance score between 0.3 and 0.95
        """
        # Log the incoming action_results for debugging
        logger.debug("Calculating performance score with action_results: %s", 
                   {k: v for k, v in action_results.items() if k != 'relationship_changes'})
        
        # Extract relationship changes from action_results
        if not isinstance(action_results, dict):
            relationship_changes = {}
            logger.debug("action_results is not a dictionary, using empty relationship_changes")
        else:
            # First try to get relationship_changes directly
            relationship_changes = action_results.get("relationship_changes", {})
            logger.debug("Extracted relationship_changes: %s", relationship_changes)
            
            # If no explicit relationship_changes, check if action_results itself looks like relationship changes
            if not relationship_changes and all(isinstance(k, str) and isinstance(v, (int, float)) 
                                             for k, v in action_results.items() if k != 'choices_made'):
                relationship_changes = {k: v for k, v in action_results.items() if k != 'choices_made'}
                logger.debug("Using action_results as relationship_changes: %s", relationship_changes)
        
        # If we still don't have relationship changes, try to find them in the game state
        if not relationship_changes and hasattr(self.game_state, 'relationship_deltas'):
            relationship_changes = self.game_state.relationship_deltas
            logger.debug("Using relationship_deltas from game state: %s", relationship_changes)
        
        # Flatten nested relationship changes if needed
        flat_changes = self._flatten_relationship_changes(relationship_changes)
        
        if not flat_changes:
            logger.debug("No relationship changes found, using base score")
            return 0.6  # Return base score if no changes
            
        logger.debug("Flattened relationship changes: %s", flat_changes)
        
        try:
            # Start with base score of 60%
            base_score = 0.6
            
            # Calculate relationship building bonus (up to 30%)
            positive_changes = [delta for delta in flat_changes.values() if delta > 0]
            relationship_bonus = 0.0
            if positive_changes:
                # Calculate average positive change and scale to 0-0.3 range
                avg_positive = sum(positive_changes) / len(positive_changes)
                # Scale factor based on dynamic threshold to cap relationship bonus
                scale_factor = self._calculate_dynamic_unlock_threshold()
                relationship_bonus = min(0.3, avg_positive * scale_factor)  # Scale based on dynamic threshold
            
            # Calculate average change bonus (up to 10%)
            all_changes = list(flat_changes.values())
            avg_change = sum(all_changes) / len(all_changes)
            avg_change_bonus = max(0, min(0.1, avg_change * 0.2))  # Scale factor to cap at 0.1
            
            # Calculate penalty for severe relationship damage (up to -20%)
            severe_damage = sum(1 for delta in all_changes if delta <= -0.3)
            damage_penalty = min(0.2, severe_damage * 0.1)  # 10% per severe damage, max 20%
            
            # Calculate raw score
            raw_score = base_score + relationship_bonus + avg_change_bonus - damage_penalty
            
            # Apply bounds (30% to 95%)
            final_score = max(0.3, min(0.95, raw_score))
            
            logger.debug(
                "Score components - Base: %.2f, Relationship Bonus: +%.2f, "
                "Avg Change Bonus: +%.2f, Damage Penalty: -%.2f, Raw: %.2f, Final: %.2f",
                base_score, relationship_bonus, avg_change_bonus, 
                damage_penalty, raw_score, final_score
            )
            
            logger.info("Calculated performance score: %.2f", final_score)
            return final_score
            
        except Exception as e:
            logger.error("Error calculating performance score: %s", str(e), exc_info=True)
            return 0.6  # Return base score on error

    def _calculate_dynamic_unlock_threshold(self, scenario: Optional[Type[BaseScenario]] = None) -> float:
        """Calculate a dynamic unlock threshold based on player progress and scenario difficulty.
        
        The threshold starts at 40% and can be reduced by up to 10% based on player progress.
        It also adjusts based on scenario difficulty if available.
        
        Args:
            scenario: Optional scenario to consider for difficulty adjustment
            
        Returns:
            A threshold between 0.3 and 0.6 (30% to 60%)
        """
        # Base threshold is 40%
        base_threshold = 0.4
        
        # Calculate progress adjustment (up to -10% as player progresses)
        progress_adjustment = 0.0
        campaign_state = getattr(self.game_state, 'campaign_state', None)
        
        if campaign_state and campaign_state.completed_scenarios:
            # Reduce threshold by up to 10% as player completes more scenarios
            completed_count = len(campaign_state.completed_scenarios)
            progress_adjustment = -0.1 * min(1.0, completed_count / 10.0)  # Full adjustment after 10 scenarios
        
        # Calculate difficulty adjustment if scenario is provided
        difficulty_adjustment = 0.0
        if scenario and hasattr(scenario, 'metadata') and hasattr(scenario.metadata, 'difficulty'):
            # Scale adjustment based on scenario difficulty (0.0 to 1.0)
            # Harder scenarios get a lower threshold (easier to unlock)
            difficulty_adjustment = -0.1 * scenario.metadata.difficulty
        
        # Calculate final threshold (bounded between 30% and 60%)
        final_threshold = base_threshold + progress_adjustment + difficulty_adjustment
        return max(0.3, min(0.6, final_threshold))

    def _update_campaign_phase(self) -> None:
        """Update the campaign phase based on completed scenarios and performance.
        
        Progresses through phases as the player completes more scenarios.
        """
        if not self.game_state.campaign_state:
            return
            
        completed = len(self.game_state.campaign_state.completed_scenarios)
        current_phase = self.game_state.campaign_state.current_phase
        
        # Phase progression thresholds (number of completed scenarios)
        phase_thresholds = {
            CampaignPhase.EARLY: 3,
            CampaignPhase.MID: 6,
            CampaignPhase.LATE: 9,
            CampaignPhase.EXPERT: 12
        }
        
        # Determine the highest phase the player qualifies for
        new_phase = CampaignPhase.EARLY
        for phase, threshold in phase_thresholds.items():
            if completed >= threshold:
                new_phase = phase
                
        # Only update if we've progressed to a new phase
        if new_phase != current_phase:
            logger.info(
                "Campaign phase updated from %s to %s (completed: %d)",
                current_phase.value,
                new_phase.value,
                completed
            )
            self.game_state.campaign_state.current_phase = new_phase
            
    def _get_scenario_phase(self, scenario_id: str) -> CampaignPhase:
        """Determine which campaign phase a scenario belongs to based on its ID.
        
        This maps specific scenario IDs to their appropriate campaign phases.
        """
        # Tutorial scenarios
        tutorial_scenarios = {
            "ai_hype_cycle_crisis",
            "coffee_machine_crisis"
        }
        
        # Early phase scenarios
        early_phase = {
            "return_to_office_mandate"
        }
        
        # Mid phase scenarios
        mid_phase = {
            "security_vs_speed_showdown",
            "technical_debt_reckoning",
            "burnout_crisis"
        }
        
        # Late phase scenarios
        late_phase = {
            "legacy_system_guardian",
            "vendor_lockin_dilemma",
            "cross_functional_conflict"
        }
        
        # Expert phase scenarios (unlocked after completing all others)
        expert_phase = {
            "executive_showdown",
            "corporate_takeover"
        }
        
        if scenario_id in tutorial_scenarios:
            return CampaignPhase.TUTORIAL
        elif scenario_id in early_phase:
            return CampaignPhase.EARLY
        elif scenario_id in mid_phase:
            return CampaignPhase.MID
        elif scenario_id in late_phase:
            return CampaignPhase.LATE
        elif scenario_id in expert_phase:
            return CampaignPhase.EXPERT
            
        # Default phase based on ID pattern (for backward compatibility)
        try:
            num_part = "".join(c for c in scenario_id if c.isdigit())
            if num_part:
                scenario_num = int(num_part[-1])
                if scenario_num <= 3:
                    return CampaignPhase.EARLY
                elif scenario_num <= 6:
                    return CampaignPhase.MID
                elif scenario_num <= 9:
                    return CampaignPhase.LATE
        except (ValueError, IndexError):
            pass
            
        return CampaignPhase.EARLY  # Default fallback
