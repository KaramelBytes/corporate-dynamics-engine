"""
Core game state management with enterprise-grade persistence and validation.
"""
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from datetime import datetime
import json
import logging

from .data_models import ValidationResult

if TYPE_CHECKING:
    from ..campaign.campaign_manager import CampaignState

logger = logging.getLogger(__name__)


class GameState:
    """
    Central game state management.
    Handles persistence, validation, and coordination with complex components.
    """

    def __init__(self, player_name: str = "Player"):
        self.player_name: str = player_name
        self.current_scenario_id: Optional[str] = None
        self.player_reputation: Dict[str, float] = {}
        self.game_metadata: Dict[str, Any] = {
            "start_time": datetime.utcnow(),
            "actions_taken": 0,
            "scenarios_completed": 0,
        }
        self.campaign_state: Optional[CampaignState] = None

        # Complex components - will be initialized later
        self.stakeholder_matrix = None
        self.consequence_calculator = None
        self.state_auditor = None

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
    def _to_serializable(self) -> Dict[str, Any]:
        """Convert internal state to something JSON serializable."""
        serialized = {
            "player_name": self.player_name,
            "current_scenario_id": self.current_scenario_id,
            "player_reputation": self.player_reputation,
            "game_metadata": {
                **self.game_metadata,
                "start_time": self.game_metadata["start_time"].isoformat(),
            },
        }
        
        # Add campaign state if it exists
        if self.campaign_state:
            # Convert CampaignState to dict and handle datetime serialization
            campaign_dict = self.campaign_state.model_dump()
            campaign_dict["start_time"] = campaign_dict["start_time"].isoformat()
            for completion in campaign_dict.get("completed_scenarios", []):
                completion["completed_at"] = completion["completed_at"].isoformat()
            serialized["campaign_state"] = campaign_dict
            
        return serialized

    @classmethod
    def _from_serializable(cls, data: Dict[str, Any]) -> "GameState":
        instance = cls()
        instance.player_name = data.get("player_name", "Player")
        instance.current_scenario_id = data.get("current_scenario_id")
        instance.player_reputation = data.get("player_reputation", {})
        
        # Handle game metadata
        gm = data.get("game_metadata", {})
        gm["start_time"] = datetime.fromisoformat(gm.get("start_time")) if gm.get("start_time") else datetime.utcnow()
        instance.game_metadata = gm
        
        # Handle campaign state if it exists
        if "campaign_state" in data:
            from ..campaign.campaign_manager import CampaignState, ScenarioCompletion
            
            campaign_data = data["campaign_state"]
            
            # Convert string timestamps back to datetime objects
            if "start_time" in campaign_data and isinstance(campaign_data["start_time"], str):
                campaign_data["start_time"] = datetime.fromisoformat(campaign_data["start_time"])
                
            # Handle completed scenarios
            if "completed_scenarios" in campaign_data:
                for i, completion in enumerate(campaign_data["completed_scenarios"]):
                    if isinstance(completion.get("completed_at"), str):
                        campaign_data["completed_scenarios"][i]["completed_at"] = datetime.fromisoformat(
                            completion["completed_at"]
                        )
            
            instance.campaign_state = CampaignState(**campaign_data)
            
        return instance

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def save_to_file(self, filename: str) -> bool:
        """Save game state to JSON file"""
        try:
            with open(filename, "w", encoding="utf-8") as fp:
                json.dump(self._to_serializable(), fp, indent=2)
            logger.info("Game state saved to %s", filename)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to save game state: %s", exc)
            return False

    @classmethod
    def load_from_file(cls, filename: str) -> Optional["GameState"]:
        """Load game state from JSON file"""
        try:
            with open(filename, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return cls._from_serializable(data)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to load game state: %s", exc)
            return None

    def validate_state_consistency(self) -> ValidationResult:
        """Validate current state for consistency and correctness (simple rules)."""
        errors: List[str] = []
        warnings: List[str] = []

        # Basic type checks
        if not isinstance(self.player_reputation, dict):
            errors.append("player_reputation must be a dictionary")
        if not isinstance(self.game_metadata, dict):
            errors.append("game_metadata must be a dictionary")

        # Logical numeric checks
        actions_taken = self.game_metadata.get("actions_taken", 0)
        scenarios_completed = self.game_metadata.get("scenarios_completed", 0)
        if not isinstance(actions_taken, int) or actions_taken < 0:
            errors.append("actions_taken must be a non-negative integer")
        if not isinstance(scenarios_completed, int) or scenarios_completed < 0:
            errors.append("scenarios_completed must be a non-negative integer")

        # Reputation value validation
        for stakeholder, reputation in self.player_reputation.items():
            if not isinstance(reputation, (int, float)):
                errors.append(f"Reputation for {stakeholder} must be numeric")
            elif not (0.0 <= reputation <= 1.0):
                warnings.append(
                    f"Reputation for {stakeholder} ({reputation}) outside range [0.0, 1.0]"
                )

        return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings)

    def get_stakeholder_relationship(self, stakeholder_id: str) -> Dict[str, float]:
        """Get current relationship data for a stakeholder"""
        if self.stakeholder_matrix is None:
            raise RuntimeError("Stakeholder matrix not initialized")
        return self.stakeholder_matrix.get_relationship(stakeholder_id)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def increment_action_count(self) -> None:
        """Increment the action counter in metadata."""
        self.game_metadata["actions_taken"] = self.game_metadata.get("actions_taken", 0) + 1

    def set_current_scenario(self, scenario_id: str) -> None:
        """Set the current scenario identifier after validation."""
        if not isinstance(scenario_id, str):
            raise ValueError("scenario_id must be a string")
        self.current_scenario_id = scenario_id

    # FUTURE: Additional state-management helpers will be added in later phases
