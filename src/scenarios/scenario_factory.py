"""Scenario Factory System – Phase 1 infrastructure.

This module introduces a *lightweight* scenario discovery, registration, and
selection mechanism. It does **not** implement concrete scenarios beyond the
existing POC; it merely prepares the groundwork for the upcoming 8-12 scenario
implementation drop (see *Phase 1* in ``WINDSURF_DEVELOPMENT_GUIDE.md``).

Design goals
------------
1. **Automatic discovery.** Subclasses of ``BaseScenario`` inside the
   ``src.scenarios`` package are auto-registered at import time, providing a
   *zero-boilerplate* authoring experience.
2. **Unlock condition filtering.** The factory evaluates ``UnlockCondition``
   objects against the live ``GameState`` to decide which scenarios are
   currently eligible.
3. **Progression orchestration.** A pluggable *selection strategy* picks one of
   the eligible scenarios. The default strategy is *weighted randomness* based
   on ``difficulty``. More sophisticated strategies can be injected later.
4. **Extensibility.** The factory is intentionally minimal; upcoming features
   (dependency graphs, difficulty curves, dynamic composition) will extend this
   foundation without breaking API contracts.

All functions include comprehensive type hints and Google-style docstrings, in
line with workspace coding standards.
"""
from __future__ import annotations

import importlib
import importlib
import logging
import pkgutil
import random
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Type, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.state_management.game_state import GameState
    from .base_scenario import BaseScenario, UnlockCondition

from pydantic import BaseModel, Field

from src.state_management.game_state import GameState

from .base_scenario import BaseScenario, UnlockCondition

__all__ = [
    "ScenarioFactory",
]


class _ScenarioRegistry(BaseModel):
    """Internal singleton registry holding scenario classes."""

    scenarios: Dict[str, Type[BaseScenario]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    # NOTE: Because the model is frozen, we use object.__setattr__ for mutation.
    def add(self, scenario_cls: Type[BaseScenario]) -> None:
        """Add a scenario class to the registry (idempotent)."""
        scenario_id = scenario_cls.metadata.scenario_id  # type: ignore[attr-defined]
        if scenario_id in self.scenarios:
            return
        object.__setattr__(self, "scenarios", {**self.scenarios, scenario_id: scenario_cls})

    def get(self, scenario_id: str) -> Optional[Type[BaseScenario]]:  # noqa: D401
        """Retrieve a scenario class by identifier."""
        return self.scenarios.get(scenario_id)

    def all(self) -> Sequence[Type[BaseScenario]]:  # noqa: D401
        """Return all registered scenario classes."""
        return list(self.scenarios.values())


# Global registry instance – instantiated eagerly.
_registry = _ScenarioRegistry()


def register_scenario(scenario_cls: Type[BaseScenario]) -> Type[BaseScenario]:
    """Class decorator to register ``BaseScenario`` subclasses automatically.

    Example::
        @register_scenario
        class MyScenario(BaseScenario):
            ...
    """

    if not issubclass(scenario_cls, BaseScenario):
        raise TypeError("Only subclasses of BaseScenario can be registered")
    _registry.add(scenario_cls)
    return scenario_cls


class ScenarioFactory:
    """Factory orchestrating scenario discovery and selection."""

    def __init__(self, package: str = "src.scenarios") -> None:
        """Create a factory instance and perform package discovery.

        Args:
            package: Python import path containing scenario modules.
        """
        self._package = package
        self._discover_package(package)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def available_scenarios(
        self, 
        game_state: 'GameState', 
        campaign_mode: bool = True
    ) -> List['BaseScenario']:
        """Instantiate and return scenarios unlocked for *game_state*.
        
        Args:
            game_state: Current game state
            campaign_mode: If True, use campaign unlock logic. If False, only use
                         basic unlock conditions.
                
        Returns:
            List of instantiated scenario objects that are currently available
        """
        candidates: List['BaseScenario'] = []
        
        # If in campaign mode and campaign manager exists, use it
        if campaign_mode and hasattr(game_state, 'campaign_state'):
            from src.campaign.campaign_manager import CampaignManager
            campaign_manager = CampaignManager(game_state, self)
            return [scenario_cls() for scenario_cls in _registry.all() 
                   if campaign_manager.evaluate_unlock_conditions(scenario_cls)]
        
        # Fall back to basic unlock conditions
        for scenario_cls in _registry.all():
            if self._is_unlocked(scenario_cls, game_state):
                candidates.append(scenario_cls())
                
        return candidates

    def select_next_scenario(
        self, 
        game_state: 'GameState',
        campaign_mode: bool = True
    ) -> Optional['BaseScenario']:
        """Select the next scenario to play.
        
        Args:
            game_state: Current game state
            campaign_mode: If True, use campaign recommendation logic.
                         If False, use simple weighted random.
                         
        Returns:
            A scenario instance or None if none available
        """
        # In campaign mode, use the campaign manager's recommendation
        if campaign_mode and hasattr(game_state, 'campaign_state'):
            from src.campaign.campaign_manager import CampaignManager
            campaign_manager = CampaignManager(game_state, self)
            return campaign_manager.recommend_next_scenario()
            
        # Fall back to weighted random selection based on difficulty
        unlocked = self.available_scenarios(game_state, campaign_mode=False)
        if not unlocked:
            return None
            
        # Lower difficulty = higher weight (easier scenarios more likely)
        weights = [1.0 - s.metadata.difficulty for s in unlocked]
        return random.choices(unlocked, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _discover_package(package_path: str) -> None:
        """Import all modules inside *package_path* to trigger registration.

        Re-importing is safe because ``importlib.import_module`` caches modules.
        """
        logger = logging.getLogger(__name__)
        logger.debug("Starting discovery of package: %s", package_path)
        try:
            pkg = importlib.import_module(package_path)
            logger.debug("Successfully imported package: %s", package_path)
            if not hasattr(pkg, "__path__"):
                logger.debug("Package %s has no __path__, skipping", package_path)
                return

            logger.debug("Package path: %s", pkg.__path__)
            modules = list(pkgutil.iter_modules(pkg.__path__))
            logger.debug("Found %d modules in package %s", len(modules), package_path)

            for i, module_info in enumerate(modules, 1):
                module_name = f"{package_path}.{module_info.name}"
                logger.debug("Importing module %d/%d: %s", i, len(modules), module_info.name)
                ScenarioFactory._import_module(module_name)
        except ImportError as e:
            logger.warning("Failed to import package %s: %s", package_path, e)
            import traceback
            traceback.print_exc()

    @staticmethod
    def _import_module(module_name: str) -> None:
        """Import a single module and register any scenario classes found."""
        logger = logging.getLogger(__name__)
        logger.debug("Attempting to import scenario module: %s", module_name)
        try:
            module = importlib.import_module(module_name)
            logger.debug("Successfully imported %s", module_name)

            # Find all scenario classes in the module
            for name, obj in vars(module).items():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseScenario)
                    and obj is not BaseScenario
                ):
                    register_scenario(obj)
                    logger.debug("Found scenario class: %s in %s", name, module_name)
        except ImportError as e:
            logger.warning("Failed to import scenario module %s: %s", module_name, e)

    # ------------------------------------------------------------------
    # Unlock evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _is_unlocked(
        scenario_cls: Type['BaseScenario'], 
        game_state: 'GameState'
    ) -> bool:
        """Check if a scenario is unlocked based on basic conditions.
        
        This is a fallback method used when not in campaign mode or when
        campaign state is not available.
        """
        # Check if the scenario has any unlock conditions
        if not hasattr(scenario_cls, 'unlock_conditions') or not scenario_cls.unlock_conditions:
            return True
            
        # Check each condition
        for condition in scenario_cls.unlock_conditions:
            # For basic relationship conditions
            if hasattr(condition, 'stakeholder_id') and hasattr(condition, 'min_relationship'):
                rel = game_state.player_reputation.get(condition.stakeholder_id, 0.0)
                if rel < condition.min_relationship:
                    return False
                    
        return True
