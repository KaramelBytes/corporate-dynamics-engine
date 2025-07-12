"""Scenarios package containing scenario definitions and engines."""

import importlib
import logging
import pkgutil
import sys
from typing import List, Type

from .base_scenario import BaseScenario, ScenarioMetadata, ScenarioType, UnlockCondition  # noqa: F401
from .scenario_factory import ScenarioFactory, _registry  # noqa: F401

# Set up logger
logger = logging.getLogger(__name__)

# List of all scenario modules to import
SCENARIO_MODULES = [
    'ai_hype_cycle_crisis',
    'return_to_office_mandate',
    'security_vs_speed_showdown',
    'technical_debt_reckoning',
    'burnout_crisis',
    'legacy_system_guardian',
    'vendor_lockin_dilemma',
    'cross_functional_conflict',
    'coffee_machine_crisis',  # Ensure coffee machine crisis is included
]

def import_scenario_modules():
    """Explicitly import all scenario modules to ensure they're registered."""
    logger.debug("Starting explicit import of scenario modules")
    for module_name in SCENARIO_MODULES:
        full_module_name = f"src.scenarios.{module_name}" if not module_name.startswith('src.') else module_name
        try:
            logger.debug("Importing scenario module: %s", full_module_name)
            module = importlib.import_module(full_module_name)
            logger.debug("Successfully imported %s", full_module_name)
            
            # Log all scenario classes in the module
            for name, obj in vars(module).items():
                if (isinstance(obj, type) and 
                    issubclass(obj, BaseScenario) and 
                    obj is not BaseScenario):
                    logger.debug("Found scenario class: %s in %s", name, full_module_name)
        except Exception as e:
            logger.error("Failed to import %s: %s", full_module_name, e, exc_info=True)

# Import all scenario modules when the package is loaded
import_scenario_modules()

# Log registration status
logger.debug("Number of registered scenarios: %d", len(_registry.all()))
for scenario in _registry.all():
    logger.debug("Registered scenario: %s - %s", 
                scenario.metadata.scenario_id, 
                scenario.metadata.title)

def _import_scenarios() -> List[Type[BaseScenario]]:
    """Dynamically import all scenario modules to register them."""
    imported = []
    for module_name in SCENARIO_MODULES:
        try:
            module = importlib.import_module(f'.{module_name}', __name__)
            imported.append(module)
        except ImportError as e:
            print(f"Warning: Failed to import scenario module {module_name}: {e}")
    return imported

# Import all scenarios when the package is loaded
_import_scenarios()
