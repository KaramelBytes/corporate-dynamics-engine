"""Generic scenario engine to orchestrate scenario lifecycle."""
from typing import Protocol, List, Dict, Any


class Scenario(Protocol):  # pragma: no cover
    """Protocol representing scenario interface."""

    scenario_id: str
    title: str
    description: str

    def get_available_actions(self) -> List[Any]:  # noqa: ANN401
        ...

    def get_scenario_context(self) -> Dict[str, Any]:
        ...


class ScenarioEngine:
    """Runs scenario lifecycle and communicates with `GameState`."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    def start(self) -> None:
        """Kick off scenario and return initial context."""
        # Placeholder implementation
        print(f"Starting scenario: {self.scenario.title}\n")
        print(self.scenario.description)

    # FUTURE: Implement turn-based engine for complex scenario chains
