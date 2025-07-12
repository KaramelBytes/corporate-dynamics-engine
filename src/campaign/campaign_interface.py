"""
Campaign Mode Interface for Corporate Dynamics Simulator.

This module provides the Rich-based CLI interface for campaign mode,
including campaign selection, progress tracking, and scenario management.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich import box

from src.campaign.campaign_manager import CampaignManager, CampaignPhase
from src.scenarios.base_scenario import BaseScenario
from src.scenarios.scenario_factory import ScenarioFactory
from src.state_management.game_state import GameState


class CampaignInterface:
    """Handles the Rich-based CLI interface for campaign mode."""
    
    def __init__(self, console: Console, game_state: GameState, scenario_factory: ScenarioFactory):
        """Initialize the campaign interface.
        
        Args:
            console: Rich console instance for output
            game_state: Current game state
            scenario_factory: Factory for creating scenario instances
        """
        self.console = console
        self.game_state = game_state
        self.scenario_factory = scenario_factory
        self.campaign_manager = CampaignManager(game_state, scenario_factory)
        self.logger = logging.getLogger(__name__)
    
    def show_campaign_menu(self) -> str:
        """Display the main campaign menu and return the selected option.
        
        Returns:
            str: Selected menu option
        """
        self.console.rule("[bold blue]Campaign Mode[/]")
        
        options = [
            "New Campaign",
            "Continue Campaign" if self.campaign_manager.game_state.campaign_state else "",
            "View Campaign Progress",
            "Return to Main Menu"
        ]
        
        # Filter out empty options (like Continue when no campaign exists)
        options = [opt for opt in options if opt]
        
        for i, option in enumerate(options, 1):
            self.console.print(f"[cyan]{i}.[/] {option}")
        
        choice = Prompt.ask("\nSelect an option", 
                          choices=[str(i) for i in range(1, len(options) + 1)])
        return options[int(choice) - 1]
    
    def start_new_campaign(self) -> bool:
        """Start a new campaign with the given name.
        
        Returns:
            bool: True if a new campaign was started, False otherwise
        """
        self.console.print("\n[bold]New Campaign[/]")
        campaign_name = Prompt.ask("Enter a name for your campaign", 
                                 default="Corporate Journey")
        
        if Confirm.ask(f"Start new campaign '{campaign_name}'?"):
            self.campaign_manager.start_new_campaign(campaign_name)
            self.console.print(f"\n[green]✓[/] Started new campaign: {campaign_name}")
            return True
        return False
    
    def show_campaign_progress(self) -> None:
        """Display detailed campaign progress with Rich formatting."""
        if not hasattr(self.game_state, 'campaign_state') or not self.game_state.campaign_state:
            self.console.print("[yellow]No active campaign found. Starting a new one...[/]")
            self.start_new_campaign()
            return
            
        campaign = self.game_state.campaign_state
        
        # Campaign header
        self.console.rule(f"[bold blue]Campaign: {campaign.campaign_id}[/]")
        
        # Progress summary
        progress_table = Table(show_header=False, box=box.ROUNDED)
        progress_table.add_column("Metric", style="cyan", width=25)
        progress_table.add_column("Value")
        
        # Debug logging for scenario registration
        from src.scenarios.scenario_factory import _registry
        all_scenario_classes = list(_registry.all())
        self.logger.debug(f"Total registered scenarios: {len(all_scenario_classes)}")
        for scenario_cls in all_scenario_classes:
            self.logger.debug(f"Registered: {scenario_cls.metadata.scenario_id} - {scenario_cls.metadata.title}")
        
        # Calculate campaign completion percentage using all registered scenarios
        total_scenarios = len(all_scenario_classes)
        completed_count = len(campaign.completed_scenarios)
        completion_pct = (completed_count / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        progress_table.add_row("Campaign Phase", f"[bold]{campaign.current_phase.upper()}[/]")
        progress_table.add_row("Scenarios Completed", f"{completed_count}/{total_scenarios}")
        progress_table.add_row("Campaign Performance", 
                             f"[green]{campaign.total_performance_score:.0%}" if campaign.total_performance_score >= 0.7 
                             else f"[yellow]{campaign.total_performance_score:.0%}" 
                             if campaign.total_performance_score >= 0.4 
                             else f"[red]{campaign.total_performance_score:.0%}")
        
        self.console.print(progress_table)
        
        # Progress bar
        with Progress(
            TextColumn("Progress"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task("", total=100)
            progress.update(task, completed=completion_pct)
        
        # Completed scenarios
        if campaign.completed_scenarios:
            self.console.print("\n[bold]Completed Scenarios:[/]")
            for comp in campaign.completed_scenarios:
                score_color = "green" if comp.performance_score >= 0.7 else "yellow" if comp.performance_score >= 0.4 else "red"
                self.console.print(
                    f"- [bold]{comp.scenario_id}[/] "
                    f"({comp.completed_at.strftime('%Y-%m-%d')}): "
                    f"[{score_color}]{comp.performance_score:.0%}[/]"
                )
        
        # Available scenarios
        available = self.campaign_manager.get_available_scenarios()
        all_scenarios = {s.metadata.scenario_id: s for s in all_scenario_classes}
        
        # Show available scenarios
        if available:
            self.console.print("\n[bold]Available Scenarios:[/]")
            for scenario in available:
                self.console.print(f"- [green]{scenario.metadata.title}[/]")
        
        # Show locked scenarios with unlock progress
        unlocked = {s.metadata.scenario_id for s in available}
        completed = {c.scenario_id for c in campaign.completed_scenarios}
        locked = [
            s for s in all_scenarios.values()
            if s.metadata.scenario_id not in unlocked 
            and s.metadata.scenario_id not in completed
        ]
        
        if locked:
            self.console.print("\n[bold]Locked Scenarios:[/]")
            unlock_progress = self.campaign_manager.calculate_unlock_readiness()
            for scenario in locked:
                progress = unlock_progress.get(scenario.metadata.scenario_id, 0)
                if progress > 0:
                    self.console.print(
                        f"- [dim]{scenario.metadata.title} "
                        f"[yellow]({progress:.0%} to unlock)[/]"
                    )
                else:
                    self.console.print(f"- [dim]{scenario.metadata.title} [red](locked)[/]")
    
    def select_campaign_scenario(self) -> Optional[Type[BaseScenario]]:
        """Display available scenarios and let the user select one.
        
        Returns:
            Optional[Type[BaseScenario]]: Selected scenario class or None if going back
        """
        self.logger.debug("Getting available scenarios from campaign manager")
        available = self.campaign_manager.get_available_scenarios()
        self.logger.debug("Found %d available scenarios", len(available))
        
        if not available:
            self.console.print("[yellow]No scenarios available. Complete other scenarios to unlock more.[/]")
            return None
        
        self.console.rule("[bold blue]Available Scenarios[/]")
        
        # Group by phase
        scenarios_by_phase: Dict[CampaignPhase, List[Type[BaseScenario]]] = {}
        for scenario in available:
            phase = scenario.campaign_phase
            if phase not in scenarios_by_phase:
                scenarios_by_phase[phase] = []
            scenarios_by_phase[phase].append(scenario)
        
        self.logger.debug("Grouped scenarios by phase: %s", [p.value for p in scenarios_by_phase.keys()])
        
        # Display all available scenarios first, regardless of phase
        options: List[Tuple[str, Type[BaseScenario]]] = []
        self.console.print("\n[bold]AVAILABLE SCENARIOS[/]")
        
        # Process all scenarios, regardless of phase
        all_scenarios = []
        for phase in scenarios_by_phase.values():
            all_scenarios.extend(phase)
            
        # Add scenarios to options
        for i, scenario in enumerate(all_scenarios, 1):
            self.logger.debug("Adding scenario %d: %s (ID: %s)", i, scenario.metadata.title, scenario.metadata.scenario_id)
            options.append((str(i), scenario))
        
        self.logger.debug("Total options added to menu: %d", len(options))
        self.logger.debug("Menu options: %s", [opt[1].metadata.title for opt in options])
        
        # Add back option
        options.append((str(len(options) + 1), "back"))
        
        # Display menu
        for i, option in enumerate(options, 1):
            if isinstance(option[1], str):
                self.console.print(f"\n[cyan]{i}.[/] Back to Campaign Menu")
            else:
                self.console.print(
                    f"\n[cyan]{i}.[/] [bold]{option[1].metadata.title}[/]\n"
                    f"   [dim]{option[1].metadata.description}"
                )
        
        # Get user choice
        choice = Prompt.ask(
            "\nSelect a scenario",
            choices=[str(i) for i in range(1, len(options) + 2)]
        )
        
        if int(choice) > len(options):
            return None
            
        return options[int(choice) - 1][1]
    
    def complete_scenario(self, scenario: Type[BaseScenario], action_results: Dict[str, Any]) -> None:
        """Handle scenario completion and update campaign state.
        
        Args:
            scenario: The completed scenario class
            action_results: Results from the completed scenario
        """
        # Update campaign state and get updated results with performance score
        updated_results = self.campaign_manager.complete_scenario(
            scenario.metadata.scenario_id, 
            action_results
        )
        
        # Show completion message with the performance score from the updated results
        self.console.print("\n[green]✓[/] Scenario completed!")
        self.console.print(f"Performance: [bold]{updated_results.get('performance_score', 0):.0%}")
        
        # Use the updated results for relationship changes
        action_results.update(updated_results)
        
        # Show relationship changes
        if 'relationship_changes' in action_results:
            self.console.print("\n[bold]Relationship Changes:[/]")
            relationship_changes = action_results['relationship_changes']
            
            for stakeholder, changes in relationship_changes.items():
                if isinstance(changes, dict):
                    # Handle nested attribute changes
                    self.console.print(f"[bold]{stakeholder}:[/]")
                    for attr, delta in changes.items():
                        if isinstance(delta, (int, float)):
                            color = "green" if delta >= 0 else "red"
                            sign = "+" if delta >= 0 else ""
                            self.console.print(f"  - {attr}: [{color}]{sign}{delta:+.2f}[/]")
                elif isinstance(changes, (int, float)):
                    # Handle flat changes (legacy format)
                    color = "green" if changes >= 0 else "red"
                    sign = "+" if changes >= 0 else ""
                    self.console.print(f"- {stakeholder}: [{color}]{sign}{changes:+.2f}[/]")
        
        # Check for newly unlocked scenarios
        available_before = set(s.metadata.scenario_id for s in self.campaign_manager.get_available_scenarios())
        self.campaign_manager.update_available_scenarios()
        available_after = set(s.metadata.scenario_id for s in self.campaign_manager.get_available_scenarios())
        
        newly_unlocked = available_after - available_before
        if newly_unlocked:
            self.console.print("\n[bold green]New Scenarios Unlocked![/]")
            for scenario_id in newly_unlocked:
                scenario = self.scenario_factory.get_scenario(scenario_id)
                if scenario:
                    self.console.print(f"- [green]{scenario.metadata.title}[/]")
        
        # Show updated campaign progress
        self.console.print("\n[bold]Campaign Progress:[/]")
        self.show_campaign_progress()
        
        # Auto-save campaign progress
        if hasattr(self.game_state, 'save_path') and self.game_state.save_path:
            self.game_state.save(self.game_state.save_path)
            self.console.print(f"\n[dim]Campaign progress saved to {self.game_state.save_path}[/]")
