#!/usr/bin/env python
"""
Corporate Dynamics Simulator - Enterprise-grade corporate adventure game

This is the main entry point for the Corporate Dynamics Simulator, showcasing:
- Sophisticated AI integration with dynamic content generation
- Complex stakeholder relationship modeling
- Production-ready architecture patterns
- Enterprise error handling and state management
"""

# Standard library imports
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Third-party imports
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich import box

# Local imports

from src.scenarios.base_scenario import BaseScenario, ScenarioType, Action
from src.scenarios.scenario_factory import ScenarioFactory
from src.state_management.game_state import GameState
from src.state_management.stakeholder_matrix import StakeholderRelationshipMatrix
from src.ai_integration.corporate_game_ai_integration import CorporateGameAIIntegration
from src.ai_integration.data_models import AIConfig
from src.ai_integration.service_orchestrator import AIServiceOrchestrator
from src.utils.logging_config import setup_logging, get_logging_mode_from_env
from src.campaign.campaign_interface import CampaignInterface
from src.scenarios.coffee_machine_crisis import CoffeeMachineCrisis
from src.scenarios.ai_hype_cycle_crisis import AIHypeCycleCrisis
from src.scenarios.return_to_office_mandate import ReturnToOfficeMandate
from src.scenarios.security_vs_speed_showdown import SecurityVsSpeedShowdown
from src.scenarios.technical_debt_reckoning import TechnicalDebtReckoning
from src.scenarios.burnout_crisis import BurnoutCrisis
from src.scenarios.legacy_system_guardian import LegacySystemGuardian
from src.scenarios.vendor_lockin_dilemma import VendorLockinDilemma
from src.scenarios.cross_functional_conflict import CrossFunctionalConflict
from src.utils.env_loader import load_environment_variables

# Configure logging
setup_logging(get_logging_mode_from_env())
logger = logging.getLogger(__name__)

# Constants
VERSION = "1.0.0"
APP_TITLE = "Corporate Dynamics Simulator"
AUTHOR = "Technology Leadership"
RELEASE_DATE = "2025-07-02"

# Define save directory
SAVE_DIRECTORY = Path.home() / ".corporate_simulator" / "saves"
SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)


class CorporateSimulator:
    """Main entry point for the Corporate Dynamics Simulator.
    
    This class orchestrates the entire simulation experience, leveraging:
    - Dynamic AI-generated content for scenarios and dialogues
    - Sophisticated stakeholder relationship modeling
    - Enterprise-grade error handling
    - Professional UI with Rich
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None) -> None:
        """Initialize the simulator with all required components.
        
        Args:
            api_keys: Dictionary containing API keys for AI providers
        """
        self.console = Console()
        self.game_state = GameState()
        self.relationship_matrix = StakeholderRelationshipMatrix()
        self.ai_integration = CorporateGameAIIntegration()
        self.scenario_factory = ScenarioFactory()
        self.current_scenario: Optional[BaseScenario] = None
        
        # Apply API keys directly to the AI integration if provided
        if api_keys:
            # Add detailed debug logging for API key flow
            logging.debug(f"API keys provided to simulator: {', '.join([k for k, v in api_keys.items() if v])}")
            
            # Log presence of each key (not the actual keys for security)
            if api_keys.get("openai_api_key"):
                logging.info("OpenAI API key provided to simulator")
                
            if api_keys.get("anthropic_api_key"):
                logging.info("Anthropic API key provided to simulator")
                
            if api_keys.get("gemini_api_key"):
                logging.info("Gemini API key provided to simulator")
                
            openai_key = api_keys.get("openai_api_key")
            anthropic_key = api_keys.get("anthropic_api_key")
            gemini_key = api_keys.get("gemini_api_key")
            
            if openai_key:
                self.ai_integration.orchestrator.config.openai_api_key = openai_key
                logger.info("OpenAI API key provided to simulator")
            if anthropic_key:
                self.ai_integration.orchestrator.config.anthropic_api_key = anthropic_key
                logger.info("Anthropic API key provided to simulator")
            if gemini_key:
                self.ai_integration.orchestrator.config.gemini_api_key = gemini_key
                logger.info("Gemini API key provided to simulator")
            
            # Force re-initialization of providers with updated keys
            logging.info("Re-initializing AI providers with updated API keys")
            self.ai_integration.orchestrator.providers = self.ai_integration.orchestrator._initialize_providers()
            
            # Check which providers are now available
            available_providers = list(self.ai_integration.orchestrator.providers.keys())
            logging.info(f"Available AI providers after initialization: {available_providers}")
        

        
        # Current state
        self.current_scenario = None
        self.current_scenario_data = None
        self.save_path = None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    
    async def run(self) -> None:
        """Main entry point for the Corporate Dynamics Simulator."""
        try:
            # Display welcome screen
            self._display_welcome()
            
            # Game state flag - continue showing main menu until user decides to exit
            continue_game = True
            
            while continue_game:
                # Initialize or load game based on user selection
                continue_game = await self._initialize_game()
                
                # Run the main game loop
                if continue_game and not (hasattr(self.game_state, 'campaign_state') and self.game_state.campaign_state):
                    continue_game = await self._main_loop()
                        
        except KeyboardInterrupt:
            self.console.print("\n\n[red]Game interrupted by user.[/]")
        except Exception as e:
            self.console.print(f"\n[red]An error occurred: {e}[/]")
            logging.exception("Unhandled exception in main game loop")
        finally:
            self.console.print("\nThank you for playing Corporate Dynamics Simulator!")
            
            # Display AI metrics at the end of the session
            try:
                self._display_ai_metrics()
            except Exception as e:
                logging.error(f"Error displaying AI metrics: {e}")   # ------------------------------------------------------------------
    
    def _display_welcome(self) -> None:
        """Display professional welcome screen with branding."""
        self.console.clear()
        
        # Create title banner
        banner = Panel(
            f"[bold cyan]🏢 Corporate Dynamics Simulator v{VERSION}[/bold cyan]\n\n"
            "[green]✓[/] Clean initialization (verbose logging suppressed)\n"
            "[green]✓[/] AI-driven enterprise scenarios\n"
            "[green]✓[/] Sophisticated stakeholder relationship modeling\n"
            "[green]✓[/] Real-world business dynamics\n"
            "[green]✓[/] Production-ready architecture patterns",
            title="Enterprise Leadership Portfolio",
            subtitle="Professional Demo Mode",
            border_style="blue",
            expand=False
        )
        
        self.console.print(banner)
        self.console.print("\n")
        
    def _display_main_menu(self) -> str:
        """Display main menu and return selected option."""
        self.console.rule("[bold blue]Corporate Dynamics Simulator[/]")
        
        options = [
            "Campaign Mode",
            "View AI Metrics", 
            "Help",
            "Exit"
        ]
        
        for i, option in enumerate(options, 1):
            self.console.print(f"[cyan]{i}.[/] {option}")
        
        choice = Prompt.ask("\nSelect an option", choices=[str(i) for i in range(1, len(options) + 1)])
        return options[int(choice) - 1]
        
    # ------------------------------------------------------------------
    # Game initialization methods
    # ------------------------------------------------------------------
    
    async def _main_loop(self) -> bool:
        """Main game loop with scenario selection and execution.
        
        Returns:
            bool: True to continue showing the main menu, False to exit the game.
        """
        while True:
            # Show scenario selection
            scenario = self._select_scenario()
            if not scenario:
                # User chose to go back to main menu
                return True
                
            # Run the selected scenario
            await self._run_scenario(scenario)
            
            # Auto-save after each scenario
            if hasattr(self.game_state, 'save_path') and self.game_state.save_path:
                self.game_state.save(self.game_state.save_path)
                self.console.print(f"\n[dim]Game saved to {self.game_state.save_path}[/]")
            
            # Ask if user wants to play another scenario
            if not Confirm.ask("\nWould you like to play another scenario?"):
                return True
    
    async def _initialize_game(self) -> bool:
        """Initialize or load a game based on user selection."""
        while True:
            choice = self._display_main_menu()
            
            if choice == "Campaign Mode":
                campaign_interface = CampaignInterface(self.console, self.game_state, self.scenario_factory)
                return await self._handle_campaign_mode(campaign_interface)
                


                    
            elif choice == "View AI Metrics":
                self._display_ai_metrics()
                
            elif choice == "Help":
                self._display_help()
                
            elif choice == "Exit":
                self.console.print("Thank you for using Corporate Dynamics Simulator!")
                return False
    
    async def _handle_campaign_mode(self, campaign_interface: CampaignInterface) -> bool:
        """Handle the campaign mode flow.
        
        Args:
            campaign_interface: Initialized campaign interface
            
        Returns:
            bool: True to continue showing the main menu, False to exit
        """
        while True:
            choice = campaign_interface.show_campaign_menu()
            
            if choice == "New Campaign":
                if campaign_interface.start_new_campaign():
                    # After starting a new campaign, show progress and let user select a scenario
                    await self._handle_campaign_loop(campaign_interface)
                
            elif choice == "Continue Campaign":
                # Handle the campaign loop
                await self._handle_campaign_loop(campaign_interface)
                    
            elif choice == "View Campaign Progress":
                campaign_interface.show_campaign_progress()
                
            elif choice == "Return to Main Menu":
                return True
    
    async def _handle_campaign_loop(self, campaign_interface: CampaignInterface) -> None:
        """Handle the campaign loop with scenario selection and post-scenario flow.
        
        Args:
            campaign_interface: Initialized campaign interface
        """
        # Store the campaign interface for use in _run_scenario
        self.campaign_interface = campaign_interface
        
        while True:
            # Show campaign progress
            campaign_interface.show_campaign_progress()
            
            # Let user select a scenario
            scenario = campaign_interface.select_campaign_scenario()
            if not scenario:
                break  # User chose to go back
                
            try:
                # Run the selected scenario and get action results
                action_results = await self._run_scenario(scenario)
                
                # After scenario completion, show campaign progress
                campaign_interface.show_campaign_progress()
                
                # Auto-save after scenario completion
                if hasattr(self.game_state, 'save_path') and self.game_state.save_path:
                    self.game_state.save(self.game_state.save_path)
                    
                # Ask if user wants to continue to next scenario or return to campaign menu
                self.console.print("\n[bold]What would you like to do next?[/]")
                choices = ["1. Select another scenario", "2. Return to Campaign Menu"]
                for choice in choices:
                    self.console.print(choice)
                    
                next_action = Prompt.ask(
                    "\nChoose an option",
                    choices=["1", "2"],
                    default="1"
                )
                
                if next_action == "2":
                    break
                    
            except Exception as e:
                self.console.print(f"\n[red]Error during scenario execution: {e}[/]")
                logger.exception("Error during scenario execution")
                # Give user option to continue or exit
                if not Confirm.ask("\nWould you like to continue with the campaign?"):
                    break

    def _initialize_new_game(self) -> None:
        """Initialize a new game state."""
        self.game_state = GameState()
        self.relationship_matrix = StakeholderRelationshipMatrix()
        
        # Player name for personalization
        player_name = Prompt.ask("\nEnter your name", default="Executive")
        self.game_state.player_name = player_name
        
        # Set default save path
        self.game_state.save_path = Path(SAVE_DIRECTORY) / f"{player_name.lower().replace(' ', '_')}.json"
        
        # Relationships are already initialized in the StakeholderRelationshipMatrix constructor
        
        self.console.print(f"\nWelcome, {player_name}! Your corporate journey begins.")
        
        # Save the initial game state
        self.game_state.save(self.game_state.save_path)
    
    # ------------------------------------------------------------------
    # Scenario selection and execution methods 
    # ------------------------------------------------------------------
    
    async def _main_loop(self) -> bool:
        """Main game loop with scenario selection and execution.
        
        Returns:
            bool: True to continue showing the main menu, False to exit the game.
        """
        while True:
            # Display scenario selection
            selected_scenario = await self._select_scenario()
            
            if not selected_scenario:
                # User chose to return to main menu
                return True
                
            # Run the selected scenario with AI-driven content
            await self._run_scenario(selected_scenario)
            
            # After scenario completion, offer save
            if Confirm.ask("Would you like to save your progress?"):
                save_path = Path(SAVE_DIRECTORY) / f"{self.game_state.player_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self._save_game(save_path)
                
            # Continue or exit
            if not Confirm.ask("Continue to next scenario?"):
                # User chose not to continue with scenarios
                # Ask if they want to return to main menu or exit game
                if Confirm.ask("Return to main menu?"):
                    return True
                else:
                    return False
    
    async def _select_scenario(self) -> Optional[BaseScenario]:
        """Display available scenarios and allow user to select one."""
        self.console.rule("[bold blue]Available Scenarios[/]")
        
        # Get all available scenarios
        available_scenarios = self.scenario_factory.available_scenarios(self.game_state)
        
        if not available_scenarios:
            self.console.print("[bold red]No scenarios available![/]") 
            return None
            
        # Create scenario selection table
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("#", style="cyan")
        table.add_column("Scenario", style="white")
        table.add_column("Type", style="green")
        table.add_column("Difficulty", style="yellow")
        table.add_column("Themes", style="magenta")
        
        # Add scenarios to table
        for idx, scenario in enumerate(available_scenarios, start=1):
            difficulty_display = "★" * int(round(scenario.metadata.difficulty * 5))
            themes = ", ".join(scenario.metadata.themes[:2])  # Limit to first 2 themes
            
            table.add_row(
                str(idx),
                scenario.metadata.title,
                scenario.metadata.scenario_type.name.capitalize(),
                difficulty_display,
                themes
            )
            
        # Add exit option
        table.add_row("0", "Return to Main Menu", "", "", "")
        
        self.console.print(table)
        self.console.print("")
        
        # Get user selection
        choice = Prompt.ask(
            "\nSelect a scenario", 
            choices=[str(i) for i in range(0, len(available_scenarios) + 1)],
            default="1"
        )
        
        choice_idx = int(choice)
        if choice_idx == 0:
            return None
            
        selected_scenario = available_scenarios[choice_idx - 1]
        return selected_scenario
        
    async def _run_scenario(self, scenario: BaseScenario) -> Dict[str, Any]:
        """Run a specific scenario with AI-generated content.
        
        Returns:
            Dict containing action results including relationship changes and choices made
        """
        self.current_scenario = scenario
        self.console.rule(f"[bold blue]{scenario.metadata.title}[/]")
        
        # Create rich scenario context for AI generation
        scenario_context = self._create_enhanced_scenario_context(scenario)
        self.current_scenario_data = scenario_context
        
        # Generate scenario content using AI integration
        ai_generated_content = await self._generate_scenario_content(scenario_context)
        
        # Display AI-generated scenario description
        if ai_generated_content:
            self._display_scenario_description(ai_generated_content)
        else:
            # Fallback to static description if AI generation fails
            self._display_scenario_description({"description": scenario.metadata.description})
        
        # Get available actions from the scenario
        actions = scenario.get_available_actions()
        
        # Execute the main scenario interaction loop and get the action results
        action_results = await self._scenario_interaction_loop(actions, scenario_context)
        
        # Initialize action results if not provided by the interaction loop
        if action_results is None:
            action_results = {
                "choices_made": [],
                "relationship_changes": {},
                "stakeholder_impact": {}
            }
        
        # Update game state after scenario completion
        if "scenarios_completed" not in self.game_state.game_metadata:
            self.game_state.game_metadata["scenarios_completed"] = []
        elif not isinstance(self.game_state.game_metadata["scenarios_completed"], list):
            # Convert from int to list if incorrectly stored
            count = self.game_state.game_metadata["scenarios_completed"]
            self.game_state.game_metadata["scenarios_completed"] = []
            logging.info(f"Converted scenarios_completed from int ({count}) to list")
            
        if scenario.metadata.scenario_id not in self.game_state.game_metadata["scenarios_completed"]:
            self.game_state.game_metadata["scenarios_completed"].append(scenario.metadata.scenario_id)
        
        # If we're in campaign mode, update the campaign state
        if hasattr(self, 'campaign_interface') and self.campaign_interface:
            self.campaign_interface.complete_scenario(scenario, action_results)
            
        self.console.print(f"\n[bold green]Scenario '{scenario.metadata.title}' completed successfully![/]")
        
        return action_results
    
    def _create_enhanced_scenario_context(self, scenario: BaseScenario) -> Dict[str, Any]:
        """Create an enhanced context for AI generation with all required fields."""
        # Get base context from scenario
        context = scenario.get_scenario_context()
        
        # Ensure all required fields are present
        if "context_description" not in context:
            context["context_description"] = scenario.metadata.description
            
        # Add stakeholder descriptions if not present
        if "stakeholder_descriptions" not in context:
            context["stakeholder_descriptions"] = {}
            for stakeholder in scenario.metadata.stakeholders:
                context["stakeholder_descriptions"][stakeholder] = f"{stakeholder.replace('_', ' ').title()} with a role in this scenario"
        
        # Add stakeholder profiles for each stakeholder (required by Pydantic validation)
        if "stakeholder_profiles" not in context:
            context["stakeholder_profiles"] = {}
            for stakeholder in scenario.metadata.stakeholders:
                name = stakeholder.replace('_', ' ').title()
                context["stakeholder_profiles"][stakeholder] = {
                    "name": name,
                    "role": name,
                    "background": context["stakeholder_descriptions"].get(stakeholder, ""),
                    "priorities": ["primary concern", "secondary concern"],
                    "trust": 0.6,
                    "respect": 0.6,
                    "influence": 0.6
                }
        
        # Add player information
        context["player"] = {
            "name": self.game_state.player_name,
            "role": "Technology Leader",
            "completed_scenarios": self.game_state.game_metadata.get("scenarios_completed", 0)
        }
        
        # Add corporate profile if needed
        if "corporate_profile" not in context:
            context["corporate_profile"] = {
                "company_name": "TechDynamics Inc.",
                "industry": "Enterprise Software",
                "size": "Mid-sized (500-1000 employees)",
                "culture": "Innovation-focused with a strong emphasis on work-life balance"
            }
        
        # Add previous_events (required by AI prompt template)
        if "previous_events" not in context:
            # For first scenario, use an empty list or generic corporate history
            completed_scenarios = self.game_state.game_metadata.get("scenarios_completed", [])
            if not completed_scenarios:
                context["previous_events"] = "This is the first scenario in your role as Technology Leader."
            else:
                # If there are completed scenarios, we could provide a summary
                context["previous_events"] = f"You have completed {len(completed_scenarios)} previous scenarios."
        
        # Add technical_context for AI integration
        context["technical_context"] = {
            "simulator_version": VERSION,
            "generation_timestamp": datetime.now().isoformat(),
            "generation_type": "scenario"
        }
            
        return context
        
    async def _generate_scenario_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scenario content using AI integration."""
        self.console.print("\nGenerating scenario using AI integration...")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Generating scenario via AI…", total=None)
            try:
                # Key part: Use AI integration to dynamically generate scenario content
                scenario_payload = await self.ai_integration.generate_scenario(context)
                progress.remove_task(task)
                
                # Extract content from different possible response formats
                content = {}
                if hasattr(scenario_payload, "content") and isinstance(scenario_payload.content, dict):
                    # It's a ScenarioResponse with content attribute
                    content = scenario_payload.content
                elif hasattr(scenario_payload, "description"):
                    # It has a direct description attribute
                    content = {"description": scenario_payload.description}
                elif isinstance(scenario_payload, dict):
                    # It's a plain dictionary
                    content = scenario_payload
                    
                return content
                
            except Exception as e:
                logger.error(f"Error generating scenario: {e}", exc_info=True)
                progress.remove_task(task)
                self.console.print(f"[yellow]AI generation encountered an issue: {e}[/]")
                self.console.print("[yellow]Falling back to template content...[/]")
                return {}
    
    def _display_scenario_description(self, content: Dict[str, Any]) -> None:
        """Display the AI-generated scenario description."""
        description = content.get("description", "No description available.")
        context = content.get("context", "")
        
        # Format the description as markdown for rich display
        md = Markdown(f"# Scenario Brief\n\n{description}")
        
        self.console.print("\n")
        self.console.print(Panel(md, title="AI-Generated Scenario", border_style="blue", expand=False))
        self.console.print("\n")
        
        # If there's additional context, display it
        if context:
            self.console.print(Panel(context, title="Additional Context", border_style="cyan"))
            
    async def _scenario_interaction_loop(self, actions: List[Action], scenario_context: Dict[str, Any]) -> Dict[str, Any]:
        """Main interaction loop for scenario execution with AI-generated dialogue.
        
        Returns:
            Dict containing action results including relationship changes and choices made
        """

        # Initialize result tracking
        result = {
            "choices_made": [],
            "relationship_changes": {},
            "stakeholder_impact": {}
        }
        
        # Display stakeholders involved in this scenario
        self._display_stakeholders(list(scenario_context.get("stakeholder_profiles", {}).keys()))
        
        while True:
            # Display available actions with AI-generated context
            self._display_available_actions(actions)
            
            # Get user action choice
            choice = Prompt.ask(
                "\nChoose your action", 
                choices=[str(i) for i in range(1, len(actions) + 1)] + ["0"],
                default="1"
            )
            
            choice_idx = int(choice)
            if choice_idx == 0:
                # Option to exit scenario
                if Confirm.ask("Are you sure you want to exit this scenario?"):
                    self.console.print("\n[yellow]Exiting scenario...[/]")
                    break
                else:
                    continue
                    
            # Execute chosen action with AI-generated consequences
            chosen_action = actions[choice_idx - 1]
            action_result = await self._execute_action(chosen_action, scenario_context)
            
            # Track the choice and its results
            result["choices_made"].append({
                "action_id": chosen_action.id,
                "action_description": chosen_action.description,
                "result": action_result
            })
            
            # Merge relationship changes
            if action_result and "relationship_changes" in action_result:
                for stakeholder, changes in action_result["relationship_changes"].items():
                    if stakeholder not in result["relationship_changes"]:
                        result["relationship_changes"][stakeholder] = {}
                    result["relationship_changes"][stakeholder].update(changes)
            
            # Update stakeholder impact
            if action_result and "stakeholder_impact" in action_result:
                result["stakeholder_impact"].update(action_result["stakeholder_impact"])
            
            # Scenario is complete after processing one action
            break
                
        return result
                
    async def _execute_action(self, action: Action, scenario_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chosen action with AI-generated dialogue and consequences.
        
        Returns:
            Dict containing action results including relationship changes and stakeholder impact
        """
        self.console.rule(f"[bold blue]Executing: {action.description}[/]")
        
        # Prepare context for AI dialogue generation
        dialogue_context = self._prepare_dialogue_context(action, scenario_context)
        
        # Store relationship state before the action
        previous_relationships = {}
        for stakeholder_id in action.directly_affects:
            if stakeholder_id in self.relationship_matrix.direct_relationships:
                previous_relationships[stakeholder_id] = self.relationship_matrix.direct_relationships[stakeholder_id].copy()
        
        # Generate AI dialogue for affected stakeholders
        for stakeholder_id in action.directly_affects:
            if stakeholder_id not in scenario_context.get("stakeholder_profiles", {}):
                continue
                
            stakeholder_profile = scenario_context["stakeholder_profiles"][stakeholder_id]
            stakeholder_name = stakeholder_profile["name"]
            
            # Generate dialogue using AI integration
            self.console.print(f"\n[cyan]Generating response from {stakeholder_name}...[/]")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Generating dialogue via AI…", total=None)
                try:
                    # Create stakeholder-specific context by updating base context
                    stakeholder_dialogue_context = dict(dialogue_context)
                    
                    # Fill in stakeholder-specific details
                    stakeholder_dialogue_context["stakeholder_profile"] = {
                        "id": stakeholder_id,
                        "name": stakeholder_name,
                        "role": stakeholder_profile.get("role", stakeholder_name),
                        "description": stakeholder_profile.get("description", ""),
                        "personality": stakeholder_profile.get("personality", ""),
                        "current_trust": dialogue_context["relationship_status"].get(stakeholder_id, {}).get("trust", 0.0),
                    }
                    
                    # Use AI integration to dynamically generate dialogue
                    dialogue_response = await self.ai_integration.generate_dialogue(
                        stakeholder_id,
                        stakeholder_dialogue_context
                    )
                    progress.remove_task(task)
                    
                    # Extract dialogue from response
                    dialogue = ""
                    logger.debug(f"Dialogue response type: {type(dialogue_response)}, content: {dialogue_response}")
                    
                    # Handle different response structures
                    if hasattr(dialogue_response, "content"):
                        if isinstance(dialogue_response.content, dict):
                            # First check for Gemini format where content field contains the dialogue
                            if "content" in dialogue_response.content:
                                dialogue = dialogue_response.content.get("content", "")
                                logger.info(f"Found dialogue in content.content: {dialogue[:50]}...")
                            # Then check for traditional format
                            elif "dialogue" in dialogue_response.content:
                                dialogue = dialogue_response.content.get("dialogue", "")
                                logger.info(f"Found dialogue in content.dialogue: {dialogue[:50]}...")
                    elif hasattr(dialogue_response, "dialogue"):
                        dialogue = dialogue_response.dialogue
                        logger.info(f"Found dialogue in dialogue attribute: {dialogue[:50]}...")
                    elif isinstance(dialogue_response, dict):
                        # Check for Gemini format where response is a dict with content field
                        if "content" in dialogue_response:
                            dialogue = dialogue_response.get("content", "")
                            logger.info(f"Found dialogue in dict.content: {dialogue[:50]}...")
                        # Check for traditional format
                        elif "dialogue" in dialogue_response:
                            dialogue = dialogue_response.get("dialogue", "")
                            logger.info(f"Found dialogue in dict.dialogue: {dialogue[:50]}...")
                    
                    if not dialogue:
                        dialogue = f"*{stakeholder_name} acknowledges your action but has nothing specific to say.*"
                        
                    # Display the dialogue with stakeholder attribution
                    self._display_stakeholder_dialogue(stakeholder_id, dialogue)
                    
                except Exception as e:
                    logger.error(f"Error generating dialogue: {e}", exc_info=True)
                    progress.remove_task(task)
                    self.console.print(f"[yellow]AI dialogue generation encountered an issue: {e}[/]")
                    self._display_stakeholder_dialogue(
                        stakeholder_id, 
                        f"*{stakeholder_name} acknowledges your action.*"
                    )
        
        # Update stakeholder relationships based on action
        self._update_stakeholder_relationships(action)
        
        # Calculate relationship changes
        relationship_changes = {}
        stakeholder_impact = {}
        
        for stakeholder_id in action.directly_affects:
            if stakeholder_id in self.relationship_matrix.direct_relationships:
                current_relationships = self.relationship_matrix.direct_relationships[stakeholder_id]
                if stakeholder_id in previous_relationships:
                    # Calculate deltas for each relationship attribute
                    deltas = {}
                    for key, current_value in current_relationships.items():
                        old_value = previous_relationships[stakeholder_id].get(key, 0.0)
                        # Only calculate deltas for numeric values
                        if isinstance(current_value, (int, float)) and isinstance(old_value, (int, float)):
                            delta = current_value - old_value
                            if abs(delta) > 0.001:  # Only include non-zero changes
                                deltas[key] = delta
                    
                    if deltas:
                        relationship_changes[stakeholder_id] = deltas
                        
                        # Calculate overall impact for this stakeholder
                        impact = sum(deltas.values()) / len(deltas) if deltas else 0.0
                        stakeholder_impact[stakeholder_id] = impact
        
        # Display AI usage metrics
        self._display_in_scenario_metrics()
        
        # Return action results
        return {
            "action_id": action.id,
            "action_description": action.description,
            "relationship_changes": relationship_changes,
            "stakeholder_impact": stakeholder_impact
        }
    
    def _prepare_dialogue_context(self, action: Action, scenario_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for AI dialogue generation.
        
        Formats context with all required keys for AI dialogue generation:
        - stakeholder_profile: Information about the stakeholder
        - current_context: Information about the current scenario and action
        - relationship_status: Current relationship and changes
        - conversation_history: History of dialogue (empty for now)
        """
        # Start with basic scenario context as a base
        base_context = dict(scenario_context)
        
        # Create the structured dialogue context with all required top-level keys
        dialogue_context = {
            # These will be filled per stakeholder during dialogue generation
            "stakeholder_profile": {},
            "current_context": {
                "action": {
                    "id": action.id,
                    "title": action.description,  # Action has no title attribute, use description
                    "description": action.description,
                    "directly_affects": action.directly_affects,
                    "resource_cost": action.resource_cost,
                    "difficulty": action.difficulty,
                    "action_type": action.action_type
                },
                "scenario_title": self.current_scenario.metadata.title if self.current_scenario else "Unknown Scenario",
                "scenario_description": base_context.get("scenario_description", "Corporate scenario"),
                "scenario_phase": "action_response"
            },
            "relationship_status": {},  # Will be filled per stakeholder
            "conversation_history": [],  # Empty for now, could track history in future
            
            # Add technical_context required by AI prompt templates
            "technical_context": {
                "simulation_version": VERSION,  # Use module-level constant instead of instance attribute
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "engine": "corporate-simulation",
                "model": base_context.get("model", "default"),
                "generation_type": "dialogue"
            },
            
            # Additional context that might be helpful for the AI
            "current_action": {
                "id": action.id,
                "description": action.description,
                "directly_affects": action.directly_affects,
                "resource_cost": action.resource_cost,
                "difficulty": action.difficulty,
                "action_type": action.action_type
            },
            # Include any stakeholder profiles from the scenario context
            "stakeholder_profiles": base_context.get("stakeholder_profiles", {}),
            
            # Add previous_events (required by AI prompt template)
            "previous_events": base_context.get("previous_events", "This is the current scenario in progress."),
        }
        
        # Add current relationship status for all stakeholders
        for stakeholder in base_context.get("stakeholder_profiles", {}):
            stakeholder_status = self.relationship_matrix.get_stakeholder_status(stakeholder)
            if stakeholder_status:
                dialogue_context["relationship_status"][stakeholder] = stakeholder_status
        
        # Log that we're preparing complete dialogue context
        logging.debug(f"Prepared dialogue context with all required keys for AI integration")
                
        return dialogue_context
    
    # ------------------------------------------------------------------
    # Display methods for stakeholders, actions, and relationships
    # ------------------------------------------------------------------
    
    def _display_stakeholders(self, stakeholders: List[str]) -> None:
        """Display all stakeholders involved in the scenario."""
        table = Table(title="Key Stakeholders", box=box.ROUNDED, show_header=True)
        table.add_column("Stakeholder", style="cyan")
        table.add_column("Trust", style="green")
        table.add_column("Respect", style="yellow")
        table.add_column("Influence", style="magenta")
        
        for stakeholder in stakeholders:
            # Get current stakeholder status from relationship matrix
            status = self.relationship_matrix.get_stakeholder_status(stakeholder)
            if status:
                table.add_row(
                    stakeholder.replace('_', ' ').title(),
                    self._format_relationship_value(status.get("trust", 0.5)),
                    self._format_relationship_value(status.get("respect", 0.5)),
                    self._format_relationship_value(status.get("influence", 0.5))
                )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
    
    def _display_available_actions(self, actions: List[Action]) -> None:
        """Display available actions with formatted details."""
        self.console.rule("[bold blue]Available Actions[/]")
        
        for idx, action in enumerate(actions, start=1):
            # Format action details
            difficulty_display = "★" * int(round(action.difficulty * 5))
            affects = ", ".join(s.replace('_', ' ').title() for s in action.directly_affects)
            
            # Display formatted action
            self.console.print(
                f"[cyan]{idx}.[/] [bold]{action.description}[/]\n"
                f"   [dim]Type:[/] {action.action_type.capitalize()}  "
                f"[dim]Difficulty:[/] {difficulty_display}  "
                f"[dim]Affects:[/] {affects}\n"
            )
            
        self.console.print("[cyan]0.[/] [bold]Exit scenario[/]")
    
    def _display_stakeholder_dialogue(self, stakeholder_id: str, dialogue: str) -> None:
        """Display dialogue from a stakeholder with proper formatting."""
        # Get stakeholder details
        stakeholder_name = stakeholder_id.replace('_', ' ').title()
        
        # Get stakeholder status info for panel color
        status = self.relationship_matrix.get_stakeholder_status(stakeholder_id)
        trust = status.get("trust", 0.5) if status else 0.5
        
        # Choose panel color based on relationship
        panel_style = "green" if trust >= 0.7 else "yellow" if trust >= 0.4 else "red"
        
        # Format dialogue as markdown
        md_dialogue = Markdown(dialogue)
        
        # Display in panel
        self.console.print("\n")
        self.console.print(Panel(
            md_dialogue,
            title=f"[bold]{stakeholder_name}[/]",
            border_style=panel_style,
            expand=False
        ))
    
    def _update_stakeholder_relationships(self, action: Action) -> None:
        """Update stakeholder relationships based on action and display changes."""
        # Update relationships using matrix
        deltas = self.relationship_matrix.update_relationships_from_action(action)
        
        if not deltas:
            return
            
        # Display relationship changes
        self.console.rule("[bold blue]Relationship Impact[/]")
        
        # Create impact table
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Stakeholder", style="cyan")
        table.add_column("Trust", justify="center")
        table.add_column("Reasoning", justify="left")
        
        for stakeholder, delta in deltas.items():
            # Format stakeholder name
            name = stakeholder.replace('_', ' ').title()
            
            # RelationshipDelta has 'magnitude' attribute, not trust/respect/influence
            # Use the magnitude for display
            magnitude_formatted = self._format_delta(delta.magnitude)
            
            table.add_row(name, magnitude_formatted, delta.reasoning)
            
        self.console.print(table)
        self.console.print("\n")
    
    def _format_relationship_value(self, value: float) -> str:
        """Format relationship value as stars."""
        if value is None:
            return "☆ ☆ ☆"
            
        # Convert float (0.0-1.0) to star rating (0-5 stars)
        stars = int(round(value * 5))
        return "★" * stars + "☆" * (5 - stars)
    
    def _display_in_scenario_metrics(self) -> None:
        """Display a concise summary of AI model usage for the current day."""
        orchestrator = self.ai_integration.orchestrator
        if not orchestrator:
            return  # Silently fail if orchestrator not present

        # Model Usage Table
        model_usage_table = Table(title="AI Usage Snapshot (Today)", box=box.ROUNDED, show_header=True)
        model_usage_table.add_column("Model", style="cyan")
        model_usage_table.add_column("Requests", style="white")
        model_usage_table.add_column("Limit", style="white")
        model_usage_table.add_column("Status", style="white")

        # Match the models actually being used
        models_to_track = {
            "gemini-2.0-flash": 1500,
            "gemini-1.5-flash": 1500,
            "gemini-1.5-flash-002": 1500,
            "gemini-1.5-flash-8b": 50,
        }

        today_str = datetime.now().strftime('%Y-%m-%d')

        for model_name, limit in models_to_track.items():
            daily_count = orchestrator.cost_manager.usage_logger.get_daily_count(
                model_name=model_name, date_str=today_str
            )
            status_str = orchestrator.cost_manager.usage_logger.get_quota_status(
                model_name=model_name, daily_limit=limit
            )

            status_color_map = {
                "OK": "green",
                "WARNING": "yellow",
                "CRITICAL": "red",
            }
            status_color = status_color_map.get(status_str, "white")
            status_display = f"[{status_color}]{status_str}[/]"

            model_usage_table.add_row(
                model_name,
                str(daily_count),
                str(limit),
                status_display,
            )
        
        self.console.print(model_usage_table)
        self.console.print("\n")

    def _format_delta(self, value: float) -> str:
        """Format delta value with color and +/- sign."""
        if abs(value) < 0.001:
            return "[dim]--[/]"  # No significant change
        
        # Format with sign and color based on value
        if value > 0:
            return f"[green]+{value:.2f}[/]"
        else:
            return f"[red]{value:.2f}[/]"
            
    # ------------------------------------------------------------------
    # Save/Load functionality
    # ------------------------------------------------------------------
    
    def _save_game(self, save_path: Path) -> bool:
        """Save game state to file."""
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare save data
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "player_name": self.game_state.player_name,
                "completed_scenarios": self.game_state.game_metadata.get("scenarios_completed", 0),
                "relationships": self.relationship_matrix.export_relationships(),
                "current_scenario": self.current_scenario.metadata.scenario_id if self.current_scenario else None,
                "metadata": {
                    "version": "1.0",
                    "save_format": "corporate_simulator_standard"
                }
            }
            
            # Write save file
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.console.print(f"\n[green]Game saved successfully to {save_path}[/]")
            self.save_path = save_path
            return True
            
        except Exception as e:
            logger.error(f"Error saving game: {e}", exc_info=True)
            self.console.print(f"\n[bold red]Error saving game:[/] {e}")
            return False
    
    def _load_game(self) -> bool:
        """Load game state from file."""
        self.console.rule("[bold blue]Load Game[/]")
        
        # Check for save files
        save_files = list(SAVE_DIRECTORY.glob("*.json"))
        if not save_files:
            self.console.print("\n[yellow]No save files found![/]")
            return False
            
        # Display available save files
        self.console.print("\nAvailable saves:")
        for idx, save_file in enumerate(save_files, start=1):
            try:
                # Try to read save metadata
                with open(save_file, 'r') as f:
                    save_data = json.load(f)
                    
                # Display save info
                timestamp = save_data.get("timestamp", "Unknown date")
                player_name = save_data.get("player_name", "Unknown player")
                scenarios = len(save_data.get("completed_scenarios", []))
                
                self.console.print(
                    f"[cyan]{idx}.[/] {player_name} - "
                    f"[dim]{scenarios} completed scenarios - {timestamp}[/]"
                )
                
            except Exception:
                self.console.print(f"[cyan]{idx}.[/] [yellow]{save_file.name}[/] (corrupted)")
                
        # Add cancel option
        self.console.print("[cyan]0.[/] Cancel")
        
        # Get user choice
        choice = Prompt.ask(
            "\nSelect a save file", 
            choices=[str(i) for i in range(0, len(save_files) + 1)],
            default="1"
        )
        
        choice_idx = int(choice)
        if choice_idx == 0:
            return False
            
        # Load selected save
        selected_save = save_files[choice_idx - 1]
        try:
            # Read save file
            with open(selected_save, 'r') as f:
                save_data = json.load(f)
                
            # Restore game state
            self.game_state.player_name = save_data.get("player_name", "Executive")
            self.game_state.completed_scenarios = save_data.get("completed_scenarios", [])
            
            # Restore relationships
            self.relationship_matrix.import_relationships(save_data.get("relationships", {}))
            
            self.console.print(f"\n[green]Game '{self.game_state.player_name}' loaded successfully![/]")
            self.save_path = selected_save
            return True
            
        except Exception as e:
            logger.error(f"Error loading save: {e}", exc_info=True)
            self.console.print(f"\n[bold red]Error loading save:[/] {e}")
            return False
    
    # ------------------------------------------------------------------
    # Help and metrics display
    # ------------------------------------------------------------------
    
    def _display_help(self) -> None:
        """Display help information."""
        self.console.rule("[bold blue]Corporate Dynamics Simulator Help[/]")
        
        help_md = """
        # Corporate Dynamics Simulator Help
        
        ## Overview
        The Corporate Dynamics Simulator is an enterprise-grade simulation of corporate decision-making 
        and stakeholder relationship management. It features AI-generated scenarios and dialogues 
        that respond dynamically to your choices.
        
        ## Key Features
        * **AI-Generated Content**: Each scenario and dialogue is dynamically generated
        * **Stakeholder Relationships**: Your decisions affect trust, respect, and influence
        * **Multiple Scenarios**: 9 different corporate scenarios to tackle
        * **Save/Load**: Persistent game state across sessions
        
        ## How to Play
        1. Select a scenario from the available options
        2. Read the AI-generated scenario description
        3. Choose actions from the available options
        4. See how stakeholders respond with AI-generated dialogue
        5. Observe how relationships change based on your decisions
        
        ## Relationship System
        Each stakeholder has three key metrics:
        * **Trust**: How much they trust your judgment
        * **Respect**: How much they respect your position
        * **Influence**: How much influence they have with others
        
        Your actions will affect these metrics differently for each stakeholder.
        """
        
        self.console.print(Markdown(help_md))
        self.console.print("\nPress Enter to continue...")
        input()
    
    def _display_ai_metrics(self) -> None:
        """Display detailed AI service metrics, cost, and usage information."""
        self.console.rule("[bold blue]AI Integration Metrics[/]")

        orchestrator = self.ai_integration.orchestrator
        if not orchestrator:
            self.console.print("[yellow]AI Orchestrator not available.[/]")
            return

        # AI Service Metrics Table (including cost)
        usage_stats = orchestrator.cost_manager.usage_logger.get_total_stats()
        
        service_table = Table(title="AI Service Metrics", box=box.ROUNDED)
        service_table.add_column("Metric", style="cyan")
        service_table.add_column("Value", style="white")

        service_table.add_row("Total Requests", str(usage_stats.get("total_requests", 0)))
        service_table.add_row("Cache Hits", "0")  # Not tracked in usage_logger
        service_table.add_row("Total Cost", f"${usage_stats.get('estimated_cost', 0.0):.6f}")
        service_table.add_row("Total Tokens", str(usage_stats.get("total_tokens", 0)))

        self.console.print("\n")
        self.console.print(service_table)
        self.console.print("\n")

        # Model Usage Table - Update model list to match what's actually being used
        model_usage_table = Table(title="Model Usage", box=box.ROUNDED)
        model_usage_table.add_column("Model", style="cyan")
        model_usage_table.add_column("Requests Today", style="white")
        model_usage_table.add_column("Daily Limit", style="white")
        model_usage_table.add_column("Status", style="white")

        # These should match the models actually being used by Gemini
        models_to_track = {
            "gemini-2.0-flash": 1500,
            "gemini-1.5-flash": 1500,
            "gemini-1.5-flash-002": 1500,
            "gemini-1.5-flash-8b": 50,
        }

        today_str = datetime.now().strftime('%Y-%m-%d')

        for model_name, limit in models_to_track.items():
            daily_count = orchestrator.cost_manager.usage_logger.get_daily_count(
                model_name=model_name, date_str=today_str
            )
            status_str = orchestrator.cost_manager.usage_logger.get_quota_status(
                model_name=model_name, daily_limit=limit
            )

            status_color_map = {
                "OK": "green",
                "WARNING": "yellow",
                "CRITICAL": "red",
            }
            status_color = status_color_map.get(status_str, "white")
            status_display = f"[{status_color}]{status_str}[/]"

            model_usage_table.add_row(
                model_name,
                str(daily_count),
                str(limit),
                status_display,
            )
        
        self.console.print(model_usage_table)
        self.console.print("\n")

        # Provider Health Table - Aggregated from Model Usage
        health_table = Table(title="Provider Health", box=box.ROUNDED)
        health_table.add_column("Provider", style="cyan")
        health_table.add_column("Status", style="white")
        health_table.add_column("Latency", style="yellow")
        health_table.add_column("Total Requests", style="green")
        health_table.add_column("Success Rate", style="green")
        
        # Aggregate model requests by provider
        provider_requests = {}
        for model_name in models_to_track.keys():
            provider = model_name.split('-')[0].upper()  # Extract provider name (e.g., 'gemini' from 'gemini-2.0-flash')
            daily_count = orchestrator.cost_manager.usage_logger.get_daily_count(
                model_name=model_name, 
                date_str=today_str
            )
            provider_requests[provider] = provider_requests.get(provider, 0) + daily_count
        
        # Add rows for each provider
        for provider, total_requests in sorted(provider_requests.items()):
            # Set status based on request count
            status = "healthy" if total_requests > 0 else "unknown"
            status_color = "green" if status == "healthy" else "yellow"
            
            health_table.add_row(
                provider,
                f"[{status_color}]{status.upper()}[/]",
                "N/A",  # Latency not currently tracked
                str(total_requests),
                "100.00%"  # Assuming 100% success rate for completed requests
            )
            
        self.console.print(health_table)
        self.console.print("\nPress Enter to continue...")
        input()


async def main() -> None:
    """Main entry point for the Corporate Dynamics Simulator."""
    # Set up logging configuration
    from src.utils.logging_config import LoggingMode
    
    # Get logging mode and map to logging levels
    logging_mode = get_logging_mode_from_env()
    log_levels = {
        LoggingMode.SILENT: logging.CRITICAL,
        LoggingMode.PRODUCTION: logging.WARNING,
        LoggingMode.DEMO: logging.WARNING,
        LoggingMode.DEVELOPMENT: logging.DEBUG,
    }
    log_level = log_levels.get(logging_mode, logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('corporate_simulator.log')
        ]
    )
    
    # Set logging levels for specific loggers
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    
    # Log application start
    logging.info("Starting Corporate Dynamics Simulator")
    logging.info(f"Logging level set to: {logging_mode}")
    
    # Check for API keys in environment variables
    api_keys = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY")
    }
    
    # Log which API keys were found
    keys_found = [k for k, v in api_keys.items() if v]
    if keys_found:
        logging.info(f"Found API keys for: {', '.join(keys_found)}")
    else:
        logging.warning("No API keys found in environment variables. Some features may be limited.")
    
    try:
        # Create and run the simulator
        simulator = CorporateSimulator(api_keys)
        await simulator.run()
    except Exception as e:
        logging.critical(f"Fatal error in Corporate Dynamics Simulator: {e}", exc_info=True)
        print(f"A critical error occurred: {e}")
        print("Check corporate_simulator.log for details.")
        sys.exit(1)

    # Display exit message
    console = Console()
    console.rule("[bold blue]Corporate Dynamics Simulator[/]")
    console.print("\n[green]Thank you for using the Corporate Dynamics Simulator![/]")
    console.print("Exiting gracefully...\n")


# Entry point
if __name__ == "__main__":
    # Configure asyncio to use the appropriate event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main function with proper async handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logging.exception("Unhandled exception in main")
        sys.exit(1)
