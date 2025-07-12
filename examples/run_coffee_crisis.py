"""Convenience runner for the Coffee Machine Crisis interactive CLI demo."""
import sys
import asyncio
from pathlib import Path as _P
from typing import List, Dict, Any, Optional

# Ensure project root on path for direct execution
_ROOT = _P(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from corporate_simulator import CorporateSimulator
from src.scenarios.coffee_machine_crisis import CoffeeMachineCrisis
from src.state_management.data_models import Action  # type: ignore
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt


async def run_coffee_crisis(cli: CorporateSimulator, scenario: CoffeeMachineCrisis) -> None:
    """Run the coffee machine crisis scenario with proper async handling."""
    console = Console()
    
    # Define the coffee machine crisis specific stakeholders - just the key ones
    coffee_crisis_stakeholders = [
        "ceo", "office_manager", "it_director", "hr_manager", "finance_director"
    ]
    
    # Create a customized scenario context for the coffee crisis
    scenario_context = {
        "context_description": "The office coffee machine has broken down during a critical project week.",
        "stakeholder_descriptions": {
            "ceo": "Concerned about employee morale but focused on bigger issues",
            "office_manager": "Directly responsible for office amenities",
            "it_director": "Believes coffee machine is not an IT issue",
            "hr_manager": "Worried about impact on employee satisfaction",
            "finance_director": "Concerned about unnecessary expenses"  
        },
        "stakeholder_profiles": {
            "ceo": {
                "name": "Alex Johnson",
                "role": "CEO",
                "background": "Concerned about employee morale but focused on bigger issues",
                "priorities": ["company growth", "investor relations", "strategic planning"],
                "trust": 0.70,
                "respect": 0.80,
                "influence": 0.90
            },
            "office_manager": {
                "name": "Morgan Lee",
                "role": "Office Manager",
                "background": "Directly responsible for office amenities",
                "priorities": ["employee comfort", "office functionality", "vendor management"],
                "trust": 0.50,
                "respect": 0.60,
                "influence": 0.50
            },
            "it_director": {
                "name": "Taylor Smith",
                "role": "IT Director",
                "background": "Believes coffee machine is not an IT issue",
                "priorities": ["system uptime", "cybersecurity", "technical debt reduction"],
                "trust": 0.60,
                "respect": 0.70,
                "influence": 0.60
            },
            "hr_manager": {
                "name": "Jordan Rivera",
                "role": "HR Manager",
                "background": "Worried about impact on employee satisfaction",
                "priorities": ["employee satisfaction", "retention", "workplace culture"],
                "trust": 0.65,
                "respect": 0.75,
                "influence": 0.55
            },
            "finance_director": {
                "name": "Casey Wong",
                "role": "Finance Director",
                "background": "Concerned about unnecessary expenses",
                "priorities": ["cost control", "budget adherence", "ROI analysis"],
                "trust": 0.65,
                "respect": 0.70,
                "influence": 0.75
            }
        },
        "previous_events": [
            "The coffee machine has been malfunctioning for weeks",
            "Several employees have complained about coffee quality",
            "A recent budget review has put all non-essential purchases on hold"
        ]
    }
    
    # Initialize the scenario with our context
    cli.current_scenario = scenario
    cli.current_scenario_data = scenario_context
    
    # Display scenario header
    console.rule(f"[bold blue]{scenario.metadata.title}[/]")
    
    # Display custom scenario description
    coffee_crisis_description = """
    ## The Coffee Machine Crisis
    
    The office coffee machine has broken down completely during a critical project week. 
    Multiple teams are working overtime to meet deadlines, and the lack of caffeine is 
    causing productivity and morale issues across the company.
    
    As the project lead, you need to resolve this crisis while navigating the competing 
    priorities of different stakeholders.
    
    Your decisions will affect stakeholder relationships and the overall project outcome.
    """
    
    console.print(Panel(
        Markdown(coffee_crisis_description),
        title=scenario.metadata.title,
        subtitle=f"Difficulty: {scenario.metadata.difficulty:.1f}/1.0"
    ))
    
    # Display only the coffee crisis relevant stakeholders
    display_coffee_crisis_stakeholders(cli, coffee_crisis_stakeholders)
    
    # Define specific actions for the coffee crisis
    coffee_actions = [
        Action(
            id="repair",
            description="Call repair service to fix the existing machine",
            directly_affects=["office_manager", "finance_director", "ceo"],
            resource_cost={"budget": 200.0},
            difficulty=0.3,
            action_type="maintenance",
            relationship_deltas={
                "office_manager": 0.2,
                "finance_director": 0.1,
                "ceo": -0.05
            }
        ),
        Action(
            id="replacement", 
            description="Purchase a new premium coffee machine",
            directly_affects=["hr_manager", "ceo", "finance_director"],
            resource_cost={"budget": 800.0},
            difficulty=0.6,
            action_type="purchase",
            relationship_deltas={
                "hr_manager": 0.3,
                "ceo": 0.1,
                "finance_director": -0.3
            }
        ),
        Action(
            id="temporary",
            description="Set up temporary coffee station with single-serve brewers",
            directly_affects=["office_manager", "hr_manager", "finance_director", "it_director"],
            resource_cost={"budget": 100.0, "time": 2.0},
            difficulty=0.2,
            action_type="workaround",
            relationship_deltas={
                "office_manager": 0.15,
                "hr_manager": 0.1,
                "finance_director": 0.05,
                "it_director": 0.05
            }
        ),
        Action(
            id="vouchers",
            description="Distribute coffee shop vouchers to employees",
            directly_affects=["hr_manager", "ceo", "finance_director", "office_manager"],
            resource_cost={"budget": 500.0},
            difficulty=0.4,
            action_type="external",
            relationship_deltas={
                "hr_manager": 0.2,
                "ceo": 0.15,
                "finance_director": -0.2,
                "office_manager": -0.1
            }
        )
    ]
    
    # Display actions and get user choice
    console.print("\n[bold green]Available Actions:[/]")
    for i, action in enumerate(coffee_actions):
        console.print(f"[bold]{i+1}[/]. {action.description} (Impact: {impact_summary(action.relationship_deltas or {})})")
    
    # Get user choice
    choice = Prompt.ask("\nSelect an action", choices=[str(i+1) for i in range(len(coffee_actions))])
    selected_action = coffee_actions[int(choice) - 1]
    
    # Show the results of the action
    console.print(f"\n[bold]You selected:[/] {selected_action.description}")
    console.print("\n[bold]Stakeholder Impact:[/]")
    
    # Display impact details
    display_action_results(console, selected_action, coffee_crisis_stakeholders)
    
    # Show resolution
    console.rule("[bold]Crisis Resolution[/]")
    
    # Generate a resolution based on the selected action
    resolutions = {
        "repair": "The repair service was called and fixed the machine after a 2-day wait. Some teams were unhappy with the downtime, but most understood the fiscal responsibility shown.",
        "replacement": "The new premium coffee machine was ordered and installed. It boosted morale significantly, though Finance expressed concerns about the expense during budget reviews.",
        "temporary": "The temporary coffee station was quickly set up with single-serve brewers. While not as efficient as the main machine, it provided a workable solution that kept the teams productive.",
        "vouchers": "The coffee shop vouchers were well received by the staff. The finance team noted the unexpected expense, but the CEO was impressed with the creative solution to maintain morale."
    }
    
    console.print(Markdown(f"### Outcome\n\n{resolutions[selected_action.id]}"))
    
    # Display final stakeholder status
    console.print("\n[bold]Final Stakeholder Status:[/]")
    update_and_display_stakeholder_status(cli, selected_action, coffee_crisis_stakeholders)


def impact_summary(deltas: Dict[str, float]) -> str:
    """Generate a readable summary of stakeholder relationship impacts."""
    positive = sum(1 for v in deltas.values() if v > 0)
    negative = sum(1 for v in deltas.values() if v < 0)
    neutral = sum(1 for v in deltas.values() if v == 0)
    
    if positive > 0 and negative == 0:
        return f"+{positive} stakeholders 👍"
    elif negative > 0 and positive == 0:
        return f"-{negative} stakeholders 👎"
    else:
        return f"+{positive}/-{negative} mixed 🔄"


def display_coffee_crisis_stakeholders(cli: CorporateSimulator, stakeholders: List[str]) -> None:
    """Display only the stakeholders relevant to the coffee machine crisis."""
    from rich.table import Table
    
    # Create a table for stakeholder status
    table = Table(title="Stakeholder Status")
    table.add_column("Stakeholder")
    table.add_column("Trust")
    table.add_column("Respect")
    table.add_column("Influence")
    table.add_column("Overall")
    
    # Add data for only the relevant stakeholders
    for stakeholder_id in stakeholders:
        # Get profile from the stored profiles
        profile = cli.current_scenario_data["stakeholder_profiles"][stakeholder_id]
        name = stakeholder_id.replace("_", " ").title()
        
        trust = profile.get("trust", 0.5)
        respect = profile.get("respect", 0.5)
        influence = profile.get("influence", 0.5)
        overall = round((trust + respect + influence) / 3, 2)
        
        table.add_row(
            name,
            f"{trust:.2f}",
            f"{respect:.2f}",
            f"{influence:.2f}",
            f"{overall:.2f}"
        )
    
    cli.console.print(table)


def display_action_results(console: Console, action: Action, stakeholders: List[str]) -> None:
    """Display the results of taking an action."""
    from rich.table import Table
    
    table = Table()
    table.add_column("Stakeholder")
    table.add_column("Impact")
    table.add_column("Reaction")
    
    relationship_deltas = action.relationship_deltas or {}
    
    for stakeholder_id in stakeholders:
        name = stakeholder_id.replace("_", " ").title()
        impact = relationship_deltas.get(stakeholder_id, 0.0)
        
        # Generate reaction text based on impact
        if impact > 0.2:
            reaction = "Very Positive 😄"
        elif impact > 0:
            reaction = "Positive 🙂"
        elif impact < -0.2:
            reaction = "Very Negative 😡"
        elif impact < 0:
            reaction = "Negative 😕"
        else:
            reaction = "Neutral 😐"
        
        # Format impact as string with color
        if impact > 0:
            impact_str = f"[green]+{impact:.2f}[/]"
        elif impact < 0:
            impact_str = f"[red]{impact:.2f}[/]"
        else:
            impact_str = f"0.00"
            
        table.add_row(name, impact_str, reaction)
    
    console.print(table)


def update_and_display_stakeholder_status(cli: CorporateSimulator, action: Action, stakeholders: List[str]) -> None:
    """Update stakeholder status based on action and display."""
    from rich.table import Table
    
    # Make a copy of the profiles with updates
    updated_profiles = {}
    relationship_deltas = action.relationship_deltas or {}
    
    for stakeholder_id in stakeholders:
        # Get original profile
        original = cli.current_scenario_data["stakeholder_profiles"][stakeholder_id]
        
        # Create an updated profile
        updated = {**original}  # Make a copy
        
        # Apply impact to trust, respect, and influence
        impact = relationship_deltas.get(stakeholder_id, 0.0)
        
        # Update attributes with scaled impacts
        updated["trust"] = min(1.0, max(0.0, original.get("trust", 0.5) + impact * 0.8))
        updated["respect"] = min(1.0, max(0.0, original.get("respect", 0.5) + impact * 0.6))
        updated["influence"] = min(1.0, max(0.0, original.get("influence", 0.5) + impact * 0.2))
        
        updated_profiles[stakeholder_id] = updated
    
    # Create a status table
    table = Table()
    table.add_column("Stakeholder")
    table.add_column("Trust")
    table.add_column("Trust Δ")
    table.add_column("Respect")
    table.add_column("Respect Δ")
    table.add_column("Influence")
    table.add_column("Overall")
    
    # Add data for stakeholders
    for stakeholder_id in stakeholders:
        name = stakeholder_id.replace("_", " ").title()
        original = cli.current_scenario_data["stakeholder_profiles"][stakeholder_id]
        updated = updated_profiles[stakeholder_id]
        
        # Get values and deltas
        trust = updated["trust"]
        trust_delta = trust - original.get("trust", 0.5)
        respect = updated["respect"]
        respect_delta = respect - original.get("respect", 0.5)
        influence = updated["influence"]
        overall = round((trust + respect + influence) / 3, 2)
        
        # Format deltas with colors
        if trust_delta > 0:
            trust_delta_str = f"[green]+{trust_delta:.2f}[/]"
        elif trust_delta < 0:
            trust_delta_str = f"[red]{trust_delta:.2f}[/]"
        else:
            trust_delta_str = f"0.00"
            
        if respect_delta > 0:
            respect_delta_str = f"[green]+{respect_delta:.2f}[/]"
        elif respect_delta < 0:
            respect_delta_str = f"[red]{respect_delta:.2f}[/]"
        else:
            respect_delta_str = f"0.00"
        
        table.add_row(
            name,
            f"{trust:.2f}", trust_delta_str,
            f"{respect:.2f}", respect_delta_str,
            f"{influence:.2f}",
            f"{overall:.2f}"
        )
    
    cli.console.print(table)


async def main_async() -> None:  # noqa: D401
    """Async entry point for running the coffee crisis scenario."""
    cli = CorporateSimulator()
    print("🏢 Corporate Dynamics Simulation Engine")
    print("Enterprise-grade stakeholder modeling with AI integration\n")
    
    try:
        # Create coffee machine crisis scenario and run it with async handling
        coffee_crisis = CoffeeMachineCrisis()
        await run_coffee_crisis(cli, coffee_crisis)
    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")
    except Exception as e:
        print(f"\n[red]Error:[/] {e.__class__.__name__}: {e}")


def main() -> None:
    """Entry point for running the async coffee crisis scenario."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
