"""Prompt Engineering Engine for Corporate Dynamics Simulator.

This module provides advanced prompt engineering capabilities for the AI integration
layer, focusing on creating high-quality, efficient prompts for different scenarios
and character dialogue in the corporate simulation.

The PromptEngineeringEngine handles:
- Template management for different content types
- Context assembly and relevant information extraction
- Token optimization for cost efficiency
- Narrative consistency in dialogue generation
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from .data_models import AIConfig, AIRequest, ContentType

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    """Template for generating AI prompts."""

    name: str
    content_type: ContentType
    template: str
    required_context_keys: List[str] = Field(default_factory=list)
    max_tokens: int = 1000
    
    model_config = {"arbitrary_types_allowed": True}


class PromptEngineeringEngine:
    """Engine for generating optimized prompts for AI services.
    
    Handles prompt template management, context assembly, and token optimization.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize the prompt engineering engine.
        
        Args:
            config: Configuration for the prompt engineering engine.
        """
        self.config = config or AIConfig()
        self._load_default_templates()
        
    def _load_default_templates(self) -> None:
        """Load default prompt templates for different content types."""
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Scenario prompt templates
        self.templates["standard_scenario"] = PromptTemplate(
            name="standard_scenario",
            content_type=ContentType.SCENARIO,
            template="""
You are generating a corporate scenario for an enterprise simulation focused on technology leadership dynamics.

CONTEXT:
{context_description}

STAKEHOLDERS:
{stakeholder_descriptions}

PREVIOUS EVENTS:
{previous_events}

CONSTRAINTS:
1. The scenario should involve multiple stakeholders with competing interests
2. It should present a realistic technology leadership challenge
3. It must offer 3-5 distinct action choices with different relationship impacts
4. All content must be appropriate for a professional corporate environment

Generate a JSON response with the following structure:
```json
{
  "description": "Detailed scenario description (2-3 paragraphs)",
  "stakeholders_involved": ["List of stakeholder IDs directly involved"],
  "available_actions": [
    {
      "id": "action_id_in_snake_case",
      "label": "Short action label (5-8 words)",
      "action_description": "Detailed description of the action and its potential impacts",
      "directly_affects": ["stakeholder_ids"]
    }
  ],
  "background_context": "Additional relevant context for player decision-making"
}
```
            """,
            required_context_keys=["context_description", "stakeholder_descriptions", "previous_events"],
        )
        
        self.templates["crisis_scenario"] = PromptTemplate(
            name="crisis_scenario",
            content_type=ContentType.SCENARIO,
            template="""
You are generating a corporate crisis scenario for an enterprise simulation focused on technology leadership under pressure.

CONTEXT:
{context_description}

STAKEHOLDERS:
{stakeholder_descriptions}

CRISIS PARAMETERS:
{crisis_parameters}

CONSTRAINTS:
1. The scenario should involve an urgent technology crisis requiring immediate decision
2. It should have potential impacts on business continuity, reputation, or security
3. It must offer 3-5 distinct crisis response options with significant tradeoffs
4. All content must be realistic and appropriate for enterprise technology leaders

Generate a JSON response with the following structure:
```json
{
  "crisis_title": "Short descriptive title",
  "description": "Detailed crisis description (2-3 paragraphs)",
  "stakeholders_involved": ["List of stakeholder IDs directly involved"],
  "severity": "Critical|High|Medium",
  "available_actions": [
    {
      "id": "action_id_in_snake_case",
      "label": "Short action label (5-8 words)",
      "action_description": "Detailed description of the action and its potential impacts",
      "directly_affects": ["stakeholder_ids"]
    }
  ],
  "time_pressure": "Description of time constraints and urgency"
}
```
            """,
            required_context_keys=["context_description", "stakeholder_descriptions", "crisis_parameters"],
        )
        
        # Dialogue prompt templates
        self.templates["standard_dialogue"] = PromptTemplate(
            name="standard_dialogue",
            content_type=ContentType.DIALOGUE,
            template="""
You are generating authentic dialogue for a corporate stakeholder in an enterprise simulation.

STAKEHOLDER PROFILE:
{stakeholder_profile}

CURRENT CONTEXT:
{current_context}

RELATIONSHIP STATUS:
{relationship_status}

CONVERSATION HISTORY:
{conversation_history}

CONSTRAINTS:
1. The dialogue must authentically reflect the stakeholder's personality, role, and perspective
2. It should respond directly to the current context and conversation history
3. The tone should match the stakeholder's current relationship with the player
4. All content must be realistic and appropriate for a professional corporate environment

Generate a JSON response with the following dialogue:
```json
{
  "speaker": "{stakeholder_id}",
  "content": "The dialogue content with appropriate tone and perspective",
  "tone": "professional|concerned|supportive|frustrated|neutral",
  "body_language": "Brief description of non-verbal cues if applicable",
  "relationship_impact": "positive|negative|neutral"
}
```
            """,
            required_context_keys=["stakeholder_profile", "current_context", "relationship_status", "conversation_history"],
        )
        
        # Consequence prompt templates
        self.templates["action_consequence"] = PromptTemplate(
            name="action_consequence",
            content_type=ContentType.CONSEQUENCE,
            template="""
You are generating realistic consequences for a decision made in a corporate technology leadership simulation.

DECISION CONTEXT:
{decision_context}

PLAYER ACTION:
{player_action}

STAKEHOLDER REACTIONS:
{stakeholder_reactions}

CONSTRAINTS:
1. The consequences should realistically follow from the player's action
2. Include both immediate and potential long-term effects
3. Reference specific stakeholders and their changing perspectives
4. All content must be realistic for enterprise technology environments

Generate a JSON response with the following consequence details:
```json
{
  "immediate_result": "Description of what happens immediately after the decision",
  "stakeholder_impacts": {
    "stakeholder_id1": "Specific reaction and impact",
    "stakeholder_id2": "Specific reaction and impact"
  },
  "relationship_changes": {
    "stakeholder_id1": "improved|damaged|unchanged",
    "stakeholder_id2": "improved|damaged|unchanged"
  },
  "narrative_consequences": "Broader impacts on the ongoing scenario",
  "follow_up_events": ["Potential future events that might occur as a result"]
}
```
            """,
            required_context_keys=["decision_context", "player_action", "stakeholder_reactions"],
        )
        
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template.
        
        Args:
            template: The prompt template to register.
        """
        self.templates[template.name] = template
        logger.info(f"Registered prompt template: {template.name}")
        
    def _extract_template_variables(self, template_str: str) -> Set[str]:
        """Extract all variables from a template string.
        
        Finds all patterns like {variable} in the template, including complex nested structures.
        
        Args:
            template_str: The template string to analyze.
            
        Returns:
            Set[str]: Set of variable names found in the template.
        """
        import re
        import string
        
        # First, handle standard Python string formatting variables
        formatter = string.Formatter()
        standard_vars = {field_name for _, field_name, _, _ in formatter.parse(template_str) if field_name is not None}
        
        # Special handling for JSON structures in templates
        # Look for patterns that might be part of JSON structures but aren't caught by the formatter
        json_patterns = [
            r'\{([^{}]+)\}',  # Standard format variables
        ]
        
        json_vars = set()
        for pattern in json_patterns:
            matches = re.findall(pattern, template_str)
            json_vars.update(matches)
        
        # Combine all variables
        all_vars = standard_vars.union(json_vars)
        
        return all_vars
        
    def _is_ai_generated_field(self, var_name: str) -> bool:
        """Determine if a variable is meant to be AI-generated content.
        
        Args:
            var_name: The variable name to check.
            
        Returns:
            bool: True if this is an AI-generated field, False otherwise.
        """
        # List of fields that are typically AI-generated and should not be replaced with placeholders
        ai_generated_fields = {
            # Common scenario fields
            'stakeholders_involved', 'available_actions', 'background_context',
            'description', 'immediate_result', 'stakeholder_impacts',
            'relationship_changes', 'narrative_consequences', 'follow_up_events',
            
            # Common action fields
            'id', 'label', 'action_description', 'directly_affects',
            
            # Common dialogue fields
            'content', 'tone', 'body_language', 'relationship_impact',
            
            # Common JSON structure fields
            'List of stakeholder IDs directly involved', 'stakeholder_ids',
        }
        
        # Check if the variable name is in our list of AI-generated fields
        if var_name in ai_generated_fields:
            return True
            
        # Check for common patterns in AI-generated fields
        if any(pattern in var_name for pattern in ['_id', '_ids', '_description', '_impact', '_result']):
            return True
            
        return False
        
    def create_prompt(
        self,
        template_name: str,
        context: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> AIRequest:
        """Create an AI request with an assembled prompt from a template.
        
        Args:
            template_name: Name of the template to use.
            context: Context information for populating the template.
            max_tokens: Maximum tokens for the response, overrides template default.
            
        Returns:
            AIRequest: The assembled AI request.
            
        Raises:
            ValueError: If template not found or required context missing.
        """
        # Get the template
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Validate required context keys
        missing_keys = [key for key in template.required_context_keys if key not in context]
        if missing_keys:
            raise ValueError(f"Missing required context keys: {missing_keys}")
        
        # Create a safe context dict with common defaults
        safe_context = {
            # Common defaults that might be referenced in templates
            "speaker": context.get("stakeholder_id", "unknown"),
            "description": "No description provided",
            "stakeholder_id": context.get("stakeholder_id", "unknown"),
            # Add common JSON keys that might cause problems
            '\n  "description"': "Corporate scenario requiring stakeholder management and strategic decision making.",
        }
        
        # Update safe_context with actual context values
        safe_context.update(context)
        
        # Extract all variables from the template
        template_variables = self._extract_template_variables(template.template)
        
        # Check for missing variables and add placeholders
        missing_variables = [var for var in template_variables if var not in safe_context]
        if missing_variables:
            for var in missing_variables:
                # Skip empty variables
                if not var or var.isspace():
                    continue
                    
                # Skip AI-generated fields - these should be left for the AI to fill in
                if self._is_ai_generated_field(var):
                    logger.debug(f"Skipping AI-generated field: {var}")
                    continue
                    
                # Log missing variables as warnings
                logger.warning(f"Missing template variable: {var} - adding placeholder")
                
                # Add placeholder for the missing variable
                safe_context[var] = f"[PLACEHOLDER: {var}]"
                
                # Handle common JSON formatting variations
                if var.startswith('"') and var.endswith('"'):
                    # Handle quoted variables
                    clean_var = var.strip('"')
                    if clean_var not in safe_context and not self._is_ai_generated_field(clean_var):
                        safe_context[clean_var] = f"[PLACEHOLDER: {clean_var}]"
                elif not self._is_ai_generated_field(var):
                    quoted_var = '"' + var + '"'
                    if quoted_var not in safe_context:
                        safe_context[quoted_var] = f"[PLACEHOLDER: {var}]"
                
                # Handle JSON keys with newlines
                if '\n' in var:
                    clean_var = var.replace('\n', '').strip()
                    if clean_var not in safe_context and not self._is_ai_generated_field(clean_var):
                        safe_context[clean_var] = f"[PLACEHOLDER: {clean_var}]"
                        
                # Handle JSON nested structures
                if ':' in var:
                    # This might be a JSON key-value pair
                    key_part = var.split(':', 1)[0].strip()
                    if key_part and key_part not in safe_context and not self._is_ai_generated_field(key_part):
                        safe_context[key_part] = f"[PLACEHOLDER: {key_part}]"
                        
                # Handle multiline JSON structures with indentation
                if var.strip().startswith('"') and '"' in var[1:]:
                    # Extract just the key part from something like '"id": "action_id_in_snake_case"'
                    key_match = re.search(r'"([^"]+)"', var)
                    if key_match:
                        key_only = key_match.group(1)
                        if key_only not in safe_context and not self._is_ai_generated_field(key_only):
                            safe_context[key_only] = f"[PLACEHOLDER: {key_only}]"
        
        # Format the template with the safe context
        try:
            prompt = template.template.format(**safe_context)
        except KeyError as e:
            # If we still get a KeyError despite our precautions, log it and try to fix it
            key = str(e).strip("'\n \t")
            logger.warning(f"Handling unexpected template variable: {key}")
            
            # Add this specific missing key with a placeholder
            safe_context[key] = f"[PLACEHOLDER: {key}]"
            
            # Also try variations of the key
            variations = [
                key.strip('"'),  # Without quotes
                f'"{key}"',     # With quotes
                key.replace('\n', '').strip(),  # Without newlines
                '\n' + key,    # With leading newline
                key + '\n',    # With trailing newline
            ]
            
            for var in variations:
                if var and var not in safe_context:
                    safe_context[var] = f"[PLACEHOLDER: {var}]"
            
            # Try again with the enhanced context
            try:
                prompt = template.template.format(**safe_context)
            except Exception as final_e:
                # If it still fails, use a more aggressive approach
                logger.warning(f"Still having formatting issues: {final_e}, trying string replacement")
                
                # Use string replacement as a last resort
                prompt = template.template
                for var_name in template_variables:
                    placeholder = f"{{{var_name}}}"
                    if placeholder in prompt:
                        replacement = f"[PLACEHOLDER: {var_name}]"
                        prompt = prompt.replace(placeholder, replacement)
                        
                # If we still have formatting placeholders, log it but continue
                if '{' in prompt and '}' in prompt:
                    logger.info("Template may still contain unresolved placeholders, but continuing with best effort")
        
        # Create the request
        return AIRequest(
            content_type=template.content_type,
            prompt_template=prompt,
            context=context,
            max_tokens=max_tokens or template.max_tokens,
        )
    
    def optimize_context(
        self, 
        context: Dict[str, Any], 
        template_name: str,
        token_limit: int = 2000
    ) -> Dict[str, Any]:
        """Optimize context for token efficiency.
        
        Selectively includes only relevant information to stay within token limits.
        
        Args:
            context: The original context dictionary.
            template_name: The template to optimize for.
            token_limit: Maximum tokens for context.
            
        Returns:
            Dict[str, Any]: The optimized context.
        """
        # If context is already small enough, return as is
        if self._estimate_tokens(json.dumps(context)) <= token_limit:
            return context
        
        optimized_context = {}
        template = self.templates.get(template_name)
        
        if not template:
            # If template not found, just do basic optimization
            return self._basic_context_optimization(context, token_limit)
        
        # Prioritize required context keys
        for key in template.required_context_keys:
            if key in context:
                optimized_context[key] = context[key]
        
        # Add other context keys as space allows
        remaining_keys = [k for k in context if k not in optimized_context]
        for key in remaining_keys:
            temp_context = optimized_context.copy()
            temp_context[key] = context[key]
            
            if self._estimate_tokens(json.dumps(temp_context)) <= token_limit:
                optimized_context[key] = context[key]
        
        return optimized_context
    
    def _basic_context_optimization(
        self, context: Dict[str, Any], token_limit: int
    ) -> Dict[str, Any]:
        """Perform basic context optimization without template guidance.
        
        Args:
            context: The original context.
            token_limit: Maximum tokens.
            
        Returns:
            Dict[str, Any]: The optimized context.
        """
        # Priority keys that should be preserved if possible
        priority_keys = [
            "stakeholder_descriptions",
            "current_context",
            "player_action",
            "stakeholder_profile",
        ]
        
        optimized = {}
        
        # Add priority keys first
        for key in priority_keys:
            if key in context:
                optimized[key] = context[key]
        
        # Add other keys as space allows
        remaining_keys = sorted(
            [k for k in context if k not in optimized],
            key=lambda k: len(str(context[k]))  # Sort by size (smallest first)
        )
        
        for key in remaining_keys:
            temp = optimized.copy()
            temp[key] = context[key]
            
            if self._estimate_tokens(json.dumps(temp)) <= token_limit:
                optimized[key] = context[key]
        
        return optimized
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            int: Estimated token count.
        """
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4
        
    def create_scenario_prompt(self, context: Dict[str, Any], max_tokens: Optional[int] = None) -> AIRequest:
        """Create a scenario prompt based on context.
        
        Args:
            context: The scenario context information.
            max_tokens: Maximum tokens for response generation.
            
        Returns:
            AIRequest: The assembled scenario prompt.
        """
        # Default to standard scenario unless context indicates crisis
        template_name = "crisis_scenario" if context.get("is_crisis", False) else "standard_scenario"
        
        # Add appropriate safe defaults for required keys if missing
        safe_context = context.copy()
        if "context_description" not in safe_context:
            safe_context["context_description"] = "A corporate environment facing strategic technology decisions."
            
        if "stakeholder_descriptions" not in safe_context:
            safe_context["stakeholder_descriptions"] = "Key stakeholders with various perspectives and priorities."
            
        if "previous_events" not in safe_context:
            if "recent_events" in context:
                safe_context["previous_events"] = ", ".join(context["recent_events"])
            else:
                safe_context["previous_events"] = "No significant recent events."
                
        # Create the prompt using the appropriate template
        return self.create_prompt(template_name, safe_context, max_tokens)
    
    def create_character_dialogue_prompt(self, character_id: str, context: Dict[str, Any], max_tokens: Optional[int] = None) -> AIRequest:
        """Create a dialogue prompt for a specific character.
        
        Args:
            character_id: ID of the character to generate dialogue for.
            context: Context information for the dialogue.
            max_tokens: Maximum tokens for response generation.
            
        Returns:
            AIRequest: The assembled dialogue prompt.
        """
        # Add character ID to context if not present
        safe_context = context.copy()
        safe_context["stakeholder_id"] = character_id
        
        # Add appropriate safe defaults for required keys if missing
        if "stakeholder_profile" not in safe_context:
            safe_context["stakeholder_profile"] = f"A {character_id.replace('_', ' ')} with corporate responsibilities."
            
        if "current_context" not in safe_context:
            safe_context["current_context"] = "A standard corporate situation requiring input."
            
        if "relationship_status" not in safe_context:
            # Extract relationship info from stakeholder_relationships if available
            if "stakeholder_relationships" in context and character_id in context["stakeholder_relationships"]:
                rel = context["stakeholder_relationships"][character_id]
                safe_context["relationship_status"] = f"Trust level is {rel.get('trust', 0.5)}. Professional working relationship."
            else:
                safe_context["relationship_status"] = "Standard professional relationship."
                
        if "conversation_history" not in safe_context:
            safe_context["conversation_history"] = "No previous conversation."
        
        # Try character-specific template first
        template_name = f"{character_id}_dialogue"
        if template_name not in self.templates:
            # Fall back to standard dialogue template
            template_name = "standard_dialogue"
            
        return self.create_prompt(template_name, safe_context, max_tokens)
    
    def optimize_for_tokens_and_quality(self, prompt: AIRequest, context: Dict[str, Any]) -> Any:
        """Optimize a prompt for token efficiency while maintaining quality.
        
        Args:
            prompt: The original prompt to optimize.
            context: Context information for optimization guidance.
            
        Returns:
            Any: An object with optimization metrics and the optimized prompt.
        """
        # Extract template name from prompt if possible
        template_name = None
        for name, template in self.templates.items():
            if prompt.content_type == template.content_type:
                if prompt.prompt_template and template.template in prompt.prompt_template:
                    template_name = name
                    break
        
        if not template_name:
            template_name = "standard_scenario"  # Default fallback
            
        # Optimize the context
        optimized_context = self.optimize_context(context, template_name, token_limit=1500)
        
        # Create optimized prompt
        optimized_request = None
        if hasattr(prompt, 'prompt_template') and prompt.prompt_template:
            # Simplify the prompt template while preserving essential instructions
            simplified_template = prompt.prompt_template
            
            # Remove unnecessary verbosity
            simplifications = [
                ("You are generating a corporate scenario for an enterprise simulation focused on technology leadership dynamics.", 
                 "Generate a corporate technology leadership scenario."),
                ("Generate a JSON response with the following structure:", 
                 "Generate JSON with this structure:")
            ]
            
            for original, replacement in simplifications:
                simplified_template = simplified_template.replace(original, replacement)
                
            optimized_request = AIRequest(
                content_type=prompt.content_type,
                prompt_template=simplified_template,
                context=optimized_context,
                max_tokens=prompt.max_tokens
            )
        else:
            optimized_request = prompt  # Fallback if no prompt_template attribute
            
        # Calculate optimization metrics
        class OptimizationResult:
            def __init__(self, original_request, optimized_request):
                self.original_request = original_request
                self.optimized_request = optimized_request
                
                # Calculate optimization ratio
                original_size = len(str(original_request.context))
                optimized_size = len(str(optimized_request.context))
                self.optimization_ratio = 1.0 - (optimized_size / original_size) if original_size > 0 else 0.0
        
        return OptimizationResult(prompt, optimized_request)
    
    def get_available_templates(self) -> Dict[str, PromptTemplate]:
        """Get all available prompt templates.
        
        Returns:
            Dict[str, PromptTemplate]: Dictionary of registered templates.
        """
        return self.templates


class DialogueEngineeringEngine(PromptEngineeringEngine):
    """Specialized prompt engine for character dialogue generation.
    
    Extends the base PromptEngineeringEngine with character-specific features.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize the dialogue engineering engine.
        
        Args:
            config: Configuration for the prompt engineering engine.
        """
        super().__init__(config)
        self._load_character_templates()
    
    def _load_character_templates(self) -> None:
        """Load character dialogue templates with a dynamic, role-driven approach.
        
        This system uses a single universal template with role-specific customizations
        to ensure consistent character identity enforcement across all stakeholder types.
        """
        # Universal dialogue template base that works for ALL stakeholder types
        self.templates["universal_dialogue"] = PromptTemplate(
            name="universal_dialogue",
            content_type=ContentType.DIALOGUE,
            template="""
You are generating dialogue EXCLUSIVELY for the character: {stakeholder_id} - a {stakeholder_role} in a corporate simulation.

IMPORTANT: You MUST generate dialogue ONLY for {stakeholder_id}. Do not generate dialogue for any other character.

{stakeholder_role_uppercase} PROFILE FOR {stakeholder_id}:
{stakeholder_profile}

{context_section}

RELATIONSHIP STATUS:
{relationship_status}

CONVERSATION HISTORY:
{conversation_history}

CHARACTER IDENTITY ENFORCEMENT:
- You are ONLY generating dialogue for {stakeholder_id}
- The speaker field MUST be exactly "{stakeholder_id}"
- The dialogue content MUST reflect this specific {stakeholder_role}'s personality and priorities
- The dialogue MUST be consistent with the provided {stakeholder_role} profile
- NEVER speak as if you are any other character

CONSTRAINTS:
1. The dialogue must authentically reflect {stakeholder_id}'s {stakeholder_role} perspective and priorities
2. {role_specific_focus}
3. {role_specific_language}
4. The tone should reflect the current relationship status with the player
5. NEVER generate dialogue for any character other than {stakeholder_id}

Generate a JSON response with the following {stakeholder_role} dialogue:
```json
{{
  "speaker": "{stakeholder_id}",
  "content": "The {stakeholder_role}'s dialogue with appropriate tone",
  "tone": "{role_specific_tones}",
  "body_language": "Brief description of non-verbal cues",
  {role_specific_fields}
  "relationship_impact": "positive|negative|neutral"
}}
```
            """,
            required_context_keys=["stakeholder_profile", "stakeholder_role", "relationship_status", "conversation_history"],
        )
        
        # Role-specific customization dictionary
        # This allows easy addition of new roles without creating entirely new templates
        self.role_customizations = {
            # CEO role customizations
            "CEO": {
                "role_specific_focus": "Focus on strategic concerns, business outcomes, and leadership expectations",
                "role_specific_language": "Use confident, decisive language appropriate for an executive",
                "role_specific_tones": "authoritative|strategic|concerned|approving|disappointed",
                "role_specific_fields": '"subtext": "The underlying message or expectation being communicated",',
                "context_section": "CURRENT CONTEXT:\n{current_context}"
            },
            # CFO/Finance Director role customizations
            "CFO": {
                "role_specific_focus": "Focus on financial implications, budget constraints, and ROI considerations",
                "role_specific_language": "Use precise, numbers-oriented language with financial terminology",
                "role_specific_tones": "analytical|cautious|decisive|concerned|satisfied",
                "role_specific_fields": '"financial_perspective": "Key financial considerations being emphasized",',
                "context_section": "FINANCIAL CONTEXT:\n{financial_context}"
            },
            "Finance Director": {
                "role_specific_focus": "Focus on financial implications, budget constraints, and ROI considerations",
                "role_specific_language": "Use precise, numbers-oriented language with financial terminology",
                "role_specific_tones": "analytical|cautious|decisive|concerned|satisfied",
                "role_specific_fields": '"financial_perspective": "Key financial considerations being emphasized",',
                "context_section": "FINANCIAL CONTEXT:\n{financial_context}"
            },
            # IT Director role customizations
            "IT Director": {
                "role_specific_focus": "Focus on technical feasibility, resource constraints, and implementation challenges",
                "role_specific_language": "Use specific technical terminology appropriate for the context",
                "role_specific_tones": "technical|pragmatic|concerned|supportive|frustrated",
                "role_specific_fields": '"technical_considerations": "Key technical points being emphasized",',
                "context_section": "TECHNICAL CONTEXT:\n{technical_context}"
            },
            # IT Team role customizations
            "IT Team": {
                "role_specific_focus": "Focus on technical implementation, day-to-day operations, and practical challenges",
                "role_specific_language": "Use technical terminology with a hands-on, practical perspective",
                "role_specific_tones": "technical|practical|concerned|collaborative|stressed",
                "role_specific_fields": '"implementation_details": "Key technical implementation considerations",',
                "context_section": "TECHNICAL CONTEXT:\n{technical_context}"
            },
            # Sales Director role customizations
            "Sales Director": {
                "role_specific_focus": "Focus on customer relationships, revenue targets, and market opportunities",
                "role_specific_language": "Use persuasive, results-oriented language with customer-focused terminology",
                "role_specific_tones": "confident|enthusiastic|concerned|persuasive|urgent",
                "role_specific_fields": '"market_perspective": "Key customer and sales considerations",',
                "context_section": "MARKET CONTEXT:\n{market_context}"
            },
            # Admin Team role customizations
            "Admin Team": {
                "role_specific_focus": "Focus on operational details, scheduling, and organizational support",
                "role_specific_language": "Use practical, process-oriented language focused on execution",
                "role_specific_tones": "helpful|practical|organized|concerned|supportive",
                "role_specific_fields": '"operational_details": "Key operational considerations being emphasized",',
                "context_section": "OPERATIONAL CONTEXT:\n{operational_context}"
            },
            # Security Director role customizations
            "Security Director": {
                "role_specific_focus": "Focus on security risks, compliance requirements, and protection measures",
                "role_specific_language": "Use security-focused terminology with emphasis on risk management",
                "role_specific_tones": "cautious|authoritative|concerned|firm|urgent",
                "role_specific_fields": '"security_considerations": "Key security and compliance factors",',
                "context_section": "SECURITY CONTEXT:\n{security_context}"
            },
            # HR Director/Manager role customizations
            "HR Director": {
                "role_specific_focus": "Focus on personnel issues, workplace culture, and employee wellbeing",
                "role_specific_language": "Use people-oriented language with emphasis on organizational health",
                "role_specific_tones": "supportive|diplomatic|concerned|empathetic|firm",
                "role_specific_fields": '"personnel_considerations": "Key workforce and culture factors",',
                "context_section": "HR CONTEXT:\n{hr_context}"
            },
            "HR Manager": {
                "role_specific_focus": "Focus on personnel issues, workplace culture, and employee wellbeing",
                "role_specific_language": "Use people-oriented language with emphasis on organizational health",
                "role_specific_tones": "supportive|diplomatic|concerned|empathetic|firm",
                "role_specific_fields": '"personnel_considerations": "Key workforce and culture factors",',
                "context_section": "HR CONTEXT:\n{hr_context}"
            },
            # Marketing Team role customizations
            "Marketing Team": {
                "role_specific_focus": "Focus on brand perception, market positioning, and communication strategy",
                "role_specific_language": "Use creative, audience-focused language with emphasis on messaging",
                "role_specific_tones": "creative|enthusiastic|concerned|strategic|persuasive",
                "role_specific_fields": '"brand_considerations": "Key marketing and perception factors",',
                "context_section": "MARKETING CONTEXT:\n{marketing_context}"
            },
            # Senior Developer role customizations
            "Senior Developer": {
                "role_specific_focus": "Focus on code quality, technical architecture, and development practices",
                "role_specific_language": "Use detailed technical terminology with emphasis on implementation",
                "role_specific_tones": "technical|analytical|concerned|direct|pragmatic",
                "role_specific_fields": '"technical_debt_considerations": "Key development and architecture factors",',
                "context_section": "DEVELOPMENT CONTEXT:\n{development_context}"
            },
            # Default role customizations (used for any undefined role)
            "DEFAULT": {
                "role_specific_focus": "Focus on areas relevant to your role and responsibilities",
                "role_specific_language": "Use language appropriate for your professional position",
                "role_specific_tones": "professional|concerned|supportive|neutral|frustrated",
                "role_specific_fields": '"key_considerations": "Important points being emphasized",',
                "context_section": "CURRENT CONTEXT:\n{current_context}"
            }
        }
    
    def create_character_dialogue_prompt(
        self,
        character_id: str,
        context: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> AIRequest:
        """Create a character-specific dialogue prompt using the dynamic template system.
        
        Args:
            character_id: ID of the character.
            context: Context for the dialogue.
            max_tokens: Maximum tokens for the response.
            
        Returns:
            AIRequest: The assembled AI request.
        """
        # Add the character ID to context if not present
        if "stakeholder_id" not in context:
            context["stakeholder_id"] = character_id
        
        # Ensure stakeholder_role is present
        if "stakeholder_role" not in context:
            # Try to extract role from profile or use a default
            profile = context.get("stakeholder_profile", "")
            role = self._extract_role_from_profile(profile, character_id)
            context["stakeholder_role"] = role
            logger.info(f"Extracted role '{role}' for character {character_id}")
        
        # Add uppercase version of role for template formatting
        context["stakeholder_role_uppercase"] = context["stakeholder_role"].upper()
        
        # Apply role-specific customizations
        role = context["stakeholder_role"]
        customizations = self.role_customizations.get(role, self.role_customizations["DEFAULT"])
        
        # Add customizations to context
        for key, value in customizations.items():
            context[key] = value
        
        logger.debug(f"Using role '{role}' customizations for character {character_id}")
        
        # Use the universal dialogue template
        return self.create_prompt("universal_dialogue", context, max_tokens)
    
    def _extract_role_from_profile(self, profile: str, character_id: str) -> str:
        """Extract the character's role from their profile.
        
        Args:
            profile: The character's profile text.
            character_id: The character's ID as fallback for role extraction.
            
        Returns:
            str: The extracted role or a default.
        """
        # Common corporate roles to check for
        common_roles = [
            "CEO", "CFO", "CTO", "COO", "CIO", "CHRO", "CMO", 
            "IT Director", "Finance Director", "HR Director", "Marketing Director",
            "Project Manager", "Team Lead", "Admin Team", "Board Member",
            "Department Head", "Supervisor", "Manager"
        ]
        
        # Check if any common role is in the profile
        for role in common_roles:
            if role in profile:
                return role
        
        # Check if role is in the character_id
        for role in common_roles:
            if role.lower().replace(" ", "_") in character_id.lower():
                return role
        
        # Special case handling
        if "admin" in character_id.lower():
            return "Admin Team"
        if "it" in character_id.lower():
            return "IT Director"
        if "finance" in character_id.lower() or "cfo" in character_id.lower():
            return "CFO"
        if "ceo" in character_id.lower():
            return "CEO"
        
        # Default fallback
        logger.warning(f"Could not determine role for {character_id}, using DEFAULT")
        return "Corporate Staff"
