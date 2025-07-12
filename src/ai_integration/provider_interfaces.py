"""Provider interfaces for the AI integration layer.

This module defines the abstract base classes and implementations for different
AI service providers supported by the Corporate Dynamics Simulator.

Key components:
- BaseAIProvider: Abstract base class that all providers must implement
- OpenAIProvider: Provider implementation for OpenAI services
- AnthropicProvider: Provider implementation for Anthropic services
- GeminiProvider: Provider implementation for Google Gemini services
- TemplateProvider: Fallback provider using pre-defined templates
"""
from __future__ import annotations

import abc
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .cost_optimization_manager import CostOptimizationManager

from .data_models import (
    AIConfig,
    AIRequest,
    AIResponse,
    ContentType,
    ProviderType,
    ScenarioResponse,
    DialogueResponse,
)

logger = logging.getLogger(__name__)


class BaseAIProvider(abc.ABC):
    """Abstract base class for AI service providers.
    
    All provider implementations must inherit from this class.
    """
    
    def __init__(self, config: AIConfig, cost_manager: "CostOptimizationManager"):
        """Initialize the provider with configuration.
        
        Args:
            config: Configuration for the AI service.
            cost_manager: The cost optimization manager instance.
        """
        self.config = config
        self.cost_manager = cost_manager
        self.provider_type: ProviderType = ProviderType.TEMPLATE
        
    @abc.abstractmethod
    async def generate_content(self, request: AIRequest) -> AIResponse:
        """Generate content based on the request.
        
        Args:
            request: The AI request containing prompt and context.
            
        Returns:
            AIResponse: The generated AI response.
        """
        raise NotImplementedError
    
    def estimate_cost(self, request: AIRequest) -> float:
        """Estimate the cost of the request based on token count.
        
        Args:
            request: The AI request.
            
        Returns:
            float: Estimated cost in dollars.
        """
        # Approximate token count from prompt and max tokens
        prompt_tokens = len(request.prompt_template.split()) * 1.3
        response_tokens = request.max_tokens
        
        total_tokens = prompt_tokens + response_tokens
        cost_per_1k = self.config.cost_per_1k_tokens.get(
            self.provider_type.value, 0.001
        )
        
        return (total_tokens / 1000) * cost_per_1k
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string.
        
        This is a simple approximation. Production systems would use
        provider-specific tokenizers.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            int: Approximate token count.
        """
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def create_request_id(self, request: AIRequest) -> str:
        """Create a unique ID for a request for tracking and caching.
        
        Args:
            request: The AI request.
            
        Returns:
            str: A unique request ID.
        """
        # Create a deterministic hash from the prompt and context
        content = f"{request.prompt_template}:{json.dumps(request.context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_response(
        self, content: Any, request: AIRequest, tokens_used: int, cost: float
    ) -> AIResponse:
        """Create an AIResponse from generation results.
        
        Args:
            content: The generated content.
            request: The original request.
            tokens_used: Number of tokens used.
            cost: Cost in dollars.
            
        Returns:
            AIResponse: The formatted response.
        """
        request_id = request.request_id or self.create_request_id(request)
        
        response_cls = ScenarioResponse if request.content_type == ContentType.SCENARIO else DialogueResponse
        
        return response_cls(
            content_type=request.content_type,
            content=content,
            provider=self.provider_type,
            tokens_used=tokens_used,
            cost=cost,
            request_id=request_id,
        )


class OpenAIProvider(BaseAIProvider):
    """OpenAI service provider implementation."""
    
    def __init__(self, config: AIConfig, cost_manager: "CostOptimizationManager"):
        """Initialize the OpenAI provider.
        
        Args:
            config: AI service configuration.
        """
        super().__init__(config, cost_manager)
        self.provider_type = ProviderType.OPENAI
        self.api_key = config.openai_api_key
        
        # Import conditionally to avoid hard dependency
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.available = bool(self.api_key)
        except ImportError:
            logger.warning("OpenAI package not installed. OpenAI provider unavailable.")
            self.available = False
    
    async def generate_content(self, request: AIRequest) -> AIResponse:
        """Generate content using OpenAI API.
        
        Args:
            request: The AI request.
            
        Returns:
            AIResponse: The generated response.
            
        Raises:
            RuntimeError: If OpenAI client is not available.
        """
        if not self.available:
            raise RuntimeError("OpenAI provider not available")
        
        start_time = time.perf_counter()
        
        try:
            # Prepare the prompt with context
            messages = [{"role": "user", "content": request.prompt_template}]
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a free tier model
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Process the content based on content type
            if request.content_type == ContentType.SCENARIO:
                # Attempt to parse JSON response for scenario
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse OpenAI scenario response as JSON")
                    content = {"description": content, "error": "Invalid JSON format"}
            
            # Calculate tokens and cost
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * self.config.cost_per_1k_tokens.get("openai", 0.002)
            
            # Create and return the response
            ai_response = self._create_response(content, request, tokens_used, cost)
            
            # Log performance metrics
            duration = time.perf_counter() - start_time
            logger.info(
                f"OpenAI request completed in {duration:.2f}s, {tokens_used} tokens, ${cost:.6f}"
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude service provider implementation."""
    
    def __init__(self, config: AIConfig, cost_manager: "CostOptimizationManager"):
        """Initialize the Anthropic provider.
        
        Args:
            config: AI service configuration.
        """
        super().__init__(config, cost_manager)
        self.provider_type = ProviderType.ANTHROPIC
        self.api_key = config.anthropic_api_key
        
        # Import conditionally to avoid hard dependency
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.available = bool(self.api_key)
        except ImportError:
            logger.warning("Anthropic package not installed. Anthropic provider unavailable.")
            self.available = False
    
    async def generate_content(self, request: AIRequest) -> AIResponse:
        """Generate content using Anthropic API.
        
        Args:
            request: The AI request.
            
        Returns:
            AIResponse: The generated response.
            
        Raises:
            RuntimeError: If Anthropic client is not available.
        """
        if not self.available:
            raise RuntimeError("Anthropic provider not available")
        
        start_time = time.perf_counter()
        
        try:
            # Import model configuration for current free-tier model
            from src.ai_integration.cost.provider_models import ModelType
            
            # Call the Anthropic API with latest free-tier model
            response = self.client.messages.create(
                model=ModelType.CLAUDE_35_HAIKU.value,  # Latest free-tier Claude model
                max_tokens=request.max_tokens,
                messages=[
                    {"role": "user", "content": request.prompt_template}
                ],
                temperature=request.temperature,
            )
            
            # Parse the response
            content = response.content[0].text
            
            # Process the content based on content type
            if request.content_type == ContentType.SCENARIO:
                # Attempt to parse JSON response for scenario
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Anthropic scenario response as JSON")
                    content = {"description": content, "error": "Invalid JSON format"}
            
            # Approximate token count since Anthropic doesn't return it directly
            tokens_used = self.count_tokens(request.prompt_template) + self.count_tokens(content)
            cost = (tokens_used / 1000) * self.config.cost_per_1k_tokens.get("anthropic", 0.0015)
            
            # Create and return the response
            ai_response = self._create_response(content, request, tokens_used, cost)
            
            # Log performance metrics
            duration = time.perf_counter() - start_time
            logger.info(
                f"Anthropic request completed in {duration:.2f}s, ~{tokens_used} tokens, ${cost:.6f}"
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class GeminiProvider(BaseAIProvider):
    """Google Gemini service provider implementation."""
    
    def __init__(self, config: AIConfig, cost_manager: "CostOptimizationManager"):
        """Initialize the Gemini provider.
        
        Args:
            config: AI service configuration.
        """
        super().__init__(config, cost_manager)
        self.provider_type = ProviderType.GEMINI
        self.api_key = config.gemini_api_key
        
        logger.info("Initializing Gemini provider")
        try:
            import google.generativeai as genai
            logger.debug("Successfully imported google.generativeai package")
            genai.configure(api_key=self.api_key)
            logger.debug("Configured Gemini API with provided key")
            self.client = genai
            self.available = bool(self.api_key)
            
            # Import model types for validation
            from src.ai_integration.cost.provider_models import ModelType
            self.model_types = {
                ModelType.GEMINI_20_FLASH.value,
                ModelType.GEMINI_15_FLASH_002.value,
                ModelType.GEMINI_15_FLASH.value,
                ModelType.GEMINI_15_FLASH_8B.value
            }
            
            # Log the supported models for debugging
            logger.info(f"Gemini provider initialized with supported models: {self.model_types}")
            logger.info(f"Gemini provider initialization succeeded with {len(self.model_types)} supported models")
        except ImportError:
            logger.warning("Google Generative AI package not installed. Gemini provider unavailable.")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}", exc_info=True)
            self.available = False
    
    async def generate_content(self, request: AIRequest) -> AIResponse:
        """Generate content using Google Gemini API.
        
        Args:
            request: The AI request.
            
        Returns:
            AIResponse: The generated response.
            
        Raises:
            RuntimeError: If Gemini client is not available.
        """
        # Extra debugging information
        logger.debug(f"GeminiProvider.generate_content called with content_type: {request.content_type}")
        logger.debug(f"Gemini provider status - available: {self.available}, API key: {'Present' if self.api_key else 'Missing'}")
        
        if not self.available:
            logger.error("Gemini provider not available - returning early")
            raise RuntimeError("Gemini provider not available")
        
        if not self.api_key:
            logger.error("Gemini API key missing but provider marked as available")
            raise RuntimeError("Missing Gemini API key")
        
        start_time = time.perf_counter()
        
        try:
            # Import model configuration for current free-tier model
            from src.ai_integration.cost.provider_models import ModelType
            
            # Determine which model to use based on the provider_type or model field
            selected_model = None
            actual_model = None
            fallback_reason = None
            
            # First check if provider_type is a ModelType enum
            if hasattr(request, 'provider_type') and isinstance(request.provider_type, ModelType):
                selected_model = request.provider_type.value
                logger.info(f"Selected provider model type: {request.provider_type}")
            # Then check if model field is set (string value)
            elif hasattr(request, 'model') and request.model:
                selected_model = request.model
                logger.info(f"Selected model from request.model: {selected_model}")
            
            # Define model fallback sequence for quota errors
            gemini_models = [
                ModelType.GEMINI_20_FLASH.value,
                ModelType.GEMINI_15_FLASH_002.value,
                ModelType.GEMINI_15_FLASH.value,
                ModelType.GEMINI_15_FLASH_8B.value
            ]
            
            # If selected model is valid, try it first, otherwise start with first model in sequence
            if selected_model and selected_model in self.model_types:
                # Move the selected model to the front of the list if it's in our supported models
                if selected_model in gemini_models:
                    gemini_models.remove(selected_model)
                    gemini_models.insert(0, selected_model)
                else:
                    # If it's supported but not in our standard list, add it
                    gemini_models.insert(0, selected_model)
            
            # Try models in sequence until one works or all fail
            last_error = None
            response = None
            
            for model_name in gemini_models:
                actual_model = model_name
                logger.info(f"Attempting to use Gemini model: {actual_model}")
                
                try:
                    # Create a Gemini model using the current model
                    logger.debug(f"Creating Gemini model instance for {actual_model}")
                    model = self.client.GenerativeModel(actual_model)
                    logger.debug("Gemini model instance created successfully")
                    
                    # Call the Gemini API
                    logger.debug(f"Calling Gemini API generate_content with model {actual_model}")
                    response = model.generate_content(request.prompt_template)
                    logger.debug("Gemini API generate_content call returned successfully")
                    
                    # If we get here, the model worked
                    if selected_model and selected_model != actual_model:
                        fallback_reason = f"Original model {selected_model} unavailable or quota exceeded"
                        logger.info(f"Successfully used alternative Gemini model: {actual_model}")
                    
                    # Record usage for cost tracking
                    logger.info(f"Recording Gemini usage for model: {actual_model}")
                    
                    # Include fallback information in the logs if applicable
                    if selected_model and selected_model != actual_model:
                        logger.warning(f"Model fallback occurred: {selected_model} → {actual_model}. Reason: {fallback_reason}")
                    
                    # Break out of the loop since we have a successful response
                    break
                    
                except Exception as e:
                    last_error = e
                    if "quota" in str(e).lower() or "429" in str(e):
                        logger.warning(f"Quota exceeded for Gemini model {actual_model}, trying next model")
                    else:
                        logger.warning(f"Error with Gemini model {actual_model}: {str(e)}, trying next model")
            
            # If we've tried all models and all failed, raise the last error
            if last_error is not None and response is None:
                logger.error(f"All Gemini models failed. Last error: {str(last_error)}", exc_info=True)
                raise RuntimeError(f"Gemini API call failed: {str(last_error)}")
            
            try:
                # Ensure we're using the actual Gemini model name, not template-default
                if actual_model == "template-default":
                    logger.warning(f"Detected template-default model name in GeminiProvider, using {request.model} instead")
                    usage_model = request.model
                else:
                    usage_model = actual_model
                    
                self.cost_manager.record_gemini_usage(
                    model_name=usage_model,
                    prompt_text=request.prompt_template,
                    response_text=response.text,
                )
                logger.info(f"Recorded Gemini usage with model: {usage_model}")
            except Exception as e:
                logger.error(f"Failed to record Gemini usage: {e}")
            
            try:
                # Handle potential generator responses in newer API versions
                logger.debug(f"Response type: {type(response).__name__}")
                if hasattr(response, '__iter__') and not hasattr(response, 'text'):
                    logger.debug("Detected generator response, converting to list")
                    # If response is a generator, convert to list and take first item
                    response_list = list(response)
                    logger.debug(f"Converted generator to list of length {len(response_list)}")
                    
                    if response_list:
                        response = response_list[0]
                        logger.debug("Successfully extracted first item from response list")
                    else:
                        logger.error("Empty response list after converting generator")
                        raise RuntimeError("Empty response from Gemini API")
                else:
                    logger.debug("Response is not a generator, using as-is")
                
                # Parse the response
                logger.debug("Extracting text from response")
                content = response.text
                logger.debug(f"Successfully extracted content of length {len(content)}")
            except Exception as e:
                logger.error(f"Error processing Gemini response: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to process Gemini response: {str(e)}")
            
            try:
                # Process the content based on content type
                if request.content_type == ContentType.SCENARIO:
                    logger.debug("Attempting to parse JSON response for scenario")
                    # Attempt to parse JSON response for scenario
                    try:
                        content = json.loads(content)
                        logger.debug("Successfully parsed JSON response")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Gemini scenario response as JSON")
                        content = {"description": content, "error": "Invalid JSON format"}
                elif request.content_type == ContentType.DIALOGUE:
                    logger.debug("Formatting dialogue response as dictionary")
                    # Get character ID from request context
                    character_id = request.context.get("character_id", 
                                                request.context.get("stakeholder_id", "AI"))
                    
                    # Check if content contains Markdown code blocks with JSON
                    import re
                    json_code_block_pattern = r'```(?:json)?\s*\n([\s\S]*?)\n\s*```'
                    code_block_match = re.search(json_code_block_pattern, content)
                    
                    if code_block_match:
                        logger.debug("Found JSON code block in response, extracting")
                        json_str = code_block_match.group(1).strip()
                        try:
                            # Try to parse extracted JSON
                            parsed_content = json.loads(json_str)
                            logger.debug("Successfully parsed extracted JSON from code block")
                            content = parsed_content
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse extracted JSON from code block: {e}")
                            # Fall through to other parsing methods
                    
                    # If we don't have a dictionary yet, try direct JSON parsing
                    if not isinstance(content, dict):
                        try:
                            parsed_content = json.loads(content)
                            logger.debug("Successfully parsed dialogue JSON response")
                            content = parsed_content
                        except json.JSONDecodeError:
                            logger.debug("Dialogue response is plain text, formatting as dictionary")
                            # Format plain text dialogue with expected fields
                            content = {
                                "speaker": character_id,
                                "content": content.strip(),
                                "tone": request.context.get("tone", "professional"),
                                "relationship_impact": "neutral",
                            }
                    
                    # Validate character identity and content consistency
                    if isinstance(content, dict):
                        character_mismatch = False
                        mismatch_severity = 0
                        mismatch_reasons = []
                        
                        # Check speaker field
                        if "speaker" in content and content["speaker"] != character_id:
                            character_mismatch = True
                            mismatch_severity += 2
                            mismatch_reasons.append(f"Speaker mismatch: got '{content['speaker']}', expected '{character_id}'")
                            logger.warning(f"Speaker mismatch in response: got {content['speaker']}, expected {character_id}.")
                        
                        # Check content for character voice consistency
                        if "content" in content and request.context.get("stakeholder_profile"):
                            profile = request.context.get("stakeholder_profile", "")
                            dialogue = content.get("content", "")
                            
                            # Check for role/title consistency
                            if "CEO" in profile and not any(term in dialogue.lower() for term in ["strategy", "business", "company", "leadership", "board", "executive"]):
                                character_mismatch = True
                                mismatch_severity += 1
                                mismatch_reasons.append("Content lacks CEO perspective and priorities")
                            
                            if "IT Director" in profile and not any(term in dialogue.lower() for term in ["technical", "system", "security", "implementation", "infrastructure", "technology"]):
                                character_mismatch = True
                                mismatch_severity += 1
                                mismatch_reasons.append("Content lacks IT Director technical focus")
                            
                            # Check for wrong character references
                            if re.search(r'\b(as|I am|speaking as)\s+(?!{character_id})\w+', dialogue, re.IGNORECASE):
                                character_mismatch = True
                                mismatch_severity += 2
                                mismatch_reasons.append("Content contains references to being a different character")
                        
                        # Apply corrections based on severity
                        if character_mismatch:
                            if mismatch_severity >= 3:
                                # Severe mismatch - log the issue and flag for regeneration
                                logger.error(f"Severe character identity mismatch detected: {', '.join(mismatch_reasons)}")
                                logger.error("Content does not match character identity, should regenerate with more constraints")
                                # Add regeneration flag to the response
                                content["_needs_regeneration"] = True
                                content["_regeneration_reason"] = mismatch_reasons
                            else:
                                # Minor mismatch - correct and log
                                logger.warning(f"Character identity mismatch detected: {', '.join(mismatch_reasons)}")
                                logger.info("Applying character identity corrections")
                                # Ensure speaker is correct
                                content["speaker"] = character_id
                                # Log the correction
                                content["_corrected"] = True
                                content["_correction_reasons"] = mismatch_reasons
                    
                    # Make sure required fields exist
                    if isinstance(content, dict):
                        if "content" not in content and "text" in content:
                            content["content"] = content["text"]
                        # Ensure all required fields are present
                        for required_field in ["content", "tone", "relationship_impact"]:
                            if required_field not in content:
                                if required_field == "content":
                                    content[required_field] = "I understand your perspective on this matter."
                                elif required_field == "tone":
                                    content[required_field] = "professional"
                                elif required_field == "relationship_impact":
                                    content[required_field] = "neutral"
                    
                    logger.debug(f"Dialogue formatted as dictionary: {content}")
                
                # Approximate token count since Gemini doesn't return it directly
                logger.debug("Calculating token count and cost")
                tokens_used = self.count_tokens(request.prompt_template) + self.count_tokens(content)
                cost = (tokens_used / 1000) * self.config.cost_per_1k_tokens.get("gemini", 0.0005)
                
                # Create and return the response
                logger.debug("Creating AI response object")
                ai_response = self._create_response(content, request, tokens_used, cost)
                
                # Log performance metrics
                duration = time.perf_counter() - start_time
                logger.info(
                    f"Gemini request completed in {duration:.2f}s, ~{tokens_used} tokens, ${cost:.6f}"
                )
                
                return ai_response
            except Exception as e:
                logger.error(f"Error in content processing: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed in Gemini content processing: {str(e)}")
                
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Gemini API error after {duration:.2f}s: {str(e)}")
            logger.debug("Gemini API error details:", exc_info=True)
            raise RuntimeError(f"Gemini provider failed: {str(e)}")


class TemplateProvider(BaseAIProvider):
    """Fallback template-based provider that doesn't use external APIs.
    
    This provider is used when external APIs are unavailable or budget constraints
    require a fallback solution.
    """
    
    def __init__(self, config: AIConfig, cost_manager: "CostOptimizationManager"):
        """Initialize the template provider.
        
        Args:
            config: AI service configuration.
        """
        super().__init__(config, cost_manager)
        self.provider_type = ProviderType.TEMPLATE
        self.available = True
        
        # Template repository
        self.scenario_templates = {
            "crisis": [
                {
                    "description": "Unexpected system outage affecting critical business operations. IT team is scrambling to identify root cause while leadership demands immediate restoration of service.",
                    "available_actions": [
                        "assemble_response_team",
                        "implement_backup_system",
                        "communicate_with_stakeholders",
                    ]
                },
                {
                    "description": "Data security breach detected in customer information systems. Legal team advising on compliance requirements while technical teams assess the damage.",
                    "available_actions": [
                        "forensic_investigation",
                        "customer_notification",
                        "security_remediation",
                    ]
                },
            ],
            "project": [
                {
                    "description": "Major software implementation project falling behind schedule. Vendor blaming requirements changes while internal team cites poor documentation.",
                    "available_actions": [
                        "renegotiate_timeline",
                        "reduce_project_scope",
                        "add_resources",
                    ]
                },
                {
                    "description": "Digital transformation initiative meeting resistance from key departments. Leadership wants progress while teams struggle with adoption.",
                    "available_actions": [
                        "stakeholder_engagement_sessions",
                        "revise_change_management",
                        "adjust_implementation_pace",
                    ]
                },
            ],
        }
        
        self.dialogue_templates = {
            "ceo": [
                "I need this project back on track immediately. What specific actions are you taking to resolve these delays?",
                "The board is concerned about our progress. Can you give me a clear update I can share with them?",
                "We're investing significant resources here - I need to see tangible results soon.",
            ],
            "it_director": [
                "My team is stretched thin across multiple priorities. We need to discuss realistic timelines.",
                "The technical requirements keep changing which is causing these delays. We need a stable scope.",
                "I've identified several risks that need immediate attention before we proceed further.",
            ],
            "facilities_manager": [
                "The infrastructure updates needed for this project require more lead time than we've been given.",
                "I've coordinated with vendors, but supply chain issues are affecting our timeline.",
                "We need to align our physical space constraints with the technical requirements.",
            ],
            "admin_team": [
                "We've been processing the increased workload, but we need more clarity on priorities.",
                "The new procedures are causing confusion among staff. Additional training would help.",
                "We've compiled feedback from users that should inform the next phase of implementation.",
            ],
        }
    
    async def generate_content(self, request: AIRequest) -> AIResponse:
        """Generate content using templates.
        
        Args:
            request: The AI request.
            
        Returns:
            AIResponse: The generated response.
        """
        start_time = time.perf_counter()
        
        # Generate content based on content type
        if request.content_type == ContentType.SCENARIO:
            content = self._generate_scenario(request)
        elif request.content_type == ContentType.DIALOGUE:
            content = self._generate_dialogue(request)
        else:
            # Default fallback
            content = {"description": "Template-generated content for " + request.content_type.value}
        
        # Calculate token usage (minimal for templates)
        tokens_used = 100  # Nominal value
        cost = 0.0  # Templates are free
        
        # Create and return the response
        ai_response = self._create_response(content, request, tokens_used, cost)
        
        # Log performance
        duration = time.perf_counter() - start_time
        logger.info(f"Template generation completed in {duration:.2f}s, cost $0.00")
        
        return ai_response
    
    def _generate_scenario(self, request: AIRequest) -> Dict[str, Any]:
        """Generate a scenario from templates.
        
        Args:
            request: The AI request.
            
        Returns:
            Dict[str, Any]: Generated scenario content.
        """
        # Extract scenario type from context or default to crisis
        scenario_type = request.context.get("scenario_type", "crisis")
        
        # Get templates for this type
        templates = self.scenario_templates.get(scenario_type, self.scenario_templates["crisis"])
        
        # Create a deterministic but pseudo-random selection based on request
        request_hash = self.create_request_id(request)
        template_index = int(request_hash, 16) % len(templates)
        
        # Get the selected template
        template = templates[template_index]
        
        # Ensure required keys exist in the template with fallbacks
        if "description" not in template:
            template["description"] = "Corporate scenario requiring stakeholder management and strategic decision making."
            
        # Return the template with ensured keys
        return template
    
    def _generate_dialogue(self, request: AIRequest) -> Dict[str, Any]:
        """Generate dialogue from templates.
        
        Args:
            request: The AI request.
            
        Returns:
            Dict[str, Any]: Generated dialogue content.
        """
        # Extract character ID from context, also checking stakeholder_id as fallback
        character_id = request.context.get("character_id", 
                                      request.context.get("stakeholder_id", "it_director"))
        
        # Get templates for this character
        templates = self.dialogue_templates.get(character_id, self.dialogue_templates["it_director"])
        
        # Create a deterministic but pseudo-random selection based on request
        request_hash = self.create_request_id(request)
        template_index = int(request_hash, 16) % len(templates)
        
        # Get the dialogue text
        dialogue_text = templates[template_index]
        
        # Return formatted dialogue
        return {
            "speaker": character_id,
            "content": dialogue_text,
            "tone": request.context.get("tone", "professional"),
            "relationship_impact": "neutral",
        }
