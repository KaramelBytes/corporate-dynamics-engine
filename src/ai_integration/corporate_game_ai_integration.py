"""Corporate Game AI Integration for Corporate Dynamics Simulator.

This module integrates all AI components into a cohesive interface for the game.
It orchestrates prompt engineering, cost optimization, quality assurance, and
service orchestration to provide a unified API for AI-driven features.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from .cost_optimization_manager import CostOptimizationManager
from .data_models import (
    AIConfig,
    AIRequest,
    AIResponse,
    ContentType,
    DialogueResponse,
    ProviderType,
    QualityResult,
    ScenarioResponse,
    ValidationContext,
)
from .prompt_engineering_engine import DialogueEngineeringEngine, PromptEngineeringEngine
from .quality_assurance_engine import QualityAssuranceEngine
from .service_orchestrator import AIServiceOrchestrator
from .provider_interfaces import TemplateProvider
from .cost.model_configs import get_preferred_models, MODEL_CONFIGS, PROVIDER_CONFIGS
from .cost.provider_models import ModelType

logger = logging.getLogger(__name__)


class CorporateGameAIIntegration:
    """Main integration class for all AI components in the game.
    
    Provides a unified interface for generating scenarios, dialogue, and other
    AI-driven content for the Corporate Dynamics Simulator.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the AI integration.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = self._load_config(config_path)
        
        # Initialize all component systems
        self.prompt_engine = PromptEngineeringEngine(self.config)
        self.dialogue_engine = DialogueEngineeringEngine(self.config)
        self.cost_manager = CostOptimizationManager(self.config)
        self.quality_engine = QualityAssuranceEngine(self.config)
        
        # Initialize the orchestrator instead of providers directly
        self.orchestrator = AIServiceOrchestrator(self.config)
        
        # Only use template provider as a final fallback, never as a first choice
        self.fallback_provider = TemplateProvider(self.config, self.cost_manager)
        
        logger.info("Corporate Game AI Integration initialized")
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> AIConfig:
        """Load configuration from file or environment variables.
        
        This method now properly loads from .env files and environment variables,
        ensuring API keys are available for provider initialization.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            AIConfig: Configuration for AI integration.
        """
        import os
        from dotenv import load_dotenv
        from pathlib import Path
        
        # 🔥 CRITICAL FIX: Load .env file before creating AIConfig
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            logger.info(f"Loaded .env file for AI integration: {env_path}")
        
        # Create AIConfig with environment variables
        config = AIConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            monthly_budget=float(os.environ.get("MONTHLY_BUDGET", "10.0")),
            enable_caching=os.environ.get("ENABLE_CACHING", "true").lower() in ["true", "1", "yes"],
        )
        
        # Debug logging to verify what was loaded
        api_keys_loaded = []
        if config.openai_api_key:
            api_keys_loaded.append("OpenAI")
        if config.anthropic_api_key:
            api_keys_loaded.append("Anthropic")
        if config.gemini_api_key:
            api_keys_loaded.append("Gemini")
            
        if api_keys_loaded:
            logger.info(f"AIConfig loaded with API keys: {', '.join(api_keys_loaded)}")
        else:
            logger.warning("AIConfig created with no API keys - providers will be unavailable")
        
        if config_path:
            try:
                # Future: load additional config from file if needed
                logger.info(f"Config path provided: {config_path}, using environment + defaults")
            except Exception as e:
                logger.error(f"Error loading config file: {e}, using environment + defaults")
        
        return config
    
    # Provider initialization is now handled by the orchestrator
    
    async def generate_scenario(
        self,
        context: Dict[str, Any],
        scenario_type: str = "standard_scenario",
        provider_preference: Optional[ProviderType] = None,
        max_cost: Optional[float] = None,
    ) -> ScenarioResponse:
        """Generate a corporate scenario based on context.
        
        Args:
            context: Context information for scenario generation.
            scenario_type: Type of scenario template to use.
            provider_preference: Preferred AI provider.
            max_cost: Maximum cost for this request.
            
        Returns:
            ScenarioResponse: The generated scenario.
        """
        logger.info(f"Generating {scenario_type} scenario")
        
        # Step 1: Create the prompt using the template
        request = self.prompt_engine.create_prompt(scenario_type, context)
        
        # Step 2: Set provider preference if specified
        if provider_preference:
            request.provider_preference = provider_preference
        
        # Step 3: Set maximum cost if specified
        if max_cost:
            request.max_cost = max_cost
        
        # Step 4: Optimize the request for cost efficiency
        optimization_result = self.cost_manager.optimize_request(request, max_cost)
        optimized_request = optimization_result.optimized_request or request
        
        # Step 5: Select a provider and generate content
        provider = self._select_provider(optimized_request)
        
        # Step 5.1: Try with selected provider
        # If first provider fails, try other available providers before falling back to template
        available_providers = self.orchestrator.get_available_providers()
        tried_providers = set()  # Track which providers we've already tried
        
        # First try with the selected provider
        if provider.provider_type != ProviderType.TEMPLATE:
            try:
                # Generate the response
                response = await provider.generate_content(optimized_request)
                
                # Step 6: Record usage for budget tracking
                self.cost_manager.record_usage(response)
                
                # Step 7: Validate and enhance the response
                validation_context = ValidationContext(
                    content_type=ContentType.SCENARIO,
                    original_request=request,
                    corporate_profile=context.get("corporate_profile", {}),
                    stakeholder_profiles=context.get("stakeholder_profiles", {}),
                )
                
                quality_result = self.quality_engine.validate_and_enhance(response, validation_context)
                
                # Step 8: Return the enhanced response or original if no enhancement
                final_response = quality_result.enhanced_response or response
                
                return final_response
                
            except Exception as e:
                tried_providers.add(provider.provider_type)
                logger.warning(f"Provider {provider.provider_type} failed: {e}. Trying other providers...")
                
        # Step 5.2: Try other available providers if the first one failed
        for provider_type in available_providers:
            if provider_type in tried_providers or provider_type == ProviderType.TEMPLATE:
                continue  # Skip already tried providers and template provider
                
            try:
                backup_provider = self.orchestrator.select_provider(provider_type)
                logger.info(f"Trying backup provider: {provider_type}")
                
                response = await backup_provider.generate_content(optimized_request)
                self.cost_manager.record_usage(response)
                
                validation_context = ValidationContext(
                    content_type=ContentType.SCENARIO,
                    original_request=request,
                    corporate_profile=context.get("corporate_profile", {}),
                    stakeholder_profiles=context.get("stakeholder_profiles", {}),
                )
                
                quality_result = self.quality_engine.validate_and_enhance(response, validation_context)
                final_response = quality_result.enhanced_response or response
                
                return final_response
                
            except Exception as e:
                tried_providers.add(provider_type)
                logger.warning(f"Backup provider {provider_type} failed: {e}")
        
        # Step 5.3: Only fall back to template as last resort after all real providers have failed
        logger.warning("All AI providers failed. Falling back to template provider.")
        return await self.fallback_provider.generate_content(request)
    
    async def generate_dialogue(
        self,
        character_id: str,
        context: Dict[str, Any],
        provider_preference: Optional[ProviderType] = None,
        max_cost: Optional[float] = None,
    ) -> DialogueResponse:
        """Generate dialogue for a character based on context.
        
        Args:
            character_id: ID of the character to generate dialogue for.
            context: Context information for dialogue generation.
            provider_preference: Preferred AI provider.
            max_cost: Maximum cost for this request.
            
        Returns:
            DialogueResponse: The generated dialogue.
        """
        logger.info(f"Generating dialogue for {character_id}")
        
        # Step 1: Create character-specific dialogue prompt
        request = self.dialogue_engine.create_character_dialogue_prompt(
            character_id, context
        )
        
        # Step 2: Set provider preference if specified
        if provider_preference:
            request.provider_preference = provider_preference
        
        # Step 3: Set maximum cost if specified
        if max_cost:
            request.max_cost = max_cost
        
        # Step 4: Optimize the request for cost efficiency
        optimization_result = self.cost_manager.optimize_request(request, max_cost)
        optimized_request = optimization_result.optimized_request or request
        
        # Step 5: Select a provider and generate content
        provider = self._select_provider(optimized_request)
        
        # Step 5.1: Try with selected provider
        # If first provider fails, try other available providers before falling back to template
        available_providers = self.orchestrator.get_available_providers()
        tried_providers = set()  # Track which providers we've already tried
        
        # First try with the selected provider
        if provider.provider_type != ProviderType.TEMPLATE:
            try:
                # Generate the response
                response = await provider.generate_content(optimized_request)
                
                # Step 6: Record usage for budget tracking
                self.cost_manager.record_usage(response)
                
                # Step 7: Validate and enhance the response
                validation_context = ValidationContext(
                    content_type=ContentType.DIALOGUE,
                    original_request=request,
                    corporate_profile=context.get("corporate_profile", {}),
                    stakeholder_profiles=context.get("stakeholder_profiles", {}),
                )
                
                quality_result = self.quality_engine.validate_and_enhance(response, validation_context)
                
                # Step 8: Return the enhanced response or original if no enhancement
                final_response = quality_result.enhanced_response or response
                
                return final_response
                
            except Exception as e:
                tried_providers.add(provider.provider_type)
                logger.warning(f"Provider {provider.provider_type} failed: {e}. Trying other providers...")
                
        # Step 5.2: Try other available providers if the first one failed
        for provider_type in available_providers:
            if provider_type in tried_providers or provider_type == ProviderType.TEMPLATE:
                continue  # Skip already tried providers and template provider
                
            try:
                backup_provider = self.orchestrator.select_provider(provider_type)
                logger.info(f"Trying backup provider: {provider_type}")
                
                response = await backup_provider.generate_content(optimized_request)
                self.cost_manager.record_usage(response)
                
                validation_context = ValidationContext(
                    content_type=ContentType.DIALOGUE,
                    original_request=request,
                    corporate_profile=context.get("corporate_profile", {}),
                    stakeholder_profiles=context.get("stakeholder_profiles", {}),
                )
                
                quality_result = self.quality_engine.validate_and_enhance(response, validation_context)
                final_response = quality_result.enhanced_response or response
                
                return final_response
                
            except Exception as e:
                tried_providers.add(provider_type)
                logger.warning(f"Backup provider {provider_type} failed: {e}")
        
        # Step 5.3: Only fall back to template as last resort after all real providers have failed
        logger.warning("All AI providers failed. Falling back to template provider.")
        return await self.fallback_provider.generate_content(request)
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get a summary of AI usage costs.
        
        Args:
            days: Number of days to include in summary.
            
        Returns:
            Dict[str, Any]: Usage summary statistics.
        """
        # Get metrics from orchestrator which includes cost data
        return self.orchestrator.get_metrics_summary(days=days)
    
    # Provider selection is now handled by the orchestrator
    
    def get_provider_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of provider health status.
        
        Returns:
            Dict[str, Dict[str, Any]]: Health status for each provider.
        """
        return self.orchestrator.get_provider_health_summary()
    
    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of AI usage metrics.
        
        Args:
            days: Number of days to include in summary.
            
        Returns:
            Dict[str, Any]: Usage metrics summary.
        """
        return self.orchestrator.get_metrics_summary(days=days)
    
    def _select_provider(self, request: AIRequest) -> AIServiceProvider:
        """Select the appropriate provider based on the request.
        
        Args:
            request: The AI request to process.
            
        Returns:
            An AIServiceProvider implementation to handle the request.
        """
        # Log available providers in orchestrator
        available_providers = list(self.orchestrator.providers.keys()) if hasattr(self.orchestrator, 'providers') else []
        logger.info(f"Available providers in orchestrator: {available_providers}")
        
        if not available_providers:
            logger.warning("No AI providers available. Using template fallback.")
            return self.fallback_provider
        
        # ALWAYS try real AI providers first - never immediately use templates
        try:
            # Get the preferred provider if specified
            if request.provider_preference and request.provider_preference != ProviderType.TEMPLATE:
                logger.info(f"Using preferred provider: {request.provider_preference}")
                provider = self.orchestrator.select_provider(request.provider_preference)
                logger.info(f"Selected provider: {provider.provider_type if hasattr(provider, 'provider_type') else 'unknown'}")
                return provider
            
            # Otherwise let the orchestrator decide based on availability and cost-optimization
            logger.info("No specific provider preference, letting orchestrator decide based on availability")
            provider = self.orchestrator.select_provider()
            logger.info(f"Orchestrator selected provider: {provider.provider_type if hasattr(provider, 'provider_type') else 'unknown'}")
            return provider
        except Exception as e:
            # Only use template fallback as a last resort, but fully expose the error
            logger.error(f"Provider selection failed: {type(e).__name__}: {e}")
            logger.error(f"FULL TRACEBACK: \n{traceback.format_exc()}")
            logger.warning("Using fallback provider due to above error.")
            return self.fallback_provider
    
    def cleanup_resources(self) -> None:
        """Clean up resources like cache entries and old metrics."""
        self.orchestrator.cleanup_resources()
