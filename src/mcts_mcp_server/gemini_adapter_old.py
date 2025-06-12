# -*- coding: utf-8 -*-
"""
Google Gemini LLM Adapter
=========================

This module defines the GeminiAdapter class for interacting with Google Gemini models.
Includes rate limiting for free tier models.
"""
import logging
import os
import google.generativeai as genai # type: ignore
from typing import AsyncGenerator, List, Dict, Any, Optional

from .base_llm_adapter import BaseLLMAdapter
from .rate_limiter import ModelRateLimitManager, RateLimitConfig

# Default safety settings for Gemini - can be overridden via kwargs
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

class GeminiAdapter(BaseLLMAdapter):
    """
    LLM Adapter for Google Gemini models with rate limiting support.
    """
    DEFAULT_MODEL = "gemini-1.5-flash-latest" # Using "latest" for flash

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 enable_rate_limiting: bool = True, custom_rate_limits: Optional[Dict[str, RateLimitConfig]] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided via argument or GEMINI_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)

        self.model_name = model_name or self.DEFAULT_MODEL
        # The client (GenerativeModel instance) might be model-specific.
        # We'll initialize it here but might need to re-initialize if 'model' param in methods is different.
        self.client = genai.GenerativeModel(self.model_name)
        self.logger = logging.getLogger(__name__)

        # Initialize rate limiting
        self.enable_rate_limiting = enable_rate_limiting
        if self.enable_rate_limiting:
            # Add specific rate limits for the models mentioned by user
            gemini_rate_limits = {
                "gemini-2.5-flash-preview-05-20": RateLimitConfig(requests_per_minute=10, burst_allowance=1),
                "gemini-2.0-flash-exp": RateLimitConfig(requests_per_minute=10, burst_allowance=1),
                "gemini-1.5-flash": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
                "gemini-1.5-flash-8b": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
                "gemini-1.5-pro": RateLimitConfig(requests_per_minute=360, burst_allowance=5),
                "gemini-2.0-flash-thinking-exp": RateLimitConfig(requests_per_minute=60, burst_allowance=3),
            }

            # Merge with any custom rate limits provided
            if custom_rate_limits:
                gemini_rate_limits.update(custom_rate_limits)

            self.rate_limit_manager = ModelRateLimitManager(custom_limits=gemini_rate_limits)
            self.logger.info(f"Initialized GeminiAdapter with rate limiting enabled for model: {self.model_name}")
        else:
            self.rate_limit_manager = None
            self.logger.info(f"Initialized GeminiAdapter without rate limiting for model: {self.model_name}")

    def _get_model_client(self, model_name_override: Optional[str] = None) -> Any:
        """
        Returns the appropriate model client. If a model_name_override is provided
        and differs from the instance's default, it returns a new client for that model.
        """
        if model_name_override and model_name_override != self.model_name:
            self.logger.debug(f"Using a temporary Gemini client for model: {model_name_override}")
            return genai.GenerativeModel(model_name_override)
        return self.client

    def get_rate_limit_status(self, model_name: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Get rate limit status for a specific model.

        Args:
            model_name: Model to check (defaults to instance default model)

        Returns:
            Dictionary with rate limit status or None if rate limiting disabled
        """
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            return None

        target_model = model_name if model_name else self.model_name
        limiter = self.rate_limit_manager.get_limiter(target_model)
        return limiter.get_status()

    def get_all_rate_limit_status(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get rate limit status for all models.

        Returns:
            Dictionary mapping model names to their rate limit status,
            or None if rate limiting is disabled
        """
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            return None

        return self.rate_limit_manager.get_all_status()

    def add_custom_rate_limit(self, model_name: str, requests_per_minute: int, burst_allowance: int = 1) -> None:
        """
        Add a custom rate limit for a specific model.

        Args:
            model_name: Name of the model
            requests_per_minute: Maximum requests per minute
            burst_allowance: Number of requests that can be made immediately
        """
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            self.logger.warning("Rate limiting is disabled, cannot add custom rate limit")
            return

        config = RateLimitConfig(requests_per_minute=requests_per_minute, burst_allowance=burst_allowance)
        self.rate_limit_manager.add_custom_limit(model_name, config)
        self.logger.info(f"Added custom rate limit for {model_name}: {requests_per_minute} RPM, {burst_allowance} burst")

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Converts messages to Gemini format, extracting system prompt.
        Gemini expects 'parts': [{'text': '...'}].
        Roles are 'user' and 'model'.
        """
        gemini_messages: List[Dict[str, Any]] = []
        system_instruction: Optional[str] = None

        if not messages:
            return system_instruction, gemini_messages

        # Handle system prompt: Gemini prefers it via `system_instruction` in GenerationConfig
        # or as the first part of the first 'user' message if not directly supported by model.
        # For `genai.GenerativeModel`, system_instruction is part of the model initialization.
        # If we want to change it per call, we'd need to re-init or adapt.
        # The current library version (0.8.x) supports system_instruction in generate_content_async.

        current_messages = list(messages) # Create a mutable copy

        if current_messages and current_messages[0].get("role") == "system":
            system_instruction = current_messages.pop(0).get("content", "")

        for message in current_messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                gemini_messages.append({'role': 'user', 'parts': [{'text': content}]})
            elif role == "assistant":
                gemini_messages.append({'role': 'model', 'parts': [{'text': content}]})
            # Silently ignore other roles or log them
            elif role != "system": # System role already handled
                 self.logger.warning(f"Gemini adapter: Unsupported role '{role}' encountered and skipped.")

        return system_instruction, gemini_messages

    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str: # Removed default for model
        """
        Gets a non-streaming completion from the Gemini LLM with rate limiting.
        """
        effective_model_client = self._get_model_client(model) # Handles None for model to use self.model_name
        target_model_name = model if model else self.model_name

        # Apply rate limiting if enabled
        if self.enable_rate_limiting and self.rate_limit_manager:
            self.logger.debug(f"Applying rate limit for model: {target_model_name}")
            await self.rate_limit_manager.acquire_for_model(target_model_name)

        system_instruction, gemini_messages = self._convert_messages_to_gemini_format(messages)

        if not gemini_messages: # Gemini requires non-empty messages
            self.logger.warning("No user/model messages to send to Gemini after processing. Returning empty.")
            return ""

        # Prepare generation_config and safety_settings from kwargs
        # Allow users to override defaults by passing them in kwargs
        generation_config_args = kwargs.get('generation_config', {})
        if system_instruction:
            # Some models might prefer system_instruction here.
            # The genai library allows system_instruction at the model level or in generate_content.
            # We will pass it to generate_content_async if available.
            if 'system_instruction' not in generation_config_args: # Don't override if already set
                 generation_config_args['system_instruction'] = system_instruction

        # Convert dict to GenerationConfig object if not already
        if isinstance(generation_config_args, dict):
            generation_config = genai.types.GenerationConfig(**generation_config_args)
        else: # Assume it's already a GenerationConfig object
            generation_config = generation_config_args

        safety_settings = kwargs.get('safety_settings', DEFAULT_SAFETY_SETTINGS)

        self.logger.debug(f"Gemini get_completion using model: {effective_model_client.model_name}, messages: {gemini_messages}, system_instruction (in config): {system_instruction}, config: {generation_config}, safety: {safety_settings}")

        try:
            response = await effective_model_client.generate_content_async(
                gemini_messages,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            if response.text:
                return response.text
            else: # Handle cases like blocked prompts
                self.logger.warning(f"Gemini response was empty or blocked. Prompt feedback: {response.prompt_feedback}")
                return f"Error: Gemini response empty or blocked. Feedback: {response.prompt_feedback}"
        except Exception as e: # Catching general google.api_core.exceptions too
            self.logger.error(f"Gemini API error in get_completion: {e}", exc_info=True)
            return f"Error: Gemini API request failed - {type(e).__name__}: {e}"

    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]: # Removed default for model
        """
        Gets a streaming completion from the Gemini LLM with rate limiting.
        """
        effective_model_client = self._get_model_client(model) # Handles None for model to use self.model_name
        target_model_name = model if model else self.model_name

        # Apply rate limiting if enabled
        if self.enable_rate_limiting and self.rate_limit_manager:
            self.logger.debug(f"Applying rate limit for streaming model: {target_model_name}")
            await self.rate_limit_manager.acquire_for_model(target_model_name)

        system_instruction, gemini_messages = self._convert_messages_to_gemini_format(messages)

        if not gemini_messages:
            self.logger.warning("No user/model messages to send to Gemini for streaming. Yielding nothing.")
            return

        generation_config_args = kwargs.get('generation_config', {})
        if system_instruction:
            if 'system_instruction' not in generation_config_args:
                 generation_config_args['system_instruction'] = system_instruction

        if isinstance(generation_config_args, dict):
            generation_config = genai.types.GenerationConfig(**generation_config_args)
        else:
            generation_config = generation_config_args

        safety_settings = kwargs.get('safety_settings', DEFAULT_SAFETY_SETTINGS)

        self.logger.debug(f"Gemini get_streaming_completion using model: {effective_model_client.model_name}, messages: {gemini_messages}, system_instruction (in config): {system_instruction}, config: {generation_config}, safety: {safety_settings}")

        try:
            response_stream = await effective_model_client.generate_content_async(
                gemini_messages,
                stream=True,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self.logger.error(f"Gemini API error in get_streaming_completion: {e}", exc_info=True)
            yield f"Error: Gemini API request failed during stream - {type(e).__name__}: {e}"

# Example of how to use (for testing purposes)
async def _test_gemini_adapter():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set, skipping GeminiAdapter direct test.")
        return

    try:
        # Test with rate limiting enabled (default)
        adapter = GeminiAdapter() # Uses default model

        logger.info("Testing rate limit status...")
        status = adapter.get_rate_limit_status()
        logger.info(f"Rate limit status: {status}")

        # Test adding custom rate limit
        adapter.add_custom_rate_limit("gemini-test-model", requests_per_minute=5, burst_allowance=2)

        logger.info("Testing GeminiAdapter get_completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Germany?"}
        ]
        completion = await adapter.get_completion(model=None, messages=messages)
        logger.info(f"Completion result: {completion}")
        assert "Berlin" in completion

        logger.info("Testing specific model with rate limiting...")
        # Test the specific model mentioned by user
        completion_preview = await adapter.get_completion(
            model="gemini-2.5-flash-preview-05-20",
            messages=[{"role": "user", "content": "Hello, just testing!"}]
        )
        logger.info(f"Preview model completion: {completion_preview}")

        logger.info("Testing GeminiAdapter get_streaming_completion...")
        stream_messages = [{"role": "user", "content": "Write a short fun fact about space."}]
        full_streamed_response = ""
        async for chunk in adapter.get_streaming_completion(model=None, messages=stream_messages, generation_config={"temperature": 0.7}):
            logger.info(f"Stream chunk: '{chunk}'")
            full_streamed_response += chunk
        logger.info(f"Full streamed response: {full_streamed_response}")
        assert len(full_streamed_response) > 0

        logger.info("Testing BaseLLMAdapter generate_tags...")
        tags_text = "The quick brown fox jumps over the lazy dog. This is a test for gemini."
        tags = await adapter.generate_tags(analysis_text=tags_text, config={}) # Pass empty config
        logger.info(f"Generated tags: {tags}")
        assert "fox" in tags or "gemini" in tags

        # Test rate limiting status after requests
        all_status = adapter.get_all_rate_limit_status()
        logger.info(f"All rate limit status after requests: {all_status}")

        # Test adapter without rate limiting
        logger.info("Testing adapter without rate limiting...")
        no_limit_adapter = GeminiAdapter(enable_rate_limiting=False)
        no_limit_status = no_limit_adapter.get_rate_limit_status()
        logger.info(f"No rate limit adapter status: {no_limit_status}")

        logger.info("GeminiAdapter tests completed successfully (if API key was present).")

    except ValueError as ve:
        logger.error(f"ValueError during GeminiAdapter test (likely API key issue): {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during GeminiAdapter test: {e}", exc_info=True)

if __name__ == "__main__":
    # To run this test, ensure GEMINI_API_KEY is set
    # then run: python -m src.mcts_mcp_server.gemini_adapter
    import asyncio
    if os.getenv("GEMINI_API_KEY"):
        asyncio.run(_test_gemini_adapter())
    else:
        print("Skipping GeminiAdapter test as GEMINI_API_KEY is not set.")
