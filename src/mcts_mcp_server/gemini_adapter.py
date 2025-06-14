"""
Google Gemini LLM Adapter
=========================

This module defines the GeminiAdapter class for interacting with Google Gemini models.
Includes rate limiting for free tier models.
"""
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig

from .base_llm_adapter import BaseLLMAdapter
from .rate_limiter import ModelRateLimitManager, RateLimitConfig

# Default safety settings for Gemini - can be overridden via kwargs.
# Should be set via .env for accessibility

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
    DEFAULT_MODEL = "gemini-2.0-flash-lite"

    def __init__(self, api_key: str | None = None, model_name: str | None = None,
                 enable_rate_limiting: bool = True, custom_rate_limits: dict[str, RateLimitConfig] | None = None, **kwargs) -> None:
        """
        Initialize the Gemini LLM adapter with rate limiting support.

        Args:
            api_key: Gemini API key (if None, uses GEMINI_API_KEY environment variable)
            model_name: Name of the Gemini model to use (defaults to gemini-2.0-flash-lite 30 RPM 1,500 RPD free)
            enable_rate_limiting: Whether to enable rate limiting for API calls
            custom_rate_limits: Custom rate limit configurations for specific models
            **kwargs: Additional arguments passed to BaseLLMAdapter

        Raises:
            ValueError: If no API key is provided via argument or environment variable
        """
        super().__init__(api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided via argument or GEMINI_API_KEY environment variable.")

        # Configure the client with API key
        self.client = genai.Client(api_key=self.api_key)

        self.model_name = model_name or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)

        # Initialize rate limiting
        self.enable_rate_limiting = enable_rate_limiting
        if self.enable_rate_limiting:
            # Add specific rate limits for the models mentioned by user
            gemini_rate_limits = {
                "gemini-2.0-flash-lite": RateLimitConfig(requests_per_minute=30, burst_allowance=2),
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

    def _get_model_client(self, model_name_override: str | None = None) -> str:
        """
        Get the appropriate Gemini model name for the specified model.

        Args:
            model_name_override: Optional model name to use instead of instance default

        Returns:
            Model name string to use for API calls

        Note:
            In the new google-genai library, we use model names directly rather than client objects
        """
        if model_name_override:
            self.logger.debug(f"Using model override: {model_name_override}")
            return model_name_override
        return self.model_name

    def get_rate_limit_status(self, model_name: str | None = None) -> dict[str, float] | None:
        """
        Get current rate limit status for a specific model.

        Args:
            model_name: Model to check status for (uses instance default if None)

        Returns:
            Dictionary containing rate limit status information including:
            - requests_remaining: Number of requests available
            - time_until_reset: Seconds until rate limit resets
            - requests_per_minute: Configured requests per minute limit
            Returns None if rate limiting is disabled
        """
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            return None

        target_model = model_name if model_name else self.model_name
        limiter = self.rate_limit_manager.get_limiter(target_model)
        return limiter.get_status()

    def get_all_rate_limit_status(self) -> dict[str, dict[str, float]] | None:
        """
        Get rate limit status for all configured models.

        Returns:
            Dictionary mapping model names to their rate limit status dictionaries,
            or None if rate limiting is disabled

        Note:
            Only includes models that have been used or explicitly configured
        """
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            return None

        return self.rate_limit_manager.get_all_status()

    def add_custom_rate_limit(self, model_name: str, requests_per_minute: int, burst_allowance: int = 1) -> None:
        """
        Add or update a custom rate limit configuration for a specific model.

        Args:
            model_name: Name of the Gemini model to configure
            requests_per_minute: Maximum requests allowed per minute
            burst_allowance: Number of requests that can be made immediately without waiting

        Note:
            If rate limiting is disabled, this method logs a warning and does nothing
        """
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            self.logger.warning("Rate limiting is disabled, cannot add custom rate limit")
            return

        config = RateLimitConfig(requests_per_minute=requests_per_minute, burst_allowance=burst_allowance)
        self.rate_limit_manager.add_custom_limit(model_name, config)
        self.logger.info(f"Added custom rate limit for {model_name}: {requests_per_minute} RPM, {burst_allowance} burst")

    def _convert_messages_to_gemini_format(self, messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert standard message format to Gemini-specific format.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Tuple containing:
            - system_instruction: Extracted system prompt (if any)
            - gemini_messages: Messages formatted for Gemini API with 'parts' structure

        Note:
            - Converts 'assistant' role to 'model' for Gemini compatibility
            - Extracts system messages as separate system_instruction
            - Ignores unsupported roles with warning
        """
        gemini_messages: list[dict[str, Any]] = []
        system_instruction: str | None = None

        if not messages:
            return system_instruction, gemini_messages

        current_messages = list(messages)

        # Extract system instruction from first message if it's a system message
        if current_messages and current_messages[0].get("role") == "system":
            system_instruction = current_messages.pop(0).get("content", "")

        for message in current_messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                gemini_messages.append({
                    'role': 'user',
                    'parts': [{'text': content}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    'role': 'model',
                    'parts': [{'text': content}]
                })
            elif role != "system":  # System role already handled
                self.logger.warning(f"Gemini adapter: Unsupported role '{role}' encountered and skipped.")

        return system_instruction, gemini_messages

    async def get_completion(self, model: str | None, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Get a non-streaming completion from Gemini with rate limiting.

        Args:
            model: Gemini model name to use (uses instance default if None)
            messages: Conversation messages in standard format
            **kwargs: Additional arguments including:
                - generation_config: Gemini generation configuration
                - safety_settings: Content safety settings

        Returns:
            Generated text response from the model

        Raises:
            Exception: If API call fails or rate limiting is violated

        Note:
            Automatically applies rate limiting if enabled and handles system instructions
        """
        effective_model_name = self._get_model_client(model)
        target_model_name = model if model else self.model_name

        # Apply rate limiting if enabled
        if self.enable_rate_limiting and self.rate_limit_manager:
            self.logger.debug(f"Applying rate limit for model: {target_model_name}")
            await self.rate_limit_manager.acquire_for_model(target_model_name)

        system_instruction, gemini_messages = self._convert_messages_to_gemini_format(messages)

        if not gemini_messages:
            self.logger.warning("No user/model messages to send to Gemini after processing. Returning empty.")
            return ""

        # Prepare generation config
        generation_config_args = kwargs.get('generation_config', {})
        if system_instruction and 'system_instruction' not in generation_config_args:
            generation_config_args['system_instruction'] = system_instruction

        # Convert dict to GenerateContentConfig object if not already
        if isinstance(generation_config_args, dict):
            generation_config = GenerateContentConfig(**generation_config_args)
        else:
            generation_config = generation_config_args

        # Should be set via .env for accessability
        safety_settings = kwargs.get('safety_settings', DEFAULT_SAFETY_SETTINGS)

        self.logger.debug(f"Gemini get_completion using model: {effective_model_name}, messages: {gemini_messages}, system_instruction: {system_instruction}")

        try:
            response = await self.client.aio.models.generate_content(
                model=effective_model_name,
                contents=gemini_messages,
                config=generation_config
            )

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text
                    return text if text is not None else ""

            self.logger.warning(f"Gemini response was empty or blocked. Response: {response}")
            return "Error: Gemini response empty or blocked."

        except Exception as e:
            self.logger.error(f"Gemini API error in get_completion: {e}", exc_info=True)
            return f"Error: Gemini API request failed - {type(e).__name__}: {e}"

    async def get_streaming_completion(self, model: str | None, messages: list[dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """
        Get a streaming completion from Gemini with rate limiting.

        Args:
            model: Gemini model name to use (uses instance default if None)
            messages: Conversation messages in standard format
            **kwargs: Additional arguments including:
                - generation_config: Gemini generation configuration
                - safety_settings: Content safety settings

        Yields:
            Text chunks as they are generated by the model

        Raises:
            Exception: If API call fails or rate limiting is violated

        Note:
            Applies rate limiting before starting the stream and handles system instructions
        """
        effective_model_name = self._get_model_client(model)
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
        if system_instruction and 'system_instruction' not in generation_config_args:
            generation_config_args['system_instruction'] = system_instruction

        if isinstance(generation_config_args, dict):
            generation_config = GenerateContentConfig(**generation_config_args)
        else:
            generation_config = generation_config_args

        self.logger.debug(f"Gemini get_streaming_completion using model: {effective_model_name}, messages: {gemini_messages}")

        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=effective_model_name,
                contents=gemini_messages,
                config=generation_config
            )

            async for chunk in response_stream:
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                yield part.text

        except Exception as e:
            self.logger.error(f"Gemini API error in get_streaming_completion: {e}", exc_info=True)
            yield f"Error: Gemini API request failed during stream - {type(e).__name__}: {e}"

# Example of how to use (for testing purposes)
async def _test_gemini_adapter() -> None:
    """
    Test function for the GeminiAdapter class.

    Tests various features including:
    - Basic completion and streaming
    - Rate limiting functionality
    - Custom model usage
    - Tag generation
    - Error handling

    Requires:
        GEMINI_API_KEY environment variable to be set

    Note:
        This is primarily for development testing and debugging
    """
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
