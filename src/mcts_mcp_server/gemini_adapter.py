#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini LLM Adapter (Corrected for google-genai package)
=============================================================

This module defines the GeminiAdapter class for interacting with Google Gemini models
using the new google-genai package with the correct API format.
"""
import logging
import os
import asyncio
from google import genai
from google.genai import types
from typing import AsyncGenerator, List, Dict, Optional

from .base_llm_adapter import BaseLLMAdapter
from .rate_limiter import ModelRateLimitManager, RateLimitConfig

class GeminiAdapter(BaseLLMAdapter):
    """
    LLM Adapter for Google Gemini models using the new google-genai package.
    """
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 enable_rate_limiting: bool = True, custom_rate_limits: Optional[Dict[str, RateLimitConfig]] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided via argument or GEMINI_API_KEY environment variable.")

        # Initialize the new genai client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)

        # Initialize rate limiting
        self.enable_rate_limiting = enable_rate_limiting
        if self.enable_rate_limiting:
            gemini_rate_limits = {
                "gemini-2.5-flash-preview-05-20": RateLimitConfig(requests_per_minute=10, burst_allowance=1),
                "gemini-2.0-flash": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
                "gemini-2.0-flash-exp": RateLimitConfig(requests_per_minute=10, burst_allowance=1),
                "gemini-1.5-flash": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
                "gemini-1.5-flash-8b": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
                "gemini-1.5-pro": RateLimitConfig(requests_per_minute=360, burst_allowance=5),
                "gemini-2.0-flash-thinking-exp": RateLimitConfig(requests_per_minute=60, burst_allowance=3),
            }

            if custom_rate_limits:
                gemini_rate_limits.update(custom_rate_limits)

            self.rate_limit_manager = ModelRateLimitManager(custom_limits=gemini_rate_limits)
            self.logger.info(f"Initialized GeminiAdapter with rate limiting enabled for model: {self.model_name}")
        else:
            self.rate_limit_manager = None
            self.logger.info(f"Initialized GeminiAdapter without rate limiting for model: {self.model_name}")

    def get_rate_limit_status(self, model_name: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get rate limit status for a specific model."""
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            return None

        target_model = model_name if model_name else self.model_name
        limiter = self.rate_limit_manager.get_limiter(target_model)
        return limiter.get_status()

    def get_all_rate_limit_status(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get rate limit status for all models."""
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            return None

        return self.rate_limit_manager.get_all_status()

    def add_custom_rate_limit(self, model_name: str, requests_per_minute: int, burst_allowance: int = 1) -> None:
        """Add a custom rate limit for a specific model."""
        if not self.enable_rate_limiting or not self.rate_limit_manager:
            self.logger.warning("Rate limiting is disabled, cannot add custom rate limit")
            return

        config = RateLimitConfig(requests_per_minute=requests_per_minute, burst_allowance=burst_allowance)
        self.rate_limit_manager.add_custom_limit(model_name, config)
        self.logger.info(f"Added custom rate limit for {model_name}: {requests_per_minute} RPM, {burst_allowance} burst")

    def _convert_messages_to_genai_format(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[types.Content]]:
        """
        Converts messages to the correct genai format using Content and Part objects.
        """
        genai_contents: List[types.Content] = []
        system_instruction: Optional[str] = None

        if not messages:
            return system_instruction, genai_contents

        current_messages = list(messages)

        # Extract system prompt if present
        if current_messages and current_messages[0].get("role") == "system":
            system_instruction = current_messages.pop(0).get("content", "")

        for message in current_messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                # Create Content with role and parts
                content_obj = types.Content(
                    role="user",
                    parts=[types.Part(text=content)]
                )
                genai_contents.append(content_obj)

            elif role == "assistant":
                # Gemini uses "model" as the role for assistant responses
                content_obj = types.Content(
                    role="model",
                    parts=[types.Part(text=content)]
                )
                genai_contents.append(content_obj)

            elif role != "system":
                self.logger.warning(f"Gemini adapter: Unsupported role '{role}' encountered and skipped.")

        return system_instruction, genai_contents

    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Gets a non-streaming completion from the Gemini LLM with rate limiting.
        """
        target_model_name = model if model else self.model_name

        # Apply rate limiting if enabled
        if self.enable_rate_limiting and self.rate_limit_manager:
            self.logger.debug(f"Applying rate limit for model: {target_model_name}")
            await self.rate_limit_manager.acquire_for_model(target_model_name)

        system_instruction, genai_contents = self._convert_messages_to_genai_format(messages)

        if not genai_contents:
            self.logger.warning("No user/model messages to send to Gemini after processing. Returning empty.")
            return ""

        self.logger.debug(f"Gemini get_completion using model: {target_model_name}")

        try:
            # Prepare the request data
            request_data = {
                "model": target_model_name,
                "contents": genai_contents
            }

            # Add system instruction if present
            if system_instruction:
                # System instruction goes in config
                config = kwargs.get('config', {})
                config["system_instruction"] = system_instruction
                request_data["config"] = config
            elif kwargs.get('config'):
                request_data["config"] = kwargs['config']

            # Use asyncio.to_thread to run the sync API call
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                **request_data
            )

            # Extract text from response
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to extract from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        return candidate.content.parts[0].text or ""

            self.logger.warning(f"Gemini response was empty or blocked. Response: {response}")
            return "Error: Gemini response was empty or blocked."

        except Exception as e:
            self.logger.error(f"Gemini API error in get_completion: {e}", exc_info=True)
            return f"Error: Gemini API request failed - {type(e).__name__}: {e}"

    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """
        Gets a streaming completion from the Gemini LLM with rate limiting.
        """
        target_model_name = model if model else self.model_name

        # Apply rate limiting if enabled
        if self.enable_rate_limiting and self.rate_limit_manager:
            self.logger.debug(f"Applying rate limit for streaming model: {target_model_name}")
            await self.rate_limit_manager.acquire_for_model(target_model_name)

        system_instruction, genai_contents = self._convert_messages_to_genai_format(messages)

        if not genai_contents:
            self.logger.warning("No user/model messages to send to Gemini for streaming. Yielding nothing.")
            return

        self.logger.debug(f"Gemini get_streaming_completion using model: {target_model_name}")

        try:
            # Prepare the request data
            request_data = {
                "model": target_model_name,
                "contents": genai_contents
            }

            # Add system instruction if present
            if system_instruction:
                config = kwargs.get('config', {})
                config["system_instruction"] = system_instruction
                request_data["config"] = config
            elif kwargs.get('config'):
                request_data["config"] = kwargs['config']

            # Use asyncio.to_thread for streaming
            response_stream = await asyncio.to_thread(
                self.client.models.generate_content_stream,
                **request_data
            )

            # Handle streaming response
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    # Try to extract from candidates
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            text = candidate.content.parts[0].text
                            if text:
                                yield text

        except Exception as e:
            self.logger.error(f"Gemini API error in get_streaming_completion: {e}", exc_info=True)
            yield f"Error: Gemini API request failed during stream - {type(e).__name__}: {e}"

# Test function for the corrected adapter
async def _test_gemini_adapter():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set, skipping GeminiAdapter direct test.")
        return

    try:
        # Test with rate limiting enabled (default)
        adapter = GeminiAdapter()

        logger.info("Testing rate limit status...")
        status = adapter.get_rate_limit_status()
        logger.info(f"Rate limit status: {status}")

        logger.info("Testing GeminiAdapter get_completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Germany?"}
        ]
        completion = await adapter.get_completion(model=None, messages=messages)
        logger.info(f"Completion result: {completion}")

        logger.info("Testing GeminiAdapter get_streaming_completion...")
        stream_messages = [{"role": "user", "content": "Write a short fun fact about space."}]
        full_streamed_response = ""
        async for chunk in adapter.get_streaming_completion(model=None, messages=stream_messages):
            logger.info(f"Stream chunk: '{chunk}'")
            full_streamed_response += chunk
        logger.info(f"Full streamed response: {full_streamed_response}")

        logger.info("GeminiAdapter tests completed successfully.")

    except ValueError as ve:
        logger.error(f"ValueError during GeminiAdapter test (likely API key issue): {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during GeminiAdapter test: {e}", exc_info=True)

if __name__ == "__main__":
    import asyncio
    if os.getenv("GEMINI_API_KEY"):
        asyncio.run(_test_gemini_adapter())
    else:
        print("Skipping GeminiAdapter test as GEMINI_API_KEY is not set.")
