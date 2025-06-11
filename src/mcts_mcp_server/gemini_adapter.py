#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini LLM Adapter
=========================

This module defines the GeminiAdapter class for interacting with Google Gemini models.
"""
import logging
import os
import google.generativeai as genai # type: ignore
from typing import AsyncGenerator, List, Dict, Any, Optional

from .base_llm_adapter import BaseLLMAdapter

# Default safety settings for Gemini - can be overridden via kwargs
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

class GeminiAdapter(BaseLLMAdapter):
    """
    LLM Adapter for Google Gemini models.
    """
    DEFAULT_MODEL = "gemini-1.5-flash-latest" # Using "latest" for flash

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
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
        self.logger.info(f"Initialized GeminiAdapter with model: {self.model_name}")

    def _get_model_client(self, model_name_override: Optional[str] = None) -> Any:
        """
        Returns the appropriate model client. If a model_name_override is provided
        and differs from the instance's default, it returns a new client for that model.
        """
        target_model_name = model_name_override if model_name_override else self.model_name
        if model_name_override and model_name_override != self.model_name:
            self.logger.debug(f"Using a temporary Gemini client for model: {model_name_override}")
            return genai.GenerativeModel(model_name_override)
        return self.client

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
        Gets a non-streaming completion from the Gemini LLM.
        """
        effective_model_client = self._get_model_client(model) # Handles None for model to use self.model_name

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
        Gets a streaming completion from the Gemini LLM.
        """
        effective_model_client = self._get_model_client(model) # Handles None for model to use self.model_name
        system_instruction, gemini_messages = self._convert_messages_to_gemini_format(messages)

        if not gemini_messages:
            self.logger.warning("No user/model messages to send to Gemini for streaming. Yielding nothing.")
            if False: yield # Must be a generator
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
        adapter = GeminiAdapter() # Uses default model

        logger.info("Testing GeminiAdapter get_completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Germany?"}
        ]
        completion = await adapter.get_completion(messages=messages)
        logger.info(f"Completion result: {completion}")
        assert "Berlin" in completion

        logger.info("Testing GeminiAdapter get_streaming_completion...")
        stream_messages = [{"role": "user", "content": "Write a short fun fact about space."}]
        full_streamed_response = ""
        async for chunk in adapter.get_streaming_completion(messages=stream_messages, generation_config={"temperature": 0.7}):
            logger.info(f"Stream chunk: '{chunk}'")
            full_streamed_response += chunk
        logger.info(f"Full streamed response: {full_streamed_response}")
        assert len(full_streamed_response) > 0

        logger.info("OpenAIAdapter (via BaseLLMAdapter) generate_tags...")
        tags_text = "The quick brown fox jumps over the lazy dog. This is a test for gemini."
        tags = await adapter.generate_tags(analysis_text=tags_text, config={}) # Pass empty config
        logger.info(f"Generated tags: {tags}")
        assert "fox" in tags or "gemini" in tags


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
