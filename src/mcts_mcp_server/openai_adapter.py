#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI LLM Adapter
==================

This module defines the OpenAIAdapter class for interacting with OpenAI models.
"""
import logging
import os
import openai # type: ignore
from typing import AsyncGenerator, List, Dict, Any, Optional

from .base_llm_adapter import BaseLLMAdapter
from .llm_interface import LLMInterface # For type hinting or if BaseLLMAdapter doesn't explicitly inherit

class OpenAIAdapter(BaseLLMAdapter):
    """
    LLM Adapter for OpenAI models.
    """
    DEFAULT_MODEL = "gpt-3.5-turbo" # A common default, can be overridden

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs) # Pass kwargs to BaseLLMAdapter

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided via argument or OPENAI_API_KEY environment variable.")

        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__) # Ensure logger is initialized
        self.logger.info(f"Initialized OpenAIAdapter with model: {self.model_name}")

    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str: # Removed default for model
        """
        Gets a non-streaming completion from the OpenAI LLM.
        """
        target_model = model if model is not None else self.model_name # Explicit None check
        self.logger.debug(f"OpenAI get_completion using model: {target_model}, messages: {messages}, kwargs: {kwargs}")
        try:
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages, # type: ignore
                **kwargs
            )
            content = response.choices[0].message.content
            if content is None:
                self.logger.warning("OpenAI response content was None.")
                return ""
            return content
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error in get_completion: {e}")
            # Depending on desired behavior, either re-raise or return an error string/default
            # For MCTS, returning an error string might be better than crashing.
            return f"Error: OpenAI API request failed - {type(e).__name__}: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI get_completion: {e}")
            return f"Error: Unexpected error during OpenAI request - {type(e).__name__}: {e}"

    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]: # Removed default for model
        """
        Gets a streaming completion from the OpenAI LLM.
        """
        target_model = model if model is not None else self.model_name # Explicit None check
        self.logger.debug(f"OpenAI get_streaming_completion using model: {target_model}, messages: {messages}, kwargs: {kwargs}")
        try:
            stream = await self.client.chat.completions.create(
                model=target_model,
                messages=messages, # type: ignore
                stream=True,
                **kwargs
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error in get_streaming_completion: {e}")
            yield f"Error: OpenAI API request failed - {type(e).__name__}: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI get_streaming_completion: {e}")
            yield f"Error: Unexpected error during OpenAI streaming request - {type(e).__name__}: {e}"
        # Ensure the generator is properly closed if an error occurs before any yield
        # This is mostly handled by async for, but good to be mindful.
        # No explicit 'return' needed in an async generator after all yields or errors.

# Example of how to use (for testing purposes)
async def _test_openai_adapter():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # This test requires OPENAI_API_KEY to be set in the environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, skipping OpenAIAdapter direct test.")
        return

    try:
        adapter = OpenAIAdapter(model_name="gpt-3.5-turbo") # Or your preferred model

        logger.info("Testing OpenAIAdapter get_completion...")
        messages = [{"role": "user", "content": "Hello, what is the capital of France?"}]
        completion = await adapter.get_completion(messages=messages)
        logger.info(f"Completion result: {completion}")
        assert "Paris" in completion

        logger.info("Testing OpenAIAdapter get_streaming_completion...")
        stream_messages = [{"role": "user", "content": "Write a short poem about AI."}]
        full_streamed_response = ""
        async for chunk in adapter.get_streaming_completion(messages=stream_messages):
            logger.info(f"Stream chunk: '{chunk}'")
            full_streamed_response += chunk
        logger.info(f"Full streamed response: {full_streamed_response}")
        assert len(full_streamed_response) > 0

        # Test a base class method (e.g., generate_tags)
        logger.info("Testing OpenAIAdapter (via BaseLLMAdapter) generate_tags...")
        tags_text = "This is a test of the emergency broadcast system. This is only a test."
        tags = await adapter.generate_tags(analysis_text=tags_text, config={}) # Pass empty config
        logger.info(f"Generated tags: {tags}")
        assert "test" in tags or "emergency" in tags

        logger.info("OpenAIAdapter tests completed successfully (if API key was present).")

    except ValueError as ve:
        logger.error(f"ValueError during OpenAIAdapter test (likely API key issue): {ve}")
    except openai.APIError as apie:
        logger.error(f"OpenAI APIError during OpenAIAdapter test: {apie}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAIAdapter test: {e}", exc_info=True)

if __name__ == "__main__":
    # To run this test, ensure OPENAI_API_KEY is set in your environment
    # e.g., export OPENAI_API_KEY="your_key_here"
    # then run: python -m src.mcts_mcp_server.openai_adapter
    import asyncio
    if os.getenv("OPENAI_API_KEY"):
        asyncio.run(_test_openai_adapter())
    else:
        print("Skipping OpenAIAdapter test as OPENAI_API_KEY is not set.")
