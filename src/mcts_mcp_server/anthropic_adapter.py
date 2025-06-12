#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anthropic LLM Adapter
=====================

This module defines the AnthropicAdapter class for interacting with Anthropic models.
"""
import logging
import os
import anthropic # type: ignore
from typing import AsyncGenerator, List, Dict, Any, Optional

from .base_llm_adapter import BaseLLMAdapter
# LLMInterface import might not be strictly needed if BaseLLMAdapter is comprehensive
# from .llm_interface import LLMInterface

class AnthropicAdapter(BaseLLMAdapter):
    """
    LLM Adapter for Anthropic models.
    """
    DEFAULT_MODEL = "claude-3-haiku-20240307"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided via argument or ANTHROPIC_API_KEY environment variable.")

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized AnthropicAdapter with model: {self.model_name}")

    def _prepare_anthropic_messages_and_system_prompt(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Separates the system prompt from messages if present, as Anthropic API handles it differently.
        Ensures messages conform to Anthropic's expected format.
        """
        system_prompt_content: Optional[str] = None
        processed_messages: List[Dict[str, Any]] = []

        if not messages:
            return None, []

        # Check if the first message is a system prompt
        if messages[0].get("role") == "system":
            system_prompt_content = messages[0].get("content", "")
            # Ensure messages passed to Anthropic API alternate user/assistant roles
            # and start with a 'user' role if a system prompt was extracted.
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages

        # Process remaining messages: Anthropic expects alternating user/assistant roles.
        # If the first message is assistant, or two user/assistant messages are consecutive,
        # it can cause errors. This basic processing assumes a simple alternating structure
        # or that the calling MCTS logic provides messages in an alternating user/assistant way
        # after any initial system prompt.
        # For now, we'll pass them as is, assuming the input `messages` (after system prompt extraction)
        # mostly conforms to this. More robust handling might be needed if not.

        # Ensure all 'content' fields are strings. Anthropic can handle multiple content blocks.
        for msg in remaining_messages:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"]: # Anthropic only accepts these roles in `messages`
                if isinstance(content, str):
                    processed_messages.append({"role": role, "content": content})
                elif isinstance(content, list): # For multi-modal content, though we focus on text here
                    processed_messages.append({"role": role, "content": content}) # Pass as is
                else:
                    self.logger.warning(f"Message content is not string or list, converting to string: {content}")
                    processed_messages.append({"role": role, "content": str(content)})
            else:
                self.logger.warning(f"Unsupported role '{role}' in Anthropic messages, skipping.")


        # Ensure the first message is 'user' if there's no system prompt,
        # or if there was a system prompt and now the message list is not empty.
        if processed_messages and processed_messages[0].get("role") != "user":
            # This scenario can be complex. If it's assistant, API might error.
            # Prepending a dummy user message might be a hack.
            # For now, log a warning. The MCTS message structure should ideally handle this.
            self.logger.warning(f"First message to Anthropic (after system prompt) is not 'user': {processed_messages[0].get('role')}")


        return system_prompt_content, processed_messages


    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str: # Removed default for model
        """
        Gets a non-streaming completion from the Anthropic LLM.
        """
        target_model = model if model is not None else self.model_name # Explicit None check

        system_prompt, processed_messages = self._prepare_anthropic_messages_and_system_prompt(messages)

        # Filter out max_tokens from kwargs if present, as it's explicitly managed.
        # Other kwargs like temperature, top_p, etc., can be passed through.
        api_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
        max_tokens = kwargs.get("max_tokens", 4096) # Anthropic's default is often higher, but good to control.

        if not processed_messages: # Anthropic requires at least one message
             self.logger.warning("No user/assistant messages to send to Anthropic after processing. Returning empty.")
             return ""

        self.logger.debug(f"Anthropic get_completion using model: {target_model}, system: '{system_prompt}', messages: {processed_messages}, max_tokens: {max_tokens}, kwargs: {api_kwargs}")

        try:
            response = await self.client.messages.create(
                model=target_model,
                max_tokens=max_tokens,
                system=system_prompt, # type: ignore
                messages=processed_messages, # type: ignore
                **api_kwargs
            )
            # Assuming the response structure contains content in a list, and we take the first text block.
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                return response.content[0].text
            else:
                self.logger.warning(f"Anthropic response content not in expected format: {response}")
                return ""
        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API error in get_completion: {e}")
            return f"Error: Anthropic API request failed - {type(e).__name__}: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error in Anthropic get_completion: {e}")
            return f"Error: Unexpected error during Anthropic request - {type(e).__name__}: {e}"

    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]: # Removed default for model
        """
        Gets a streaming completion from the Anthropic LLM.
        """
        target_model = model if model is not None else self.model_name # Explicit None check
        system_prompt, processed_messages = self._prepare_anthropic_messages_and_system_prompt(messages)

        api_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
        max_tokens = kwargs.get("max_tokens", 4096)

        if not processed_messages:
            self.logger.warning("No user/assistant messages to send to Anthropic for streaming. Yielding nothing.")
<<<<<<< HEAD
=======
            if False: yield # Must be a generator
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
            return

        self.logger.debug(f"Anthropic get_streaming_completion using model: {target_model}, system: '{system_prompt}', messages: {processed_messages}, max_tokens: {max_tokens}, kwargs: {api_kwargs}")

        try:
            async with await self.client.messages.stream(
                model=target_model,
                max_tokens=max_tokens,
                system=system_prompt, # type: ignore
                messages=processed_messages, # type: ignore
                **api_kwargs
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk
        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API error in get_streaming_completion: {e}")
            yield f"Error: Anthropic API request failed - {type(e).__name__}: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error in Anthropic get_streaming_completion: {e}")
            yield f"Error: Unexpected error during Anthropic streaming request - {type(e).__name__}: {e}"

<<<<<<< HEAD
    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """
        Classifies user intent using the Anthropic LLM.
        """
        system_prompt = """You are an intent classification system. Analyze the given text and classify the user's intent into one of these categories:

- ANALYZE_NEW: User wants to analyze something new or start a fresh analysis
- CONTINUE_ANALYSIS: User wants to continue, expand, or elaborate on existing analysis
- ASK_LAST_RUN_SUMMARY: User is asking for a summary, results, or what was found
- ASK_PROCESS: User is asking about how the system works, methods, algorithms, or processes
- ASK_CONFIG: User is asking about configuration, settings, or parameters
- GENERAL_CONVERSATION: User wants casual conversation or general chat

Respond with only the category name (exactly as shown above)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify the intent of this text: {text_to_classify}"}
        ]

        try:
            model = config.get("model", self.model_name)
            max_tokens = config.get("max_tokens", 50)

            result = await self.get_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for consistent classification
            )

            # Clean up the result and validate
            intent = result.strip().upper()

            # Validate the intent is one of our expected categories
            valid_intents = {
                "ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY",
                "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"
            }

            if intent not in valid_intents:
                self.logger.warning(f"Unexpected intent classification: {intent}, defaulting to 'ANALYZE_NEW'")
                return "ANALYZE_NEW"

            return intent

        except Exception as e:
            self.logger.error(f"Error in classify_intent: {e}")
            return "ANALYZE_NEW"  # Default fallback

=======
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
# Example of how to use (for testing purposes)
async def _test_anthropic_adapter():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set, skipping AnthropicAdapter direct test.")
        return

    try:
        adapter = AnthropicAdapter(model_name="claude-3-haiku-20240307")

        logger.info("Testing AnthropicAdapter get_completion...")
        # Example with system prompt
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
            {"role": "user", "content": "Hello, what is the capital of France?"}
        ]
<<<<<<< HEAD
        completion = await adapter.get_completion(model=None, messages=messages_with_system)
=======
        completion = await adapter.get_completion(messages=messages_with_system)
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
        logger.info(f"Completion result (with system prompt): {completion}")
        assert "Paris" in completion

        logger.info("Testing AnthropicAdapter get_streaming_completion...")
        stream_messages = [{"role": "user", "content": "Write a very short poem about AI."}]
        full_streamed_response = ""
<<<<<<< HEAD
        async for chunk in adapter.get_streaming_completion(model=None, messages=stream_messages, max_tokens=50):
=======
        async for chunk in adapter.get_streaming_completion(messages=stream_messages, max_tokens=50):
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
            logger.info(f"Stream chunk: '{chunk}'")
            full_streamed_response += chunk
        logger.info(f"Full streamed response: {full_streamed_response}")
        assert len(full_streamed_response) > 0

        logger.info("Testing AnthropicAdapter (via BaseLLMAdapter) generate_tags...")
        tags_text = "This is a test of the emergency broadcast system for anthropic models. This is only a test."
        tags = await adapter.generate_tags(analysis_text=tags_text, config={})
        logger.info(f"Generated tags: {tags}")
        assert "test" in tags or "anthropic" in tags

        logger.info("AnthropicAdapter tests completed successfully (if API key was present).")

    except ValueError as ve:
        logger.error(f"ValueError during AnthropicAdapter test (likely API key issue): {ve}")
    except anthropic.APIError as apie:
        logger.error(f"Anthropic APIError during AnthropicAdapter test: {apie}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during AnthropicAdapter test: {e}", exc_info=True)

if __name__ == "__main__":
    # To run this test, ensure ANTHROPIC_API_KEY is set in your environment
    # e.g., export ANTHROPIC_API_KEY="your_key_here"
    # then run: python -m src.mcts_mcp_server.anthropic_adapter
    import asyncio
    if os.getenv("ANTHROPIC_API_KEY"):
        asyncio.run(_test_anthropic_adapter())
    else:
        print("Skipping AnthropicAdapter test as ANTHROPIC_API_KEY is not set.")
