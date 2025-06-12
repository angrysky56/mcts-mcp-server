#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base LLM Adapter
================

This module defines the BaseLLMAdapter abstract base class.
"""
import abc
import logging
import re
from typing import List, Dict, Any, AsyncGenerator, Optional

from .llm_interface import LLMInterface
from .intent_handler import (
    INITIAL_PROMPT,
    THOUGHTS_PROMPT,
    UPDATE_PROMPT,
    EVAL_ANSWER_PROMPT,
    TAG_GENERATION_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    INTENT_CLASSIFIER_PROMPT
)

class BaseLLMAdapter(LLMInterface, abc.ABC):
    """
    Abstract Base Class for LLM adapters.
    Provides common prompt formatting and response processing logic.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        # Allow other kwargs to be stored if needed by subclasses
        self._kwargs = kwargs

    @abc.abstractmethod
    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Abstract method to get a non-streaming completion from the LLM.
        'model' can be None if the adapter is initialized with a specific model.
        """
        pass

    @abc.abstractmethod
    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """
        Abstract method to get a streaming completion from the LLM.
        'model' can be None if the adapter is initialized with a specific model.
        """
        # Required for async generator structure
        if False: # pragma: no cover
            yield
        pass

    async def generate_thought(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a critical thought or new direction based on context."""
        prompt = THOUGHTS_PROMPT.format(**context)
        messages = [{"role": "user", "content": prompt}]
        # Model might be specified in config or be a default for the adapter instance
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")
        return await self.get_completion(model=model_to_use, messages=messages)

    async def update_analysis(self, critique: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Revises analysis based on critique and context."""
        # Ensure 'critique' and 'answer' (draft) are in context for the prompt
        context_for_prompt = context.copy()
        context_for_prompt['critique'] = critique
        # 'answer' should be the draft text, usually passed in context as 'answer' or 'draft'
        # The UPDATE_PROMPT expects {answer} for the draft and {improvements} for the critique.
        # The MCTS core calls this with 'answer' (node.content) and 'improvements' (thought) in context.
        # Let's ensure the prompt matches the keys used in MCTS:
        # UPDATE_PROMPT uses <draft>{answer}</draft> and <critique>{improvements}</critique>
        # The MCTS context provides 'answer' for draft and 'improvements' for critique.

        prompt = UPDATE_PROMPT.format(**context_for_prompt)
        messages = [{"role": "user", "content": prompt}]
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")
        return await self.get_completion(model=model_to_use, messages=messages)

    async def evaluate_analysis(self, analysis_to_evaluate: str, context: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Evaluates analysis quality (1-10 score)."""
        context_for_prompt = context.copy()
        context_for_prompt['answer_to_evaluate'] = analysis_to_evaluate
        prompt = EVAL_ANSWER_PROMPT.format(**context_for_prompt)
        messages = [{"role": "user", "content": prompt}]
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")

        raw_response = await self.get_completion(model=model_to_use, messages=messages)

        try:
            # Extract numbers, prioritize integers.
            # This regex finds integers or floats in the string.
            numbers = re.findall(r'\b\d+\b', raw_response) # Prioritize whole numbers
            if not numbers: # If no whole numbers, try to find any number including float
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_response)

            if numbers:
                score = int(round(float(numbers[0]))) # Take the first number found
                if 1 <= score <= 10:
                    return score
                else:
                    self.logger.warning(f"LLM evaluation score {score} out of range (1-10). Defaulting to 5. Raw: '{raw_response}'")
            else:
                self.logger.warning(f"Could not parse score from LLM evaluation response: '{raw_response}'. Defaulting to 5.")
        except ValueError:
            self.logger.warning(f"Could not convert score to int from LLM response: '{raw_response}'. Defaulting to 5.")
        return 5 # Default score

    async def generate_tags(self, analysis_text: str, config: Dict[str, Any]) -> List[str]:
        """Generates keyword tags for the analysis."""
        context = {"analysis_text": analysis_text}
        prompt = TAG_GENERATION_PROMPT.format(**context)
        messages = [{"role": "user", "content": prompt}]
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")

        raw_response = await self.get_completion(model=model_to_use, messages=messages)
        if raw_response:
            # Remove potential markdown list characters and split
            tags = [tag.strip().lstrip("-* ").rstrip(",.") for tag in raw_response.split(',')]
            # Filter out empty tags that might result from splitting
            return [tag for tag in tags if tag]
        self.logger.warning(f"Tag generation returned empty or invalid response: '{raw_response}'")
        return []

    async def synthesize_result(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a final synthesis based on the MCTS results."""
        prompt = FINAL_SYNTHESIS_PROMPT.format(**context)
        messages = [{"role": "user", "content": prompt}]
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")
        return await self.get_completion(model=model_to_use, messages=messages)

    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """Classifies user intent using the LLM."""
        # Context for intent classification typically just needs the raw input text
        context = {"raw_input_text": text_to_classify}
        prompt = INTENT_CLASSIFIER_PROMPT.format(**context)
        messages = [{"role": "user", "content": prompt}]
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")

        response = await self.get_completion(model=model_to_use, messages=messages)
        # Basic cleaning, specific adapters might need more
        return response.strip().upper().split()[0] if response else "UNKNOWN"

<<<<<<< HEAD
    async def generate_initial_analysis(self, question: str, config: Dict[str, Any]) -> str:
        """Generates initial analysis for a given question."""
        context = {"question": question}
        prompt = INITIAL_PROMPT.format(**context)
        messages = [{"role": "user", "content": prompt}]
        model_to_use = config.get("model_name") or self._kwargs.get("model_name")
        return await self.get_completion(model=model_to_use, messages=messages)

=======
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
"""
# Example usage (for testing, not part of the class itself)
if __name__ == '__main__':
    # This part would require a concrete implementation of BaseLLMAdapter
    # and an asyncio event loop to run.
    class MyAdapter(BaseLLMAdapter):
        async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str:
            # Mock implementation
            print(f"MyAdapter.get_completion called with model: {model}, messages: {messages}")
            if "evaluate_analysis" in messages[0]["content"]:
                return "This is a test evaluation. Score: 8/10"
            if "generate_tags" in messages[0]["content"]:
                return "tag1, tag2, tag3"
            return "This is a test completion."

        async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
            print(f"MyAdapter.get_streaming_completion called with model: {model}, messages: {messages}")
            yield "Stream chunk 1 "
            yield "Stream chunk 2"
            # Must include this for the method to be a valid async generator
            if False: # pragma: no cover
                 yield

    async def main():
        adapter = MyAdapter(model_name="default_test_model")

        # Test generate_thought
        thought_context = {
            "previous_best_summary": "Old summary", "unfit_markers_summary": "None",
            "learned_approach_summary": "Rational", "question_summary": "What is life?",
            "best_answer": "42", "best_score": "10", "current_sequence": "N1",
            "current_answer": "Deep thought", "current_tags": "philosophy"
        }
        thought = await adapter.generate_thought(thought_context, {})
        print(f"Generated thought: {thought}")

        # Test evaluate_analysis
        eval_score = await adapter.evaluate_analysis("Some analysis text", {"best_score": "7"}, {})
        print(f"Evaluation score: {eval_score}")

        # Test generate_tags
        tags = await adapter.generate_tags("Some text to tag.", {})
        print(f"Generated tags: {tags}")

    if False: # Keep example code from running automatically
       import asyncio
       asyncio.run(main())
"""
