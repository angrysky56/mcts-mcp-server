#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama LLM Adapter for MCTS
==========================

This module adapts the Ollama API to the LLMInterface
required by the MCTS implementation.
"""
import asyncio
import re
import logging
import json
from typing import List, Dict, Any, AsyncGenerator, Optional

try:
    import ollama
except ImportError:
    raise ImportError("The ollama package is required. Install it with `pip install ollama`")

# Import the LLMInterface protocol - try different import paths
try:
    # Direct import (recommended)
    from llm_adapter import LLMInterface
except ImportError:
    try:
        # Package import
        from mcts_mcp_server.llm_adapter import LLMInterface
    except ImportError:
        # Load dynamically
        import os
        import sys
        import importlib.util
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        llm_adapter_path = os.path.join(current_dir, "llm_adapter.py")
        
        if os.path.exists(llm_adapter_path):
            spec = importlib.util.spec_from_file_location("llm_adapter", llm_adapter_path)
            llm_adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_adapter)
            LLMInterface = llm_adapter.LLMInterface
        else:
            raise ImportError(f"Could not find llm_adapter.py at {llm_adapter_path}")

logger = logging.getLogger("ollama_adapter")

class OllamaAdapter(LLMInterface):
    """
    LLM adapter that connects to local Ollama models.
    """

    def __init__(self, model_name="llama3", host="http://localhost:11434", mcp_server=None):
        """
        Initialize the adapter.
        
        Args:
            model_name: Name of the Ollama model to use
            host: URL of the Ollama server
            mcp_server: Optional MCP server instance (not used directly)
        """
        self.model_name = model_name
        self.host = host
        self.mcp_server = mcp_server
        self._client = None
        logger.info(f"Initialized OllamaAdapter with model {model_name} on {host}")
        
        # Test connection
        try:
            import httpx
            client = httpx.Client(base_url=host, timeout=5.0)
            response = client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in available models. You may need to pull it.")
            else:
                logger.warning(f"Failed to get models from Ollama: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {host}: {e}")

    async def get_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Gets a non-streaming completion from the Ollama model."""
        try:
            # Extract the relevant content from messages
            if not messages:
                return "No input provided."
            
            # Check if it's a chat completion or regular completion
            if len(messages) > 1 or messages[0].get("role") in ["user", "system", "assistant"]:
                # Use chat API for chat-style prompts
                response = await asyncio.to_thread(
                    ollama.chat,
                    model=model or self.model_name,
                    messages=messages,
                    stream=False
                )
                content = response.get("message", {}).get("content", "")
            else:
                # Use generate API for standard prompts
                prompt = messages[0].get("content", "")
                response = await asyncio.to_thread(
                    ollama.generate,
                    model=model or self.model_name,
                    prompt=prompt,
                    stream=False
                )
                content = response.get("response", "")
            
            # Clean content of <think> tags if present
            if content and "<think>" in content:
                # First try to remove entire <think> blocks
                clean_attempt = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                # If that removes everything, keep the original but strip just the tags
                if not clean_attempt.strip():
                    content = re.sub(r'</?think>', '', content)
                else:
                    content = clean_attempt
            
            return content
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            return f"Error: {str(e)}"

    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Gets a streaming completion from the Ollama model."""
        try:
            # Similar logic to get_completion but with streaming
            if not messages:
                yield "No input provided."
                return
            
            if len(messages) > 1 or messages[0].get("role") in ["user", "system", "assistant"]:
                # Use chat API
                response_stream = ollama.chat(
                    model=model or self.model_name,
                    messages=messages,
                    stream=True
                )
                
                for chunk in response_stream:
                    yield chunk.get("message", {}).get("content", "")
            else:
                # Use generate API
                prompt = messages[0].get("content", "")
                response_stream = ollama.generate(
                    model=model or self.model_name,
                    prompt=prompt,
                    stream=True
                )
                
                for chunk in response_stream:
                    yield chunk.get("response", "")
        except Exception as e:
            logger.error(f"Error in get_streaming_completion: {e}")
            yield f"Error: {str(e)}"

    async def generate_thought(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a critical thought or new direction based on context."""
        try:
            from mcts_core import THOUGHTS_PROMPT
            
            # Format the prompt with context
            prompt = THOUGHTS_PROMPT.format(**context)
            
            # Send to Ollama
            messages = [{"role": "user", "content": prompt}]
            return await self.get_completion(self.model_name, messages)
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}")
            return f"Error generating thought: {str(e)}"

    async def update_analysis(self, critique: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Revises analysis based on critique and context."""
        try:
            from mcts_core import UPDATE_PROMPT
            
            # Add the critique to the context
            context["improvements"] = critique
            
            # Format the prompt
            prompt = UPDATE_PROMPT.format(**context)
            
            # Send to Ollama
            messages = [{"role": "user", "content": prompt}]
            return await self.get_completion(self.model_name, messages)
        except Exception as e:
            logger.error(f"Error in update_analysis: {e}")
            return f"Error updating analysis: {str(e)}"

    async def evaluate_analysis(self, analysis_to_evaluate: str, context: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Evaluates analysis quality (1-10 score)."""
        try:
            from mcts_core import EVAL_ANSWER_PROMPT
            
            # Add the analysis to the context
            context['answer_to_evaluate'] = analysis_to_evaluate
            
            # Format the prompt
            prompt = EVAL_ANSWER_PROMPT.format(**context)
            
            # Send to Ollama
            messages = [{"role": "user", "content": prompt}]
            response = await self.get_completion(self.model_name, messages)
            
            # Extract the numeric score from the response
            score_matches = re.findall(r'\b([0-9]|10)\b', response)
            if score_matches:
                # Get the first numeric match that could be a score
                try:
                    score = int(score_matches[0])
                    # Ensure score is in range 1-10
                    return max(1, min(10, score))
                except ValueError:
                    logger.warning(f"Could not parse score from response: '{response}'. Defaulting to 5.")
                    return 5
            else:
                logger.warning(f"No score found in response: '{response}'. Defaulting to 5.")
                return 5
        except Exception as e:
            logger.warning(f"Error in evaluate_analysis: {e}. Defaulting to 5.")
            return 5

    async def generate_tags(self, analysis_text: str, config: Dict[str, Any]) -> List[str]:
        """Generates keyword tags for the analysis."""
        try:
            from mcts_core import TAG_GENERATION_PROMPT
            
            # Format the prompt
            prompt = TAG_GENERATION_PROMPT.format(analysis_text=analysis_text)
            
            # Send to Ollama
            messages = [{"role": "user", "content": prompt}]
            response = await self.get_completion(self.model_name, messages)
            
            # Parse the response into tags
            # Clean and parse the response
            response = response.strip()
            # Split by commas or newlines
            if ',' in response:
                tags = [tag.strip() for tag in response.split(',')]
            else:
                tags = [tag.strip() for tag in response.split('\n')]
            
            # Filter out empty tags and normalize
            tags = [tag.lower() for tag in tags if tag]
            
            # Truncate to a maximum of 5 tags
            return tags[:5]
        except Exception as e:
            logger.error(f"Error in generate_tags: {e}")
            return ["error"]

    async def synthesize_result(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a final synthesis based on the MCTS results."""
        try:
            from mcts_core import FINAL_SYNTHESIS_PROMPT
            
            # Format the prompt
            prompt = FINAL_SYNTHESIS_PROMPT.format(**context)
            
            # Send to Ollama
            messages = [{"role": "user", "content": prompt}]
            return await self.get_completion(self.model_name, messages)
        except Exception as e:
            logger.error(f"Error in synthesize_result: {e}")
            return f"Error synthesizing result: {str(e)}"

    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """Classifies user intent using the LLM."""
        try:
            from mcts_core import INTENT_CLASSIFIER_PROMPT
            
            # Format the prompt
            prompt = INTENT_CLASSIFIER_PROMPT.format(raw_input_text=text_to_classify)
            
            # Send to Ollama
            messages = [{"role": "user", "content": prompt}]
            response = await self.get_completion(self.model_name, messages)
            
            # Extract the classification from the response
            # Look for any of the expected intents in the response
            intent_types = [
                "ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY",
                "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"
            ]
            
            for intent in intent_types:
                if intent in response.upper():
                    return intent
            
            # Default to ANALYZE_NEW if we couldn't find a match
            logger.warning(f"Could not determine intent from response: '{response}'. Defaulting to ANALYZE_NEW.")
            return "ANALYZE_NEW"
        except Exception as e:
            logger.error(f"Error in classify_intent: {e}")
            return "ANALYZE_NEW"  # Default on error
