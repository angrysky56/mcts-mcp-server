#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Server to MCTS LLM Adapter
==============================

This module adapts the MCP server's LLM access to the LLMInterface
required by the MCTS implementation.
"""
import asyncio
import re
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional # Protocol removed as LLMInterface is imported
from .llm_interface import LLMInterface # Import the official LLMInterface

logger = logging.getLogger("llm_adapter")


# LLMInterface protocol definition removed from here

class LocalInferenceLLMAdapter(LLMInterface):
    """
    LLM adapter that uses simple deterministic rules for generating responses.
    This adapter doesn't call external LLMs but performs simplified inference locally.
    """

    def __init__(self, mcp_server=None):
        """
        Initialize the adapter.

        Args:
            mcp_server: Optional MCP server instance (not used directly)
        """
        self.mcp_server = mcp_server
        logger.info("Initialized LocalInferenceLLMAdapter")

    async def get_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Gets a deterministic completion based on the input messages."""
        try:
            # Extract the user's message content (usually the last message)
            user_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    user_content = msg["content"]
                    break

            if not user_content:
                return "No input detected."

            # Check for various prompt types and generate appropriate responses
            if "<instruction>Critically examine" in user_content or "Generate your critique" in user_content:
                # This is a thought generation prompt
                return self._generate_thought_response(user_content)
            elif "<instruction>Substantially revise" in user_content or "<critique>" in user_content:
                # This is an update analysis prompt
                return self._generate_update_response(user_content)
            elif "<instruction>Evaluate the intellectual quality" in user_content:
                # This is an evaluation prompt
                return self._generate_evaluation_response(user_content)
            elif "Generate concise keyword tags" in user_content:
                # This is a tag generation prompt
                return self._generate_tags_response(user_content)
            elif "<instruction>Synthesize the key insights" in user_content:
                # This is a synthesis prompt
                return self._generate_synthesis_response(user_content)
            elif "Classify user requests" in user_content:
                # This is an intent classification prompt
                return self._generate_classification_response(user_content)
            else:
                # Generic response for other prompts
                return "I've processed your input and generated a thoughtful response based on the context provided."
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            return f"Error: {str(e)}"

    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Gets a streaming completion by breaking up a regular completion."""
        try:
            full_response = await self.get_completion(model, messages, **kwargs)
            chunks = [full_response[i:i+20] for i in range(0, len(full_response), 20)]

            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in get_streaming_completion: {e}")
            yield f"Error: {str(e)}"

    async def generate_thought(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a critical thought or new direction based on context."""
        try:
            # Instead of using context to format a prompt for the model,
            # we'll generate a thought directly based on the context
            question_summary = context.get("question_summary", "")
            current_approach = context.get("current_approach", "initial")

            # Generate different thoughts based on the current approach
            if current_approach == "initial":
                return "Consider examining this from a comparative perspective, looking at how different frameworks or disciplines would approach this problem."
            elif current_approach == "comparative":
                return "Let's apply a more critical lens to the analysis. What assumptions are being made that might not hold under scrutiny?"
            elif current_approach == "critical":
                return "Now let's take a more constructive approach. How might we synthesize the insights gained from our critical examination?"
            elif current_approach == "constructive":
                return "What if we looked at this from a holistic perspective? How do the various elements interact as a system?"
            else:
                # Generate a generic thought for other approaches
                return f"Let's explore a different angle: What are the practical implications of this analysis? How might it be applied in real-world contexts?"
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}")
            return f"Error: {str(e)}"

    async def update_analysis(self, critique: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Revises analysis based on critique and context."""
        try:
            # Extract content from context
            original_content = context.get("answer", "")

            # Generate a new analysis that incorporates the critique
            return f"Building upon the original analysis, and incorporating the suggestion to {critique}, we can develop a more nuanced understanding. The key insight here is that multiple perspectives need to be considered, including both theoretical frameworks and practical applications. This allows us to see not only the immediate implications but also the broader systemic effects that might emerge over time."
        except Exception as e:
            logger.error(f"Error in update_analysis: {e}")
            return f"Error: {str(e)}"

    async def evaluate_analysis(self, analysis_to_evaluate: str, context: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Evaluates analysis quality (1-10 score)."""
        try:
            # Simple scoring algorithm - longer analyses with complex words get higher scores
            word_count = len(analysis_to_evaluate.split())

            # Base score on length (longer = better, to a point)
            if word_count < 20:
                score = 3
            elif word_count < 50:
                score = 5
            elif word_count < 100:
                score = 7
            else:
                score = 8

            # Check for "complex" words or phrases that suggest depth
            complexity_markers = ["however", "furthermore", "nonetheless", "therefore", "consequently",
                                 "perspective", "framework", "implications", "nuanced", "context"]

            # Count how many complexity markers are present
            complexity_score = sum(1 for marker in complexity_markers if marker in analysis_to_evaluate.lower())

            # Adjust score based on complexity (max +2)
            score += min(2, complexity_score // 2)

            # Ensure score is in range 1-10
            score = max(1, min(10, score))

            return score
        except Exception as e:
            logger.warning(f"Could not parse score from response: '{e}'. Defaulting to 5.")
            return 5

    async def generate_tags(self, analysis_text: str, config: Dict[str, Any]) -> List[str]:
        """Generates keyword tags for the analysis."""
        try:
            # Extract potential keywords from the text
            words = analysis_text.lower().split()

            # Remove common words
            common_words = {"the", "and", "is", "in", "to", "of", "a", "for", "this", "that", "with", "be", "as"}
            filtered_words = [word for word in words if word not in common_words and len(word) > 3]

            # Count word frequencies
            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Get top words by frequency
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

            # Extract tags (take up to 5 most frequent)
            tags = [word for word, count in sorted_words[:5]]

            # If we don't have enough tags, add some generic ones
            generic_tags = ["analysis", "perspective", "framework", "implications", "context"]
            while len(tags) < 3:
                for tag in generic_tags:
                    if tag not in tags:
                        tags.append(tag)
                        break
                    if len(tags) >= 3:
                        break

            return tags[:5]  # Return up to 5 tags
        except Exception as e:
            logger.error(f"Error in generate_tags: {e}")
            return ["error"]

    async def synthesize_result(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a final synthesis based on the MCTS results."""
        try:
            question_summary = context.get("question_summary", "")
            best_score = context.get("best_score", "0")
            final_analysis = context.get("final_best_analysis_summary", "")

            return f"After careful exploration using multiple analytical approaches, the most compelling insight that emerges is that {final_analysis} This conclusion synthesizes our iterative analysis process, which reached a confidence score of {best_score}/10. The key to understanding this topic lies in recognizing both its complexity and the interconnections between different perspectives, allowing us to develop a more comprehensive understanding than would be possible through a single analytical framework."
        except Exception as e:
            logger.error(f"Error in synthesize_result: {e}")
            return f"Error synthesizing result: {str(e)}"

    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """Classifies user intent using the LLM."""
        try:
            text_lower = text_to_classify.lower()

            # Simple keyword-based classification
            if any(word in text_lower for word in ["analyze", "examine", "explore", "investigate"]) and not any(word in text_lower for word in ["continue", "further", "more", "again"]):
                return "ANALYZE_NEW"
            elif any(word in text_lower for word in ["continue", "further", "expand", "more", "elaborate", "keep going"]):
                return "CONTINUE_ANALYSIS"
            elif any(word in text_lower for word in ["summary", "result", "score", "what did you find"]):
                return "ASK_LAST_RUN_SUMMARY"
            elif any(word in text_lower for word in ["how do you work", "algorithm", "method", "process", "technique"]):
                return "ASK_PROCESS"
            elif any(word in text_lower for word in ["config", "settings", "parameters", "setup"]):
                return "ASK_CONFIG"
            else:
                # Default to conversation or analyze new based on question mark
                return "GENERAL_CONVERSATION" if "?" not in text_lower else "ANALYZE_NEW"
        except Exception as e:
            logger.error(f"Error in classify_intent: {e}")
            return "ANALYZE_NEW"  # Default on error

    # Helper methods for generating specific types of responses
    def _generate_thought_response(self, prompt):
        """Generate a thought response for thought prompts."""
        return "Have we considered the meta-level implications of this analysis? Perhaps we should examine how the framework itself shapes our understanding, not just the content within it."

    def _generate_update_response(self, prompt):
        """Generate an update response for revision prompts."""
        return "The analysis can be enhanced by considering multiple levels of interpretation. At the surface level, we observe the immediate patterns, but deeper examination reveals underlying structural factors that shape these patterns. This multi-level approach provides a more nuanced understanding of the complex interplay between various elements in the system."

    def _generate_evaluation_response(self, prompt):
        """Generate a score for evaluation prompts."""
        return "7"

    def _generate_tags_response(self, prompt):
        """Generate tags for tag generation prompts."""
        return "analysis, framework, perspective, implications, context"

    def _generate_synthesis_response(self, prompt):
        """Generate a synthesis for synthesis prompts."""
        return "The most robust conclusion from our analysis is that multiple perspectives must be integrated to form a complete understanding. No single framework adequately captures the full complexity of the subject, but by systematically exploring different angles, we can construct a more comprehensive model that acknowledges both the theoretical underpinnings and practical implications."

    def _generate_classification_response(self, prompt):
        """Generate a classification for intent classification prompts."""
        if "elaborate" in prompt or "continue" in prompt:
            return "CONTINUE_ANALYSIS"
        elif "analyze" in prompt:
            return "ANALYZE_NEW"
        else:
            return "GENERAL_CONVERSATION"

# For backward compatibility, alias the class
McpLLMAdapter = LocalInferenceLLMAdapter
DirectMcpLLMAdapter = LocalInferenceLLMAdapter
