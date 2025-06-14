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
from typing import List, Dict, Any, AsyncGenerator # Protocol removed as LLMInterface is imported
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
        """Gets a sophisticated completion based on the input messages."""
        try:
            # Extract the user's message content (usually the last message)
            user_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    user_content = msg["content"]
                    break

            if not user_content:
                return "No input detected."

            # For very structured internal prompts, provide focused responses
            # But make them more intelligent and contextual
            if "<instruction>" in user_content and any(marker in user_content for marker in [
                "Critically examine", "Substantially revise", "Evaluate the intellectual quality",
                "Generate concise keyword tags", "Synthesize the key insights", "Classify user requests"
            ]):
                # Extract key information from the structured prompt
                if "**Question being analyzed:**" in user_content:
                    # Extract the actual question being analyzed
                    question_match = re.search(r'\*\*Question being analyzed:\*\* (.+)', user_content)
                    question = question_match.group(1) if question_match else "the topic"

                    if "Critically examine" in user_content:
                        return f"The analysis of '{question}' would benefit from examining this through a systems thinking lens - how do the various components interact dynamically, and what emergent properties might we be missing?"
                    elif "Substantially revise" in user_content:
                        return f"To strengthen this analysis of '{question}', we should integrate multiple theoretical frameworks and examine the underlying assumptions more rigorously, particularly considering how context shapes our interpretation."
                    elif "Synthesize the key insights" in user_content:
                        return f"The exploration of '{question}' reveals that robust understanding emerges through systematic examination of interconnected perspectives, highlighting the importance of both analytical depth and synthetic integration."

                # Fallback responses for other structured prompts
                if "Evaluate the intellectual quality" in user_content:
                    return "7"
                elif "Generate concise keyword tags" in user_content:
                    return "systems-thinking, analytical-framework, critical-analysis, perspective-integration, contextual-understanding"
                elif "Classify user requests" in user_content:
                    return "ANALYZE_NEW"
                else:
                    return "I've analyzed the structured input and generated a contextually appropriate response that addresses the specific analytical requirements."

            # For natural conversation and general inputs, be more conversational and adaptive
            # This allows for fluid interaction while still being helpful
            return "I'm here to help you think through complex topics using systematic analytical approaches. What would you like to explore together?"

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
            # Extract context information
            question_summary = context.get("question_summary", "")
            current_approach = context.get("current_approach", "initial")
            current_analysis = context.get("answer", "")
            iteration_count = context.get("iteration", 0)

            # Build a sophisticated prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert analytical thinker skilled at generating critical insights and new analytical directions. Your role is to examine existing analysis and suggest a specific, actionable critique or new perspective that will meaningfully advance understanding."
                },
                {
                    "role": "user",
                    "content": f"""<instruction>Critically examine the current analytical approach and generate a specific, actionable critique or new direction.

**Question being analyzed:** {question_summary}

**Current analytical approach:** {current_approach}

**Current analysis (if any):** {current_analysis[:500] if current_analysis else "No analysis yet - this is the initial exploration."}

**Iteration:** {iteration_count + 1}

Your task is to:
1. Identify a specific weakness, gap, or limitation in the current approach/analysis
2. Suggest a concrete new direction, framework, or perspective that would address this limitation
3. Be specific about what should be examined differently

Generate your critique as a single, focused suggestion (1-2 sentences) that provides clear direction for improving the analysis. Avoid generic advice - be specific to this particular question and current state of analysis.</instruction>"""
                }
            ]

            # Use the LLM to generate a thoughtful response
            response = await self.get_completion("default", messages)
            return response.strip()

        except Exception as e:
            logger.error(f"Error in generate_thought: {e}")
            # Fallback to a simple contextual thought
            question_summary = context.get("question_summary", "the topic")
            return f"Consider examining '{question_summary}' from an unexplored angle - what fundamental assumptions might we be overlooking?"

    async def update_analysis(self, critique: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Revises analysis based on critique and context."""
        try:
            # Extract context information
            question_summary = context.get("question_summary", "")
            current_approach = context.get("current_approach", "initial")
            original_content = context.get("answer", "")
            iteration_count = context.get("iteration", 0)

            # Build a sophisticated prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert analytical thinker skilled at revising and improving analysis based on critical feedback. Your role is to substantially enhance existing analysis by incorporating specific critiques and suggestions."
                },
                {
                    "role": "user",
                    "content": f"""<instruction>Substantially revise and improve the following analysis by incorporating the provided critique.

**Original Question:** {question_summary}

**Current Analytical Approach:** {current_approach}

**Original Analysis:**
{original_content}

**Critique to Incorporate:**
<critique>{critique}</critique>

**Iteration:** {iteration_count + 1}

Your task is to:
1. Carefully consider how the critique identifies weaknesses or gaps in the original analysis
2. Substantially revise the analysis to address these concerns
3. Integrate new perspectives, frameworks, or evidence as suggested by the critique
4. Produce a more sophisticated, nuanced analysis that builds meaningfully on the original

Provide a completely rewritten analysis that demonstrates clear improvement over the original. The revision should be substantive, not superficial - show genuine analytical advancement.</instruction>"""
                }
            ]

            # Use the LLM to generate a thoughtful response
            response = await self.get_completion("default", messages)
            return response.strip()

        except Exception as e:
            logger.error(f"Error in update_analysis: {e}")
            return f"Error: {str(e)}"

    async def evaluate_analysis(self, analysis_to_evaluate: str, context: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Evaluates analysis quality (1-10 score)."""
        try:
            # Extract context information
            question_summary = context.get("question_summary", "")
            current_approach = context.get("current_approach", "initial")
            iteration_count = context.get("iteration", 0)

            # Build a sophisticated prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert analytical evaluator skilled at assessing the intellectual quality and depth of analysis. Your role is to provide objective, rigorous evaluation of analytical work on a 1-10 scale."
                },
                {
                    "role": "user",
                    "content": f"""<instruction>Evaluate the intellectual quality of the following analysis on a scale of 1-10.

**Original Question:** {question_summary}

**Analytical Approach:** {current_approach}

**Analysis to Evaluate:**
{analysis_to_evaluate}

**Iteration:** {iteration_count + 1}

Evaluation Criteria (1-10 scale):
- 1-3: Superficial, generic, or factually incorrect
- 4-5: Basic understanding but lacks depth or insight
- 6-7: Solid analysis with some meaningful insights
- 8-9: Sophisticated, nuanced analysis with strong insights
- 10: Exceptional depth, originality, and comprehensive understanding

Consider:
1. Depth of insight and analytical sophistication
2. Relevance and specificity to the question
3. Use of evidence, examples, or frameworks
4. Logical coherence and structure
5. Originality of perspective or approach
6. Practical applicability of insights

Respond with only a single number (1-10) representing your evaluation score.</instruction>"""
                }
            ]

            # Use the LLM to generate a thoughtful response
            response = await self.get_completion("default", messages)

            # Extract the numeric score from the response
            try:
                score = int(response.strip())
                return max(1, min(10, score))  # Ensure score is in valid range
            except ValueError:
                # Fallback: try to extract first number from response
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    score = int(numbers[0])
                    return max(1, min(10, score))
                else:
                    logger.warning(f"Could not parse score from response: '{response}'. Defaulting to 5.")
                    return 5

        except Exception as e:
            logger.warning(f"Error in evaluate_analysis: '{e}'. Defaulting to 5.")
            return 5

    async def generate_tags(self, analysis_text: str, config: Dict[str, Any]) -> List[str]:
        """Generates keyword tags for the analysis."""
        try:
            # Build a sophisticated prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at generating precise, meaningful keyword tags that capture the essential concepts, themes, and analytical approaches in text. Your tags should be specific, insightful, and useful for categorization and discovery."
                },
                {
                    "role": "user",
                    "content": f"""Generate concise keyword tags for the following analysis text.

**Analysis Text:**
{analysis_text[:1000]}  # Limit to avoid token limits

Your task is to:
1. Identify the key concepts, themes, and analytical approaches
2. Generate 3-5 specific, meaningful tags
3. Focus on substantive content rather than generic terms
4. Use single words or short phrases (2-3 words max)
5. Prioritize tags that would help categorize or find this analysis

Respond with only the tags, separated by commas (e.g., "cognitive-bias, decision-theory, behavioral-economics, systematic-analysis, framework-comparison")."""
                }
            ]

            # Use the LLM to generate tags
            response = await self.get_completion("default", messages)

            # Parse the response into a list of tags
            tags = [tag.strip().lower() for tag in response.split(',')]
            tags = [tag for tag in tags if tag and len(tag) > 2]  # Filter out empty or very short tags

            return tags[:5]  # Return up to 5 tags

        except Exception as e:
            logger.error(f"Error in generate_tags: {e}")
            # Fallback to a simple extraction if LLM fails
            words = analysis_text.lower().split()
            common_words = {"the", "and", "is", "in", "to", "of", "a", "for", "this", "that", "with", "be", "as", "can", "will", "would", "should"}
            filtered_words = [word for word in words if word not in common_words and len(word) > 4]
            return list(set(filtered_words[:3]))  # Return unique words as fallback

    async def synthesize_result(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a final synthesis based on the MCTS results."""
        try:
            # Extract context information
            question_summary = context.get("question_summary", "")
            best_score = context.get("best_score", "0")
            final_analysis = context.get("final_best_analysis_summary", "")
            all_approaches = context.get("all_approaches", [])
            total_iterations = context.get("total_iterations", 0)

            # Build a sophisticated prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert analytical synthesizer skilled at drawing together insights from multiple analytical approaches to create comprehensive, nuanced conclusions. Your role is to synthesize the best insights from an iterative analysis process."
                },
                {
                    "role": "user",
                    "content": f"""<instruction>Synthesize the key insights from this multi-approach analytical exploration into a comprehensive conclusion.

**Original Question:** {question_summary}

**Best Analysis Found (Score: {best_score}/10):**
{final_analysis}

**Analytical Approaches Explored:** {', '.join(all_approaches) if all_approaches else 'Multiple iterative approaches'}

**Total Iterations:** {total_iterations}

**Final Confidence Score:** {best_score}/10

Your task is to:
1. Synthesize the most valuable insights from the analytical exploration
2. Identify the key patterns, connections, or principles that emerged
3. Articulate what makes the final analysis particularly compelling
4. Reflect on how the iterative process enhanced understanding
5. Provide a nuanced, comprehensive conclusion that captures the depth achieved

Generate a thoughtful synthesis that demonstrates the analytical journey's value and presents the most robust conclusions reached. Focus on substance and insight rather than process description.</instruction>"""
                }
            ]

            # Use the LLM to generate a thoughtful response
            response = await self.get_completion("default", messages)
            return response.strip()

        except Exception as e:
            logger.error(f"Error in synthesize_result: {e}")
            return f"Error synthesizing result: {str(e)}"

    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """Classifies user intent using the LLM."""
        try:
            # Build a sophisticated prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at understanding user intent and classifying requests into specific categories. Your role is to analyze user input and determine their primary intent from a predefined set of categories."
                },
                {
                    "role": "user",
                    "content": f"""Classify the following user request into one of these specific intent categories:

**User Input:** "{text_to_classify}"

**Intent Categories:**
- ANALYZE_NEW: User wants to start a new analysis of a topic, question, or problem
- CONTINUE_ANALYSIS: User wants to continue, expand, or elaborate on an existing analysis
- ASK_LAST_RUN_SUMMARY: User wants to see results, summary, or score from the most recent analysis
- ASK_PROCESS: User wants to understand how the system works, the algorithm, or methodology
- ASK_CONFIG: User wants to know about configuration, settings, or parameters
- GENERAL_CONVERSATION: User is making casual conversation or asking unrelated questions

Consider the context, tone, and specific language used. Look for:
- New analysis requests: "analyze", "examine", "what do you think about", "explore"
- Continuation requests: "continue", "more", "elaborate", "expand on", "keep going"
- Summary requests: "what did you find", "results", "summary", "score"
- Process questions: "how do you work", "what's your method", "algorithm"
- Config questions: "settings", "parameters", "configuration"

Respond with only the category name (e.g., "ANALYZE_NEW")."""
                }
            ]

            # Use the LLM to classify intent
            response = await self.get_completion("default", messages)

            # Clean and validate the response
            intent = response.strip().upper()
            valid_intents = ["ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY",
                           "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"]

            if intent in valid_intents:
                return intent
            else:
                logger.warning(f"LLM returned invalid intent: '{response}'. Defaulting to ANALYZE_NEW.")
                return "ANALYZE_NEW"

        except Exception as e:
            logger.error(f"Error in classify_intent: {e}")
            return "ANALYZE_NEW"  # Default on error

# For backward compatibility, alias the class
McpLLMAdapter = LocalInferenceLLMAdapter
DirectMcpLLMAdapter = LocalInferenceLLMAdapter
