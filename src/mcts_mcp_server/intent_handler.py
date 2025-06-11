#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Handler for MCTS
=======================

This module defines the IntentHandler class, related data structures,
and prompts for classifying and handling user intents.
"""
import logging
import json
import re
import os # Moved here
from collections import namedtuple
from typing import Dict, Any, Optional, List

# DEFAULT_CONFIG is now in mcts_config
from .mcts_config import DEFAULT_CONFIG
from .llm_interface import LLMInterface # Moved from mcts_core
from .state_manager import StateManager

# Setup logger for this module
logger = logging.getLogger(__name__)

# ==============================================================================
# Prompts (Moved from mcts_core.py)
# ==============================================================================

INITIAL_PROMPT = """<instruction>Provide an initial analysis and interpretation of the core themes, arguments, and potential implications presented. Identify key concepts. Respond with clear, natural language text ONLY.</instruction><question>{question}</question>"""

THOUGHTS_PROMPT = """<instruction>Critically examine the current analysis below. Suggest a SIGNIFICANTLY DIFFERENT interpretation, identify a MAJOR underlying assumption or weakness, or propose a novel connection to another domain or concept. Push the thinking in a new direction.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}
- Learned Approach Preferences: {learned_approach_summary}

Consider this context. Avoid repeating unfit areas unless you have a novel mutation. Build upon previous success if appropriate, or diverge strongly if needed.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis (Node {current_sequence}): {current_answer}\nCurrent Analysis Tags: {current_tags}</context>
Generate your critique or alternative direction.</instruction>"""

UPDATE_PROMPT = """<instruction>Substantially revise the draft analysis below to incorporate the core idea from the critique. Develop the analysis further based on this new direction.

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Ensure the revision considers past findings and avoids known unproductive paths unless the critique justifies revisiting them.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nCurrent Analysis Tags: {current_tags}</context>
<draft>{answer}</draft>
<critique>{improvements}</critique>
Write the new, revised analysis text."""

EVAL_ANSWER_PROMPT = """<instruction>Evaluate the intellectual quality and insightfulness of the analysis below (1-10) concerning the original input. Higher scores for depth, novelty, and relevance. Use the full 1-10 scale. Reserve 9-10 for truly exceptional analyses that significantly surpass previous best analysis ({best_score}/10).

**Previous Run Context (If Available):**
- Previous Best Analysis Summary: {previous_best_summary}
- Previously Marked Unfit Concepts/Areas: {unfit_markers_summary}

Consider if this analysis productively builds upon or diverges from past findings.</instruction>
<context>Original Text Summary: {question_summary}\nBest Overall Analysis (Score {best_score}/10): {best_answer}\nAnalysis Tags: {current_tags}</context>
<answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate>
How insightful, deep, relevant, and well-developed is this analysis compared to the best so far? Does it offer a genuinely novel perspective or intelligently navigate known issues? Rate 1-10 based purely on merit. Respond with a logical rating from 1 to 10.</instruction>"""

TAG_GENERATION_PROMPT = """<instruction>Generate concise keyword tags summarizing the main concepts in the following text. Output the tags, separated by commas.</instruction>\n<text_to_tag>{analysis_text}</text_to_tag>"""

FINAL_SYNTHESIS_PROMPT = """<instruction>Synthesize the key insights developed along the primary path of analysis below into a conclusive statement addressing the original question. Focus on the progression of ideas represented.</instruction>
<original_question_summary>{question_summary}</original_question_summary>
<initial_analysis>{initial_analysis_summary}</initial_analysis>
<best_analysis_score>{best_score}/10</best_analysis_score>
<development_path>
{path_thoughts}
</development_path>
<final_best_analysis>{final_best_analysis_summary}</final_best_analysis>
Synthesize into a final conclusion:"""

INTENT_CLASSIFIER_PROMPT = """Classify user requests. Choose the *single best* category from the list. Respond appropriately.

Categories:
- CONTINUE_ANALYSIS: User wants to continue, refine, or build upon the previous MCTS analysis run (e.g., "elaborate", "explore X further", "what about Y?").
- ANALYZE_NEW: User wants a fresh MCTS analysis on a new provided text/task, ignoring any previous runs in this chat.
- ASK_LAST_RUN_SUMMARY: User asks about the outcome, score, or details of the previous MCTS run (e.g., "what was the score?", "summarize last run").
- ASK_PROCESS: User asks how the analysis process works (e.g., "how do you work?", "what algorithm is this?").
- ASK_CONFIG: User asks about the current MCTS parameters or settings (e.g., "show config", "what are the settings?").
- GENERAL_CONVERSATION: The input is conversational, off-topic, or a simple greeting/closing.

User Input:
"{raw_input_text}"

Classification:
"""

# ==============================================================================
# Intent Handling Structures (Moved from mcts_core.py)
# ==============================================================================

# Define result structures for handlers
IntentResult = namedtuple("IntentResult", ["type", "data"]) # type = 'message', 'error', 'mcts_params'

class IntentHandler:
    """Handles different user intents based on classification."""

    def __init__(self, llm_interface: LLMInterface, state_manager: StateManager, config: Optional[Dict[str, Any]] = None):
        self.llm = llm_interface
        self.state_manager = state_manager
        # Use provided config or fall back to DEFAULT_CONFIG if None
        self.config = config if config is not None else DEFAULT_CONFIG.copy()


    async def classify_intent(self, user_input: str) -> str:
        """Classifies user intent using the LLM."""
        # Use the LLM interface's classification method
        try:
            # Ensure config is passed to the LLM call if it expects it
            classification_result = await self.llm.classify_intent(user_input, self.config)
            # Basic validation against known intents
            valid_intents = [
                "ANALYZE_NEW", "CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY",
                "ASK_PROCESS", "ASK_CONFIG", "GENERAL_CONVERSATION"
            ]
            clean_result = classification_result.strip().upper().split()[0] if classification_result else ""
            clean_result = re.sub(r"[.,!?;:]$", "", clean_result) # Remove trailing punctuation

            if clean_result in valid_intents:
                logger.info(f"Intent classified as: {clean_result}")
                return clean_result
            else:
                logger.warning(f"Intent classification returned unexpected result: '{classification_result}'. Defaulting to ANALYZE_NEW.")
                # Simple keyword check as fallback
                if any(keyword in user_input.lower() for keyword in ["continue", "elaborate", "further", "what about", "refine"]):
                    logger.info("Input text suggests continuation despite classification. Setting intent to CONTINUE_ANALYSIS.")
                    return "CONTINUE_ANALYSIS"
                return "ANALYZE_NEW"
        except Exception as e:
             logger.error(f"Intent classification LLM call failed: {e}", exc_info=True)
             return "ANALYZE_NEW" # Default on error

    async def handle_intent(self, intent: str, user_input: str, chat_id: Optional[str]) -> IntentResult:
        """Dispatches to the appropriate handler based on intent."""

        loaded_state = None
        if chat_id and intent in ["CONTINUE_ANALYSIS", "ASK_LAST_RUN_SUMMARY"]:
             loaded_state = self.state_manager.load_state(chat_id)
             if intent == "CONTINUE_ANALYSIS" and not loaded_state:
                  logger.info(f"Cannot continue analysis for chat {chat_id}: No state found. Switching to ANALYZE_NEW.")
                  intent = "ANALYZE_NEW"

        if intent == "ASK_PROCESS":
            return await self.handle_ask_process()
        elif intent == "ASK_CONFIG":
            return await self.handle_ask_config()
        elif intent == "ASK_LAST_RUN_SUMMARY":
             if not loaded_state:
                  return IntentResult(type='message', data="I don't have any saved results from a previous analysis run in this chat session.")
             return await self.handle_ask_last_run_summary(loaded_state)
        elif intent == "GENERAL_CONVERSATION":
            return await self.handle_general_conversation()
        elif intent == "ANALYZE_NEW":
             return IntentResult(type='mcts_params', data={'question': user_input, 'initial_state': None})
        elif intent == "CONTINUE_ANALYSIS":
            if not loaded_state:
                 logger.error("CONTINUE_ANALYSIS intent but no loaded state. This shouldn't happen.")
                 return IntentResult(type='error', data="Internal error: Cannot continue without loaded state.")
            return IntentResult(type='mcts_params', data={'question': user_input, 'initial_state': loaded_state})
        else:
            logger.error(f"Unhandled intent: {intent}")
            return IntentResult(type='error', data=f"Unknown intent: {intent}")

    async def handle_ask_process(self) -> IntentResult:
        logger.info("Handling intent: ASK_PROCESS")
        # Access db_file from state_manager if it's public, or assume it's not needed for explanation
        db_file_info = ""
        if hasattr(self.state_manager, 'db_file') and self.state_manager.db_file:
            db_file_info = f" using a local database (`{os.path.basename(self.state_manager.db_file)}`)"

        explanation = f"""I use an Advanced Bayesian Monte Carlo Tree Search (MCTS) algorithm. Key aspects include:
- **Exploration vs. Exploitation:** Balancing trying new ideas with focusing on promising ones.
- **Bayesian Evaluation:** (Optional) Using Beta distributions for score uncertainty.
- **Node Expansion:** Generating new 'thoughts' via LLM calls.
- **Simulation:** Evaluating analysis quality using LLM calls.
- **Backpropagation:** Updating scores/priors up the tree.
- **State Persistence:** (Optional) Saving key results between turns{db_file_info}.
- **Intent Handling:** Classifying your requests to guide the process.
You can adjust parameters via the configuration."""
        return IntentResult(type='message', data=f"**About My Process:**\n{explanation}")

    async def handle_ask_config(self) -> IntentResult:
        logger.info("Handling intent: ASK_CONFIG")
        try:
            config_to_display = self.config.copy()
            config_str = json.dumps(config_to_display, indent=2, default=str)
            return IntentResult(type='message', data=f"**Current Run Configuration:**\n```json\n{config_str}\n```")
        except Exception as e:
            logger.error(f"Failed to format/emit config: {e}")
            return IntentResult(type='error', data="Could not display configuration.")

    async def handle_ask_last_run_summary(self, loaded_state: Dict[str, Any]) -> IntentResult:
         logger.info("Handling intent: ASK_LAST_RUN_SUMMARY")
         try:
             summary = "**Summary of Last Analysis Run:**\n"
             summary += f"- **Best Score:** {loaded_state.get('best_score', 'N/A'):.1f}/10\n"
             summary += f"- **Best Analysis Tags:** {', '.join(loaded_state.get('best_node_tags', [])) or 'N/A'}\n"
             summary += f"- **Best Analysis Summary:** {loaded_state.get('best_solution_summary', 'N/A')}\n"

             priors = loaded_state.get("approach_priors")
             if priors and "alpha" in priors and "beta" in priors:
                 means = {}
                 alphas, betas = priors.get("alpha", {}), priors.get("beta", {})
                 for app, alpha_val in alphas.items(): # Renamed alpha to alpha_val
                     beta_val = betas.get(app, 1.0) # Renamed beta to beta_val
                     alpha_val, beta_val = max(1e-9, alpha_val), max(1e-9, beta_val)
                     if alpha_val + beta_val > 1e-9: means[app] = (alpha_val / (alpha_val + beta_val)) * 10
                 if means:
                     sorted_means = sorted(means.items(), key=lambda item: item[1], reverse=True)
                     top = [f"{app} ({score:.1f})" for app, score in sorted_means[:3]]
                     summary += f"- **Learned Approach Preferences:** Favored {', '.join(top)}...\n"
                 else: summary += "- **Learned Approach Preferences:** (No valid priors loaded)\n"

             unfit = loaded_state.get("unfit_markers", [])
             if unfit:
                  first_unfit = unfit[0]
                  summary += f"- **Potential Unfit Areas Noted:** {len(unfit)} (e.g., '{first_unfit.get('summary','...')}' due to {first_unfit.get('reason','...')})\n"
             else: summary += "- **Potential Unfit Areas Noted:** None\n"

             return IntentResult(type='message', data=summary)
         except Exception as e:
             logger.error(f"Error formatting last run summary: {e}", exc_info=True)
             return IntentResult(type='error', data="Could not display summary of last run.")

    async def handle_general_conversation(self) -> IntentResult:
        logger.info("Handling intent: GENERAL_CONVERSATION")
        # This is a placeholder, actual LLM call would be made by the calling service
        # or a more sophisticated response generation mechanism.
        response = "This is a general conversation response. How can I help you further with analysis?"
        return IntentResult(type='message', data=response)
