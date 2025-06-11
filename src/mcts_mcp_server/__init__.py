"""
MCTS MCP Server Package
======================

A Model Context Protocol (MCP) server that exposes an Advanced
Bayesian Monte Carlo Tree Search (MCTS) engine for AI reasoning.

MCTS Core Implementation
=======================

This package contains the core MCTS implementation.
"""

# Import key components to make them available at package level
from .mcts_config import DEFAULT_CONFIG, APPROACH_TAXONOMY, APPROACH_METADATA
from .utils import setup_logger, truncate_text, calculate_semantic_distance, _summarize_text, SKLEARN_AVAILABLE
from .node import Node
from .state_manager import StateManager
from .intent_handler import (
    IntentHandler,
    IntentResult,
    INITIAL_PROMPT,
    THOUGHTS_PROMPT,
    UPDATE_PROMPT,
    EVAL_ANSWER_PROMPT,
    TAG_GENERATION_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    INTENT_CLASSIFIER_PROMPT
)
from .llm_interface import LLMInterface # Moved from mcts_core
from .mcts_core import MCTS, MCTSResult # LLMInterface moved

# For Ollama related utilities, if they are meant to be public
from .ollama_utils import OLLAMA_AVAILABLE, OllamaAdapter, check_available_models, get_recommended_models, DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL

__all__ = [
    'MCTS', 'LLMInterface', 'Node', 'StateManager', 'IntentHandler', 'IntentResult', 'MCTSResult',
    'DEFAULT_CONFIG', 'APPROACH_TAXONOMY', 'APPROACH_METADATA',
    'setup_logger', 'truncate_text', 'calculate_semantic_distance', '_summarize_text', 'SKLEARN_AVAILABLE',
    'INITIAL_PROMPT', 'THOUGHTS_PROMPT', 'UPDATE_PROMPT', 'EVAL_ANSWER_PROMPT',
    'TAG_GENERATION_PROMPT', 'FINAL_SYNTHESIS_PROMPT', 'INTENT_CLASSIFIER_PROMPT',
    'OLLAMA_AVAILABLE', 'OllamaAdapter', 'check_available_models', 'get_recommended_models', 'OLLAMA_DEFAULT_MODEL'
]

__version__ = "0.1.0"
