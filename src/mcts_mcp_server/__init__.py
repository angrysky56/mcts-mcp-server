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
from .mcts_core import (
    MCTS, LLMInterface, Node, StateManager, IntentHandler, IntentResult,
    DEFAULT_CONFIG, setup_logger, truncate_text, calculate_semantic_distance, MCTSResult
)

__all__ = [
    'MCTS', 'LLMInterface', 'Node', 'StateManager', 'IntentHandler', 'IntentResult',
    'DEFAULT_CONFIG', 'setup_logger', 'truncate_text', 'calculate_semantic_distance', 'MCTSResult'
]

__version__ = "0.1.0"
