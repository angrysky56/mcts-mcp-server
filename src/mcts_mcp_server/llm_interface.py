#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Interface Protocol
======================

This module defines the LLMInterface protocol for MCTS.
"""
from typing import List, Dict, Any, Protocol, AsyncGenerator

class LLMInterface(Protocol):
    """Defines the interface required for LLM interactions."""

    async def get_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Gets a non-streaming completion from the LLM."""
        ...

    async def get_streaming_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Gets a streaming completion from the LLM."""
        # This needs to be an async generator
        # Example: yield "chunk1"; yield "chunk2"
        if False: # pragma: no cover
             yield
        ...

    async def generate_thought(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a critical thought or new direction based on context."""
        ...

    async def update_analysis(self, critique: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Revises analysis based on critique and context."""
        ...

    async def evaluate_analysis(self, analysis_to_evaluate: str, context: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Evaluates analysis quality (1-10 score)."""
        ...

    async def generate_tags(self, analysis_text: str, config: Dict[str, Any]) -> List[str]:
        """Generates keyword tags for the analysis."""
        ...

    async def synthesize_result(self, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generates a final synthesis based on the MCTS results."""
        ...

    async def classify_intent(self, text_to_classify: str, config: Dict[str, Any]) -> str:
        """Classifies user intent using the LLM."""
        ...
