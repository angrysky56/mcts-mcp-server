#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini LLM Adapter - Fixed Version
========================================

Simple Gemini adapter using google-generativeai package.
"""
import logging
import os
import asyncio
from typing import List, Dict, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .base_llm_adapter import BaseLLMAdapter

class GeminiAdapter(BaseLLMAdapter):
    """
    Simple LLM Adapter for Google Gemini models.
    """
    DEFAULT_MODEL = "gemini-2.0-flash-lite"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

        if genai is None:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

        # Configure the API
        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized GeminiAdapter with model: {self.model_name}")

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict]]:
        """Convert messages to Gemini format."""
        system_instruction = None
        gemini_messages = []
        
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
        
        return system_instruction, gemini_messages

    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str:
        """Get completion from Gemini."""
        try:
            target_model = model or self.model_name
            system_instruction, gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            # Create the model
            model_obj = genai.GenerativeModel(
                model_name=target_model,
                system_instruction=system_instruction
            )
            
            # Convert messages to conversation format
            if gemini_messages:
                # For multi-turn conversation
                chat = model_obj.start_chat(history=gemini_messages[:-1])
                last_message = gemini_messages[-1]["parts"][0]
                
                # Run in thread to avoid blocking
                response = await asyncio.to_thread(chat.send_message, last_message)
            else:
                # Single message
                response = await asyncio.to_thread(
                    model_obj.generate_content, 
                    messages[-1]["content"] if messages else "Hello"
                )
            
            return response.text if response.text else "No response generated."
            
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return f"Error: {str(e)}"

    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs):
        """Get streaming completion (simplified to non-streaming for now)."""
        # For simplicity, just return the regular completion
        result = await self.get_completion(model, messages, **kwargs)
        yield result

    async def synthesize_result(self, context: Dict[str, str], config: Dict[str, any]) -> str:
        """Generate synthesis of MCTS results."""
        synthesis_prompt = f"""
Based on the MCTS exploration, provide a comprehensive synthesis:

Question: {context.get('question_summary', 'N/A')}
Initial Analysis: {context.get('initial_analysis_summary', 'N/A')}
Best Score: {context.get('best_score', 'N/A')}
Exploration Path: {context.get('path_thoughts', 'N/A')}
Final Analysis: {context.get('final_best_analysis_summary', 'N/A')}

Please provide a clear, comprehensive synthesis that:
1. Summarizes the key findings
2. Highlights the best solution approach
3. Explains why this approach is optimal
4. Provides actionable insights
"""
        
        messages = [{"role": "user", "content": synthesis_prompt}]
        return await self.get_completion(None, messages)
