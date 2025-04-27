#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for MCTS MCP Server LLM Adapter
===========================================

This script tests the LocalInferenceLLMAdapter which replaces the broken call_model approach.
"""
import os
import sys
import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcts_test")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import just the adapter
sys.path.insert(0, os.path.join(project_root, "src"))
from src.mcts_mcp_server.llm_adapter import LocalInferenceLLMAdapter

async def test_llm_adapter():
    """Test the local inference adapter."""
    logger.info("Testing LocalInferenceLLMAdapter...")
    adapter = LocalInferenceLLMAdapter()
    
    # Test basic completion
    test_messages = [{"role": "user", "content": "Generate a thought about AI safety."}]
    result = await adapter.get_completion(None, test_messages)
    logger.info(f"Basic completion test result: {result}")
    
    # Test thought generation
    context = {
        "question_summary": "What are the implications of AI in healthcare?",
        "current_approach": "initial",
        "best_score": "0",
        "best_answer": "",
        "current_answer": "",
        "current_sequence": "1"
    }
    
    # Use a dictionary for config
    config = {
        "max_children": 10,
        "exploration_weight": 3.0,
        "max_iterations": 1,
        "simulations_per_iteration": 10,
        "debug_logging": False,
    }
    
    thought = await adapter.generate_thought(context, config)
    logger.info(f"Thought generation test result: {thought}")
    
    # Test evaluation
    context["answer_to_evaluate"] = "AI in healthcare presents both opportunities and challenges. While it can improve diagnosis accuracy, there are ethical concerns about privacy and decision-making."
    score = await adapter.evaluate_analysis(context["answer_to_evaluate"], context, config)
    logger.info(f"Evaluation test result (score 1-10): {score}")
    
    # Test tag generation
    tags = await adapter.generate_tags("AI in healthcare can revolutionize patient care through improved diagnostics and personalized treatment plans.", config)
    logger.info(f"Tag generation test result: {tags}")
    
    # Test streaming
    logger.info("Testing streaming completion...")
    stream_messages = [{"role": "user", "content": "This is a test of streaming."}]
    async for chunk in adapter.get_streaming_completion(None, stream_messages):
        logger.info(f"Received chunk: {chunk}")
    
    logger.info("All LLM adapter tests completed successfully!")
    return True

async def main():
    """Run tests for the MCTS MCP server components."""
    try:
        # Test the LLM adapter
        adapter_result = await test_llm_adapter()
        if adapter_result:
            logger.info("✅ LLM adapter tests passed")
        
        logger.info("\nThe MCTS MCP server should now work with Claude Desktop.")
        logger.info("To use it with Claude Desktop:")
        logger.info("1. Copy the claude_desktop_config.json file to your Claude Desktop config location")
        logger.info("2. Restart Claude Desktop")
        logger.info("3. Ask Claude to analyze a topic using MCTS")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
