#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Tools for MCTS
=================

This module defines the MCP tools that expose the MCTS functionality.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from llm_adapter import DirectMcpLLMAdapter

# Import from the MCTS core implementation
from mcts_core import (
    MCTS, StateManager, DEFAULT_CONFIG, truncate_text
)

logger = logging.getLogger(".tools")

# Global state to maintain between tool calls
_global_state = {
    "mcts_instance": None,
    "config": None,
    "state_manager": None,
    "current_chat_id": None
}

def run_async(coro):
    """
    Utility to run an async function in a synchronous context.
    Uses a thread-based approach to avoid event loop conflicts.
    """
    import threading
    import concurrent.futures
    import functools

    # Function to run in a separate thread
    def thread_runner():
        result = None
        exception = None
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Run the coroutine and store the result
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            # Clean up
            if 'loop' in locals():
                loop.close()
        return result, exception

    # Run the function in a separate thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(thread_runner)
        result, exception = future.result()

    # If there was an exception, log and re-raise it
    if exception is not None:
        logger.error(f"Error in run_async: {exception}")
        raise exception

    return result

def register_mcts_tools(mcp: FastMCP, db_path: str):
    """
    Register all MCTS-related tools with the MCP server.

    Args:
        mcp: The FastMCP instance to register tools with
        db_path: Path to the state database
    """
    global _global_state

    # Initialize state manager for persistence
    _global_state["state_manager"] = StateManager(db_path)

    # Initialize config with defaults
    _global_state["config"] = DEFAULT_CONFIG.copy()


    @mcp.tool()
    def initialize_mcts(question: str, chat_id: str, config_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize the MCTS system with a new question.

        Args:
            question: The question or text to analyze
            chat_id: Unique identifier for the chat session
            config_updates: Optional dictionary of configuration updates

        Returns:
            Dictionary with initialization status and initial analysis
        """
        global _global_state

        try:
            logger.info(f"Initializing MCTS for chat ID: {chat_id}")

            # Update config if provided
            if config_updates:
                cfg = _global_state["config"].copy()
                cfg.update(config_updates)
                _global_state["config"] = cfg
            else:
                cfg = _global_state["config"]

            # Store chat ID for state persistence
            _global_state["current_chat_id"] = chat_id

            # Try to load previous state
            state_manager = _global_state["state_manager"]
            loaded_state = None
            if cfg.get("enable_state_persistence", True):
                loaded_state = state_manager.load_state(chat_id)
                if loaded_state:
                    logger.info(f"Loaded previous state for chat ID: {chat_id}")
                else:
                    logger.info(f"No previous state found for chat ID: {chat_id}")

            try:
                # Try to use the real DirectMcpLLMAdapter, but fall back to dummy if issues occur
                logger.info("Initializing LLM adapter...")
                llm_adapter = DirectMcpLLMAdapter(mcp)

                # Test the adapter with a simple query
                async def test_adapter():
                    test_result = await llm_adapter.get_completion(None, [{"role": "user", "content": "Test message"}])
                    return test_result

                run_async(test_adapter())
                logger.info("LLM adapter working properly")
            except Exception as e:
                logger.warning(f"Error initializing real LLM adapter {e}.")
                llm_adapter = ()

            # Generate initial analysis
            logger.info("Generating initial analysis...")
            initial_prompt = f"<instruction>Provide an initial analysis and interpretation of the core themes, arguments, and potential implications presented. Identify key concepts. Respond with clear, natural language text ONLY.</instruction><question>{question}</question>"
            initial_messages = [{"role": "user", "content": initial_prompt}]

            # Call LLM for initial analysis (synchronously)
            async def get_initial_analysis():
                return await llm_adapter.get_completion(None, initial_messages)

            initial_analysis = run_async(get_initial_analysis())

            # Initialize MCTS
            logger.info("Creating MCTS instance...")
            async def init_mcts():
                return MCTS(
                    llm_interface=llm_adapter,
                    question=question,
                    initial_analysis_content=initial_analysis,
                    config=cfg,
                    initial_state=loaded_state
                )

            _global_state["mcts_instance"] = run_async(init_mcts())

            # Return success and initial analysis
            return {
                "status": "initialized",
                "question": question,
                "chat_id": chat_id,
                "initial_analysis": initial_analysis,
                "loaded_state": loaded_state is not None,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")}  # Filter internal config
            }
        except Exception as e:
            logger.error(f"Error in initialize_mcts {e}")

    @mcp.tool()
    def run_mcts(iterations: int = 1, simulations_per_iteration: int = 5) -> Dict[str, Any]:
        """
        Run the MCTS algorithm for the specified number of iterations.

        Args:
            iterations: Number of MCTS iterations to run (default: 1)
            simulations_per_iteration: Number of simulations per iteration (default: 5)

        Returns:
            Dictionary with results of the MCTS run
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {"error": "MCTS not initialized. Call initialize_mcts first."}

        # Override config values for this run
        temp_config = mcts.config.copy()
        temp_config["max_iterations"] = iterations
        temp_config["simulations_per_iteration"] = simulations_per_iteration
        mcts.config = temp_config

        logger.info(f"Running MCTS with {iterations} iterations, {simulations_per_iteration} simulations per iteration...")

        # Run MCTS (synchronously)
        async def run_search():
            await mcts.run_search_iterations(iterations, simulations_per_iteration)
            return mcts.get_final_results()

        try:
            results = run_async(run_search())
        except Exception as e:
            logger.error(f"Error running MCTS: {e}")
            return {"error": f"MCTS run failed: {str(e)}"}

        # Save state if enabled
        if temp_config.get("enable_state_persistence", True) and _global_state["current_chat_id"]:
            try:
                _global_state["state_manager"].save_state(_global_state["current_chat_id"], mcts)
                logger.info(f"Saved state for chat ID: {_global_state['current_chat_id']}")
            except Exception as e:
                logger.error(f"Error saving state: {e}")

        # Find best node and tags
        best_node = mcts.find_best_final_node()
        tags = best_node.descriptive_tags if best_node else []

        return {
            "status": "completed",
            "best_score": results.best_score,
            "best_solution": results.best_solution_content,
            "tags": tags,
            "iterations_completed": mcts.iterations_completed,
            "simulations_completed": mcts.simulations_completed
        }

    @mcp.tool()
    def generate_synthesis() -> Dict[str, Any]:
        """
        Generate a final synthesis of the MCTS results.

        Returns:
            Dictionary with the synthesis and related information
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {"error": "MCTS not initialized. Call initialize_mcts first."}

        logger.info("Generating synthesis of MCTS results...")

        # Use same LLM adapter as the MCTS instance
        llm_adapter = mcts.llm

        async def synth():
            # Prepare context for synthesis
            path_nodes = mcts.get_best_path_nodes()
            path_thoughts_list = [
                f"- (Node {node.sequence}): {node.thought.strip()}"
                for node in path_nodes if node.thought and node.parent
            ]
            path_thoughts_str = "\n".join(path_thoughts_list) if path_thoughts_list else "No significant development path identified."

            results = mcts.get_final_results()

            synth_context = {
                "question_summary": mcts.question_summary,
                "initial_analysis_summary": truncate_text(mcts.root.content, 300) if mcts.root else "N/A",
                "best_score": f"{results.best_score:.1f}",
                "path_thoughts": path_thoughts_str,
                "final_best_analysis_summary": truncate_text(results.best_solution_content, 400),
                # Add defaults for unused but required keys
                "previous_best_summary": "N/A",
                "unfit_markers_summary": "N/A",
                "learned_approach_summary": "N/A"
            }

            # Use the synthesize_result method from the LLMInterface
            synthesis = await llm_adapter.synthesize_result(synth_context, mcts.config)

            best_node = mcts.find_best_final_node()
            tags = best_node.descriptive_tags if best_node else []

            return {
                "synthesis": synthesis,
                "best_score": results.best_score,
                "tags": tags,
                "iterations_completed": mcts.iterations_completed
            }

        try:
            return run_async(synth())
        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return {"error": f"Synthesis generation failed: {str(e)}"}

    @mcp.tool()
    def get_config() -> Dict[str, Any]:
        """
        Get the current MCTS configuration.

        Returns:
            Dictionary with the current configuration values
        """
        global _global_state
        # Filter out private config items (those starting with _)
        return {k: v for k, v in _global_state["config"].items() if not k.startswith("_")}

    @mcp.tool()
    def update_config(config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the MCTS configuration.

        Args:
            config_updates: Dictionary with configuration keys and values to update

        Returns:
            Dictionary with the updated configuration
        """
        global _global_state

        logger.info(f"Updating config with: {config_updates}")

        cfg = _global_state["config"].copy()
        cfg.update(config_updates)
        _global_state["config"] = cfg

        # If MCTS instance exists, update its config
        mcts = _global_state.get("mcts_instance")
        if mcts:
            mcts.config = cfg

        # Return filtered config (without private items)
        return {k: v for k, v in cfg.items() if not k.startswith("_")}

    @mcp.tool()
    def get_mcts_status() -> Dict[str, Any]:
        """
        Get the current status of the MCTS system.

        Returns:
            Dictionary with status information
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {
                "initialized": False,
                "message": "MCTS not initialized. Call initialize_mcts first."
            }

        try:
            # Get best node and extract information
            best_node = mcts.find_best_final_node()
            tags = best_node.descriptive_tags if best_node else []

            return {
                "initialized": True,
                "chat_id": _global_state.get("current_chat_id"),
                "iterations_completed": getattr(mcts, "iterations_completed", 0),
                "simulations_completed": getattr(mcts, "simulations_completed", 0),
                "best_score": getattr(mcts, "best_score", 0.0),
                "best_content_summary": truncate_text(getattr(mcts, "best_solution", ""), 100)
                    if hasattr(mcts, "best_solution") else "N/A",
                "tags": tags,
                "tree_depth": mcts.memory.get("depth", 0) if hasattr(mcts, "memory") else 0,
                "approach_types": getattr(mcts, "approach_types", [])
            }
        except Exception as e:
            logger.error(f"Error getting MCTS status: {e}")
            return {
                "initialized": True,
                "error": f"Error getting MCTS status: {str(e)}",
                "chat_id": _global_state.get("current_chat_id")
            }
