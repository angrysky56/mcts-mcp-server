#!/usr/bin/env python3

"""
Fixed Tools for MCTS with proper async handling
===============================================

This module fixes the async event loop issues in the MCTS MCP tools.
"""
import asyncio
import logging
import os
import threading
from collections.abc import Coroutine
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .mcts_config import DEFAULT_CONFIG
from .mcts_core import MCTS
from .ollama_adapter import OllamaAdapter
from .ollama_utils import (
    OLLAMA_PYTHON_PACKAGE_AVAILABLE,
    check_available_models,
    get_recommended_models,
)
from .state_manager import StateManager
from .utils import truncate_text

logger = logging.getLogger(__name__)

# Global state to maintain between tool calls
_global_state = {
    "mcts_instance": None,
    "config": None,
    "state_manager": None,
    "current_chat_id": None,
    "active_llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "ollama"),
    "active_model_name": os.getenv("DEFAULT_MODEL_NAME"),
    "collect_results": False,
    "current_run_id": None,
    "ollama_available_models": [],
    "background_loop": None,
    "background_thread": None
}

def get_or_create_background_loop() -> asyncio.AbstractEventLoop | None:
    """
    Get or create a background event loop that runs in a dedicated thread.

    Returns:
        The background event loop, or None if creation failed

    Note:
        This ensures all async operations use the same event loop and avoids
        "bound to different event loop" issues common in MCP tools
    """
    global _global_state

    if _global_state["background_loop"] is None or _global_state["background_thread"] is None:
        loop_created = threading.Event()  # Use threading.Event instead of asyncio.Event
        loop_container: dict[str, asyncio.AbstractEventLoop | None] = {"loop": None}

        def create_background_loop():
            """Create and run a background event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_container["loop"] = loop
            _global_state["background_loop"] = loop
            loop_created.set()

            try:
                loop.run_forever()
            except Exception as e:
                logger.error(f"Background loop error: {e}")
            finally:
                loop.close()

        # Start the background thread
        thread = threading.Thread(target=create_background_loop, daemon=True)
        thread.start()
        _global_state["background_thread"] = thread

        # Wait for loop to be created (with shorter timeout to avoid hanging)
        if not loop_created.wait(timeout=2.0):
            logger.warning("Background loop creation timed out")
            # Don't raise an error, just return None and handle gracefully
            return None

        if loop_container["loop"] is None:
            logger.warning("Failed to create background event loop")
            return None

    return _global_state["background_loop"]

def run_in_background_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run a coroutine in the background event loop.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine execution

    Raises:
        RuntimeError: If all execution methods fail

    Note:
        This avoids the "bound to different event loop" issue by using
        a dedicated background loop with fallback strategies
    """
    loop = get_or_create_background_loop()

    if loop is None:
        # Fallback: try to run in a new event loop if background loop failed
        logger.warning("Background loop not available, using fallback")
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # If we're already in an event loop, use thread executor
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result(timeout=300)
            except Exception as e:
                raise RuntimeError(f"Failed to run coroutine: {e}") from e

    if loop.is_running():
        # Submit to the running loop and wait for result
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=300)  # 5 minute timeout
    else:
        # This shouldn't happen if the background loop is properly managed
        raise RuntimeError("Background event loop is not running")

def register_mcts_tools(mcp: FastMCP, db_path: str) -> None:
    """
    Register all MCTS-related tools with the MCP server.

    Args:
        mcp: The FastMCP server instance to register tools with
        db_path: Path to the SQLite database for state persistence

    Note:
        Initializes global state, loads environment variables, and registers
        all tool functions with proper async handling
    """
    global _global_state

    # Load environment variables
    load_dotenv()

    # Initialize state manager
    _global_state["state_manager"] = StateManager(db_path)

    # Initialize config
    _global_state["config"] = DEFAULT_CONFIG.copy()

    # Don't check Ollama models during initialization to prevent hanging
    # Models will be checked when list_ollama_models() is called
    _global_state["ollama_available_models"] = []

    # Set default model for ollama if needed
    if _global_state["active_llm_provider"] == "ollama" and not _global_state["active_model_name"]:
        _global_state["active_model_name"] = OllamaAdapter.DEFAULT_MODEL

    @mcp.tool()
    def initialize_mcts(question: str, chat_id: str, provider_name: str | None = None,
                       model_name: str | None = None, config_updates: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Initialize the MCTS system with proper async handling.

        Args:
            question: The question or topic to analyze
            chat_id: Unique identifier for this conversation session
            provider_name: LLM provider to use (ollama, openai, anthropic, gemini)
            model_name: Specific model name to use
            config_updates: Optional configuration overrides

        Returns:
            dictionary containing initialization status, configuration, and metadata

        Note:
            Creates LLM adapter, generates initial analysis, and sets up MCTS instance
            with optional state loading from previous sessions
        """
        global _global_state

        try:
            logger.info(f"Initializing MCTS for chat ID: {chat_id}")

            # Determine target provider and model
            target_provider = provider_name or _global_state["active_llm_provider"]
            target_model = model_name or _global_state["active_model_name"]

            logger.info(f"Using LLM Provider: {target_provider}, Model: {target_model}")

            # Update config if provided
            if config_updates:
                cfg = _global_state["config"].copy()
                cfg.update(config_updates)
                _global_state["config"] = cfg
            else:
                cfg = _global_state["config"]

            _global_state["current_chat_id"] = chat_id
            state_manager = _global_state["state_manager"]
            loaded_state = state_manager.load_state(chat_id) if cfg.get("enable_state_persistence", True) else None

            # Instantiate the appropriate adapter
            llm_adapter = None
            if target_provider == "ollama":
                if not target_model:
                    target_model = OllamaAdapter.DEFAULT_MODEL
                if target_model not in _global_state["ollama_available_models"]:
                    return {
                        "status": "model_error",
                        "error": f"Ollama model '{target_model}' not available",
                        "available_models": _global_state["ollama_available_models"]
                    }
                llm_adapter = OllamaAdapter(model_name=target_model)

            elif target_provider == "openai":
                from .openai_adapter import OpenAIAdapter
                if not target_model:
                    target_model = OpenAIAdapter.DEFAULT_MODEL
                llm_adapter = OpenAIAdapter(api_key=os.getenv("OPENAI_API_KEY"), model_name=target_model)

            elif target_provider == "anthropic":
                from .anthropic_adapter import AnthropicAdapter
                if not target_model:
                    target_model = AnthropicAdapter.DEFAULT_MODEL
                llm_adapter = AnthropicAdapter(api_key=os.getenv("ANTHROPIC_API_KEY"), model_name=target_model)

            elif target_provider == "gemini":
                from .gemini_adapter import GeminiAdapter
                if not target_model:
                    target_model = GeminiAdapter.DEFAULT_MODEL
                llm_adapter = GeminiAdapter(api_key=os.getenv("GEMINI_API_KEY"), model_name=target_model)

            else:
                return {"error": f"Unsupported LLM provider: {target_provider}", "status": "error"}

            _global_state["active_llm_provider"] = target_provider
            _global_state["active_model_name"] = target_model

            # Generate initial analysis using the background loop
            async def generate_initial():
                initial_prompt = f"<instruction>Provide an initial analysis of the following question. Be clear and concise.</instruction><question>{question}</question>"
                initial_messages = [{"role": "user", "content": initial_prompt}]
                return await llm_adapter.get_completion(model=target_model, messages=initial_messages)

            try:
                initial_analysis = run_in_background_loop(generate_initial())
            except Exception as e:
                logger.error(f"Failed to generate initial analysis: {e}")
                return {"error": f"Failed to generate initial analysis: {str(object=e)}", "status": "error"}

            # Create MCTS instance
            _global_state["mcts_instance"] = MCTS(
                llm_interface=llm_adapter,
                question=question,
                initial_analysis_content=initial_analysis or "No initial analysis available",
                config=cfg,
                initial_state=loaded_state
            )

            return {
                "status": "initialized",
                "question": question,
                "chat_id": chat_id,
                "initial_analysis": initial_analysis,
                "loaded_state": loaded_state is not None,
                "provider": target_provider,
                "model_used": target_model,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                "run_id": _global_state.get("current_run_id")
            }

        except ValueError as ve:
            logger.error(f"Configuration error: {ve}")
            return {"error": f"Configuration error: {ve!s}", "status": "config_error"}
        except Exception as e:
            logger.error(f"Error in initialize_mcts: {e}")
            return {"error": f"Failed to initialize MCTS: {e!s}", "status": "error"}

    @mcp.tool()
    def set_active_llm(provider_name: str, model_name: str | None = None) -> dict[str, Any]:
        """
        Set the active LLM provider and model for subsequent operations.

        Args:
            provider_name: Name of the LLM provider (ollama, openai, anthropic, gemini)
            model_name: Optional specific model name to use

        Returns:
            dictionary containing status and confirmation message

        Note:
            Changes the global LLM configuration but doesn't affect already
            initialized MCTS instances
        """
        global _global_state
        supported_providers = ["ollama", "openai", "anthropic", "gemini"]
        provider_name_lower = provider_name.lower()

        if provider_name_lower not in supported_providers:
            return {
                "status": "error",
                "message": f"Unsupported provider: '{provider_name}'. Supported: {supported_providers}"
            }

        _global_state["active_llm_provider"] = provider_name_lower
        _global_state["active_model_name"] = model_name

        log_msg = f"Set active LLM provider to: {provider_name_lower}."
        if model_name:
            log_msg += f" Set active model to: {model_name}."

        return {"status": "success", "message": log_msg}

    @mcp.tool()
    def list_ollama_models() -> dict[str, Any]:
        """
        List all available Ollama models with recommendations.

        Returns:
            dictionary containing:
            - status: Success or error status
            - ollama_available_models: List of all available models
            - current_ollama_model: Currently active model
            - recommended_small_models: Models suitable for basic tasks
            - recommended_medium_models: Models for complex analysis
            - message: Status message

        Note:
            Checks Ollama server connectivity and updates global model cache
        """
        logger.info("Listing Ollama models...")

        # Check if Ollama server is running
        try:
            import httpx
            with httpx.Client(base_url="http://localhost:11434", timeout=3.0) as client:
                response = client.get("/")
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": "Ollama server not responding. Please ensure Ollama is running."
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cannot connect to Ollama server: {e!s}"
            }

        # Get available models
        available_models = check_available_models()
        if not available_models:
            return {
                "status": "error",
                "message": "No Ollama models found. Try 'ollama pull MODEL_NAME' to download a model."
            }

        # Get recommendations
        recommendations = get_recommended_models(available_models)
        current_model = _global_state.get("active_model_name") if _global_state.get("active_llm_provider") == "ollama" else None

        # Update global state
        _global_state["ollama_available_models"] = available_models

        return {
            "status": "success",
            "ollama_available_models": available_models,
            "current_ollama_model": current_model,
            "recommended_small_models": recommendations["small_models"],
            "recommended_medium_models": recommendations["medium_models"],
            "message": f"Found {len(available_models)} Ollama models"
        }

    @mcp.tool()
    def run_mcts(iterations: int = 1, simulations_per_iteration: int = 5, model_name: str | None = None) -> dict[str, Any]:
        """
        Run the MCTS algorithm with proper async handling.

        Args:
            iterations: Number of MCTS iterations to run
            simulations_per_iteration: Number of simulations per iteration
            model_name: Optional model override (currently unused)

        Returns:
            dictionary containing:
            - status: 'started' if successful
            - message: Confirmation message
            - provider: Active LLM provider
            - model: Active model name
            - background_thread_id: Thread ID for monitoring

        Note:
            Runs MCTS in a background thread to avoid blocking the MCP server
            Automatically saves state if persistence is enabled
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {"error": "MCTS not initialized. Call initialize_mcts first."}

        active_provider = _global_state.get("active_llm_provider")
        active_model = _global_state.get("active_model_name")

        if not active_provider or not active_model:
            return {"error": "Active LLM provider or model not set."}

        # Update config for this run
        temp_config = mcts.config.copy()
        temp_config["max_iterations"] = iterations
        temp_config["simulations_per_iteration"] = simulations_per_iteration
        mcts.config = temp_config

        logger.info(f"Starting MCTS run with {iterations} iterations, {simulations_per_iteration} simulations per iteration")

        def run_mcts_background():
            """Run MCTS in background thread with proper async handling."""
            try:
                # Use the background loop for all async operations
                async def run_search():
                    await mcts.run_search_iterations(iterations, simulations_per_iteration)
                    return mcts.get_final_results()

                results = run_in_background_loop(run_search())

                # Save state if enabled
                if temp_config.get("enable_state_persistence", True) and _global_state["current_chat_id"]:
                    try:
                        _global_state["state_manager"].save_state(_global_state["current_chat_id"], mcts)
                        logger.info(f"Saved state for chat ID: {_global_state['current_chat_id']}")
                    except Exception as e:
                        logger.error(f"Error saving state: {e}")

                # Get best node and tags
                best_node = mcts.find_best_final_node()
                tags = best_node.descriptive_tags if best_node else []

                # Log the tags for debugging/monitoring
                if tags:
                    logger.info(f"Best node tags: {', '.join(tags)}")

                logger.info(f"MCTS run completed. Best score: {results.best_score if results else 0.0}")

            except Exception as e:
                logger.error(f"Error in background MCTS run: {e}")

        # Start background thread
        background_thread = threading.Thread(target=run_mcts_background)
        background_thread.daemon = True
        background_thread.start()

        return {
            "status": "started",
            "message": f"MCTS process started with {iterations} iterations and {simulations_per_iteration} simulations per iteration.",
            "provider": active_provider,
            "model": active_model,
            "background_thread_id": background_thread.ident
        }

    @mcp.tool()
    def generate_synthesis() -> dict[str, Any]:
        """
        Generate a final synthesis of the MCTS results.

        Returns:
            dictionary containing:
            - synthesis: Generated synthesis text
            - best_score: Best score achieved during search
            - tags: Descriptive tags from best analysis
            - iterations_completed: Number of iterations completed
            - provider: LLM provider used
            - model: Model used

        Raises:
            Returns error dict if MCTS not initialized or synthesis fails

        Note:
            Uses the same background loop as MCTS to ensure consistency
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {"error": "MCTS not initialized. Call initialize_mcts first."}

        try:
            async def synth():
                llm_adapter = mcts.llm
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
                    "previous_best_summary": "N/A",
                    "unfit_markers_summary": "N/A",
                    "learned_approach_summary": "N/A"
                }

                synthesis = await llm_adapter.synthesize_result(synth_context, mcts.config)
                best_node = mcts.find_best_final_node()
                tags = best_node.descriptive_tags if best_node else []

                return {
                    "synthesis": synthesis,
                    "best_score": results.best_score,
                    "tags": tags,
                    "iterations_completed": mcts.iterations_completed,
                    "provider": _global_state.get("active_llm_provider"),
                    "model": _global_state.get("active_model_name"),
                }

            # Use the background loop for synthesis generation
            synthesis_result = run_in_background_loop(synth())
            return synthesis_result

        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return {"error": f"Synthesis generation failed: {e!s}"}

    @mcp.tool()
    def get_config() -> dict[str, Any]:
        """
        Get the current MCTS configuration and system status.

        Returns:
            dictionary containing all configuration parameters, active LLM settings,
            and system capabilities

        Note:
            Filters out internal configuration keys starting with underscore
        """
        global _global_state
        config = {k: v for k, v in _global_state["config"].items() if not k.startswith("_")}
        config.update({
            "active_llm_provider": _global_state.get("active_llm_provider"),
            "active_model_name": _global_state.get("active_model_name"),
            "ollama_python_package_available": OLLAMA_PYTHON_PACKAGE_AVAILABLE,
            "ollama_available_models": _global_state.get("ollama_available_models", []),
            "current_run_id": _global_state.get("current_run_id")
        })
        return config

    @mcp.tool()
    def update_config(config_updates: dict[str, Any]) -> dict[str, Any]:
        """
        Update the MCTS configuration parameters.

        Args:
            config_updates: dictionary of configuration keys and new values

        Returns:
            Updated configuration dictionary

        Note:
            Provider and model changes are ignored - use set_active_llm instead
            Updates both global config and active MCTS instance if present
        """
        global _global_state

        logger.info(f"Updating MCTS config with: {config_updates}")

        # Provider and model changes should use set_active_llm
        if "active_llm_provider" in config_updates or "active_model_name" in config_updates:
            logger.warning("Use 'set_active_llm' tool to change LLM provider or model.")
            config_updates.pop("active_llm_provider", None)
            config_updates.pop("active_model_name", None)

        # Update config
        cfg = _global_state["config"].copy()
        cfg.update(config_updates)
        _global_state["config"] = cfg

        mcts = _global_state.get("mcts_instance")
        if mcts:
            mcts.config = cfg

        return get_config()

    @mcp.tool()
    def get_mcts_status() -> dict[str, Any]:
        """
        Get the current status of the MCTS system.

        Returns:
            dictionary containing:
            - initialized: Whether MCTS is initialized
            - chat_id: Current chat session ID
            - iterations_completed: Number of iterations run
            - simulations_completed: Total simulations run
            - best_score: Best score achieved
            - best_content_summary: Truncated best solution
            - tags: Tags from best analysis
            - tree_depth: Maximum tree depth explored
            - approach_types: List of analytical approaches used
            - active_llm_provider: Current LLM provider
            - active_model_name: Current model name
            - run_id: Current run identifier

        Note:
            Provides comprehensive status for monitoring and debugging
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {
                "initialized": False,
                "message": "MCTS not initialized. Call initialize_mcts first."
            }

        try:
            best_node = mcts.find_best_final_node()
            tags = best_node.descriptive_tags if best_node else []

            return {
                "initialized": True,
                "chat_id": _global_state.get("current_chat_id"),
                "iterations_completed": getattr(mcts, "iterations_completed", 0),
                "simulations_completed": getattr(mcts, "simulations_completed", 0),
                "best_score": getattr(mcts, "best_score", 0.0),
                "best_content_summary": truncate_text(getattr(mcts, "best_solution", ""), 100),
                "tags": tags,
                "tree_depth": mcts.memory.get("depth", 0) if hasattr(mcts, "memory") else 0,
                "approach_types": getattr(mcts, "approach_types", []),
                "active_llm_provider": _global_state.get("active_llm_provider"),
                "active_model_name": _global_state.get("active_model_name"),
                "run_id": _global_state.get("current_run_id")
            }
        except Exception as e:
            logger.error(f"Error getting MCTS status: {e}")
            return {
                "initialized": True,
                "error": f"Error getting MCTS status: {e!s}",
                "chat_id": _global_state.get("current_chat_id")
            }

    @mcp.tool()
    def run_model_comparison(question: str, iterations: int = 2, simulations_per_iteration: int = 10) -> dict[str, Any]:
        """
        Run MCTS across multiple models for comparison analysis.

        Args:
            question: The question to analyze across models
            iterations: Number of MCTS iterations per model
            simulations_per_iteration: Simulations per iteration

        Returns:
            dictionary containing comparison setup or error information

        Note:
            Currently returns a placeholder - full implementation requires
            additional coordination between multiple MCTS instances
        """
        if not OLLAMA_PYTHON_PACKAGE_AVAILABLE:
            return {"error": "Ollama python package not available for model comparison."}

        # Get available models
        models = check_available_models()
        recommendations = get_recommended_models(models)
        comparison_models = recommendations["small_models"]

        if not comparison_models:
            return {"error": f"No suitable models found for comparison. Available: {models}"}

        return {
            "status": "started",
            "message": "Model comparison feature available but not implemented in this version",
            "question": question,
            "models": comparison_models,
            "iterations": iterations,
            "simulations_per_iteration": simulations_per_iteration
        }
