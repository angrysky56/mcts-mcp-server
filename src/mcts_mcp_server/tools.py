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
import datetime
import os # Added for os.getenv
from dotenv import load_dotenv # Added for .env loading
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from .llm_adapter import DirectMcpLLMAdapter # Changed to relative import

from .ollama_utils import (
    OLLAMA_PYTHON_PACKAGE_AVAILABLE, # Renamed, reflects if 'ollama' python package is installed
    # OllamaAdapter moved to its own file
    # SMALL_MODELS,
    # MEDIUM_MODELS,
    # DEFAULT_MODEL was removed from ollama_utils
    check_available_models,
    get_recommended_models
)
from .ollama_adapter import OllamaAdapter # Import new OllamaAdapter

# Import from the MCTS core implementation
# Make sure these imports are correct based on previous refactorings
from .mcts_core import MCTS # MCTS uses DEFAULT_CONFIG, APPROACH_TAXONOMY, APPROACH_METADATA internally
from .state_manager import StateManager
from .mcts_config import DEFAULT_CONFIG # For MCTS and general tool use
from .utils import truncate_text # For get_mcts_status

# Import the results collector
try:
    from results_collector import collector as results_collector
    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False
    results_collector = None

# Import the analysis tools
try:
    from analysis_tools import register_mcts_analysis_tools
    ANALYSIS_TOOLS_AVAILABLE = True
except ImportError:
    ANALYSIS_TOOLS_AVAILABLE = False
    register_mcts_analysis_tools = None

logger = logging.getLogger(__name__) # Changed to __name__ for consistency

# Global state to maintain between tool calls
_global_state = {
    "mcts_instance": None,
    "config": None, # Will be initialized with DEFAULT_CONFIG from mcts_config.py
    "state_manager": None,
    "current_chat_id": None,
    "active_llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "ollama"),
    # DEFAULT_MODEL_NAME from .env, or None. Provider-specific defaults handled in initialize_mcts.
    "active_model_name": os.getenv("DEFAULT_MODEL_NAME"),
    "collect_results": COLLECTOR_AVAILABLE,
    "current_run_id": None,
    "ollama_available_models": [] # Specifically for Ollama, populated by check_available_models
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

# check_available_models and get_recommended_models moved to ollama_utils.py

def register_mcts_tools(mcp: FastMCP, db_path: str):
    """
    Register all MCTS-related tools with the MCP server.

    Args:
        mcp: The FastMCP instance to register tools with
        db_path: Path to the state database
    """
    global _global_state

    # Load environment variables from .env file
    load_dotenv()

    # Initialize state manager for persistence
    _global_state["state_manager"] = StateManager(db_path)

    # Initialize config with defaults from mcts_config.py
    _global_state["config"] = DEFAULT_CONFIG.copy()
    
    # Populate available models from ollama_utils
    _global_state["ollama_available_models"] = check_available_models()
    if not _global_state["ollama_available_models"]:
        logger.warning("No Ollama models detected by ollama_utils.check_available_models(). Ollama provider might not function correctly if selected.")

    # Set a provider-specific default model if active_model_name is None AND current provider is ollama
    if _global_state["active_llm_provider"] == "ollama" and not _global_state["active_model_name"]:
        _global_state["active_model_name"] = OllamaAdapter.DEFAULT_MODEL # Use class default

    # Register the analysis tools if available
    if ANALYSIS_TOOLS_AVAILABLE:
        # Get the results directory path
        # Ensure os is imported if not already: import os
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(repo_dir, "results")
        
        # Register the analysis tools
        register_mcts_analysis_tools(mcp, results_dir)
        logger.info("Registered MCTS analysis tools")

    @mcp.tool()
    def initialize_mcts(question: str, chat_id: str, provider_name: Optional[str] = None, model_name: Optional[str] = None, config_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize the MCTS system with a new question, LLM provider, and model.

        Args:
            question: The question or text to analyze.
            chat_id: Unique identifier for the chat session.
            provider_name: Name of the LLM provider (e.g., "ollama", "openai", "anthropic", "gemini").
                           Defaults to DEFAULT_LLM_PROVIDER from .env or "ollama".
            model_name: Specific model name for the provider. Defaults to DEFAULT_MODEL_NAME from .env or provider-specific default.
            config_updates: Optional dictionary of configuration updates.

        Returns:
            Dictionary with initialization status and initial analysis.
        """
        global _global_state
        llm_adapter = None # Initialize llm_adapter to None

        try:
            logger.info(f"Initializing MCTS for chat ID: {chat_id}")

            # Determine target provider and model
            target_provider = provider_name or _global_state["active_llm_provider"]
            target_model = model_name or _global_state["active_model_name"] # This could be None if not in .env

            logger.info(f"Attempting to use LLM Provider: {target_provider}, Model: {target_model}")

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
            if loaded_state: logger.info(f"Loaded previous state for chat ID: {chat_id}")
            else: logger.info(f"No previous state found for chat ID: {chat_id}")

            # Instantiate the appropriate adapter
            if target_provider == "ollama":
                # OllamaAdapter might need specific checks like check_available_models
                if not target_model: target_model = OllamaAdapter.DEFAULT_MODEL # Use class default if still None
                if target_model not in _global_state["ollama_available_models"]: # Check against list for this provider
                    return {
                        "status": "model_error",
                        "error": f"Ollama model '{target_model}' not in available list: {_global_state['ollama_available_models']}",
                        "message": "Please select an available Ollama model or ensure it is pulled."
                    }
                llm_adapter = OllamaAdapter(model_name=target_model)
            elif target_provider == "openai":
                from .openai_adapter import OpenAIAdapter
                if not target_model: target_model = OpenAIAdapter.DEFAULT_MODEL
                llm_adapter = OpenAIAdapter(api_key=os.getenv("OPENAI_API_KEY"), model_name=target_model)
            elif target_provider == "anthropic":
                from .anthropic_adapter import AnthropicAdapter
                if not target_model: target_model = AnthropicAdapter.DEFAULT_MODEL
                llm_adapter = AnthropicAdapter(api_key=os.getenv("ANTHROPIC_API_KEY"), model_name=target_model)
            elif target_provider == "gemini":
                from .gemini_adapter import GeminiAdapter
                if not target_model: target_model = GeminiAdapter.DEFAULT_MODEL
                llm_adapter = GeminiAdapter(api_key=os.getenv("GEMINI_API_KEY"), model_name=target_model)
            else:
                return {"error": f"Unsupported LLM provider: {target_provider}. Supported: ollama, openai, anthropic, gemini.", "status": "error"}

            _global_state["active_llm_provider"] = target_provider
            _global_state["active_model_name"] = target_model
            logger.info(f"Successfully initialized LLM adapter for Provider: {target_provider}, Model: {target_model}")

            # Test adapter (optional, can be removed for speed)
            try:
                async def test_adapter_briefly():
                    return await llm_adapter.get_completion(model=target_model, messages=[{"role": "user", "content": "Brief test query."}])
                test_result = run_async(test_adapter_briefly())
                logger.info(f"Adapter test successful: {truncate_text(test_result, 50)}")
            except Exception as e:
                logger.error(f"Failed to test LLM adapter for {target_provider} model {target_model}: {e}", exc_info=True)
                # If adapter test fails, it's a significant issue. Let the error propagate or return specific error.
                # Removing the DirectMcpLLMAdapter fallback here as 'mcp' is not in local scope
                # and primary adapter initialization errors (e.g. API keys) are caught by ValueError.
                return {"error": f"LLM adapter for {target_provider} failed test: {e}", "status": "adapter_test_error"}

            # Generate initial analysis
            initial_prompt_format = "<instruction>Provide an initial analysis and interpretation of the core themes, arguments, and potential implications presented. Identify key concepts. Respond with clear, natural language text ONLY.</instruction><question>{question}</question>"
            initial_messages = [{"role": "user", "content": initial_prompt_format.format(question=question)}]

            initial_analysis = run_async(llm_adapter.get_completion(model=target_model, messages=initial_messages))

            _global_state["mcts_instance"] = MCTS(
                llm_interface=llm_adapter,
                question=question,
                initial_analysis_content=initial_analysis,
                config=cfg,
                initial_state=loaded_state
            )
            
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE:
                _global_state["current_run_id"] = results_collector.start_run(
                    model_name=target_model,
                    provider=target_provider, # Add provider to results
                    question=question,
                    config=cfg
                )
                logger.info(f"Started collecting results for run ID: {_global_state['current_run_id']}")

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
        except ValueError as ve: # Catch API key errors specifically
            logger.error(f"Configuration error in initialize_mcts: {ve}", exc_info=True)
            return {"error": f"Configuration error: {str(ve)}", "status": "config_error"}
        except Exception as e:
            logger.error(f"Error in initialize_mcts: {e}", exc_info=True)
            return {"error": f"Failed to initialize MCTS: {str(e)}", "status": "error"}

    @mcp.tool()
    def set_active_llm(provider_name: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Set the active LLM provider and optionally a model name for future MCTS runs.

        Args:
            provider_name: Name of the LLM provider (e.g., "ollama", "openai", "anthropic", "gemini").
            model_name: Optional specific model name for the provider. If None, provider's default will be used.

        Returns:
            Status message.
        """
        global _global_state
        supported_providers = ["ollama", "openai", "anthropic", "gemini"]
        provider_name_lower = provider_name.lower()

        if provider_name_lower not in supported_providers:
            return {
                "status": "error",
                "message": f"Unsupported LLM provider: '{provider_name}'. Supported providers are: {supported_providers}"
            }

        _global_state["active_llm_provider"] = provider_name_lower
        
        # If a model name is provided, set it. Otherwise, it will use the provider's default or .env DEFAULT_MODEL_NAME
        # specific to that provider during initialize_mcts.
        _global_state["active_model_name"] = model_name
        
        log_msg = f"Set active LLM provider to: {provider_name_lower}."
        if model_name:
            log_msg += f" Set active model to: {model_name}."
        else:
            log_msg += f" Active model will be provider's default or from .env DEFAULT_MODEL_NAME."
        logger.info(log_msg)

        return {
            "status": "success",
            "message": log_msg
        }

    @mcp.tool()
    def list_ollama_models() -> Dict[str, Any]:
        """
        List all available Ollama models.

        Returns:
            Dictionary with available models and their details
        """
        logger.info("Listing Ollama models...")
        
        # Check if Ollama server is running first
        try:
            import httpx
            client = httpx.Client(base_url="http://localhost:11434", timeout=3.0)
            response = client.get("/")
            
            if response.status_code != 200:
                logger.error(f"Ollama server health check failed with status code: {response.status_code}")
                return {
                    "status": "error",
                    "message": "Ollama server not responding. Please ensure Ollama is running with 'ollama serve'",
                    "diagnostics": {
                        "server_check": f"Failed with status {response.status_code}",
                        "server_url": "http://localhost:11434"
                    }
                }
                
            logger.info("Ollama server is running and responding to requests")
        except Exception as e:
            logger.error(f"Ollama server health check failed: {e}")
            return {
                "status": "error",
                "message": "Unable to connect to Ollama server. Please ensure Ollama is running with 'ollama serve'",
                "diagnostics": {
                    "error": str(e),
                    "server_url": "http://localhost:11434"
                }
            }
            
        # Get models using our comprehensive function
        available_models = check_available_models()
        
        # If we got no models, return detailed error
        if not available_models:
            return {
                "status": "error",
                "message": "No Ollama models detected. You may need to pull models using 'ollama pull MODEL_NAME'",
                "diagnostics": {
                    "server_check": "Server appears to be running but no models detected",
                    "suggestion": "Try running 'ollama pull qwen3:0.6b' or 'ollama pull cogito:latest' to download a model"
                }
            }
        
        # Get more detailed model information when possible
        model_details = []
        try:
            import subprocess
            import json
            import sys
            
            # Try using ollama show command to get detailed info
            for model in available_models:
                try:
                    if sys.platform == 'win32':
                        cmd = ['ollama.exe', 'show', model, '--json']
                    else:
                        cmd = ['ollama', 'show', model, '--json']
                        
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        try:
                            details = json.loads(result.stdout)
                            model_details.append({
                                "name": model,
                                "parameter_size": details.get("parameter_size", "unknown"),
                                "quantization": details.get("quantization_level", "unknown"),
                                "family": details.get("family", "unknown"),
                                "size_mb": round(details.get("size", 0) / (1024 * 1024), 1)
                            })
                        except json.JSONDecodeError:
                            # Add basic info if JSON parsing fails
                            model_details.append({
                                "name": model,
                                "parameter_size": "unknown",
                                "quantization": "unknown",
                                "family": "unknown"
                            })
                except Exception as e:
                    logger.warning(f"Error getting details for model {model}: {e}")
                    # Still add basic info
                    model_details.append({
                        "name": model,
                        "note": "Details unavailable"
                    })
        except Exception as e:
            logger.warning(f"Error getting detailed model information: {e}")
        
        # Get recommended models
        recommendations = get_recommended_models(available_models)
        
        # Determine if a model is already selected (specific to Ollama for this tool)
        current_ollama_model = _global_state.get("active_model_name") if _global_state.get("active_llm_provider") == "ollama" else None
        if not current_ollama_model and _global_state.get("active_llm_provider") == "ollama":
             # If provider is ollama but no model is set, try the class default for OllamaAdapter
            current_ollama_model = OllamaAdapter.DEFAULT_MODEL

        model_selected = current_ollama_model is not None and current_ollama_model in available_models
        
        # Customize message based on model selection status
        if model_selected:
            message = f"Current Ollama model for 'ollama' provider: {current_ollama_model}. Use set_active_llm to change provider or model."
        else:
            message = "To use Ollama, please use set_active_llm(provider_name='ollama', model_name='your_model') to select a model."
            
        # Clear active_model_name if it's an Ollama model but not in the available list
        if _global_state.get("active_llm_provider") == "ollama" and current_ollama_model and current_ollama_model not in available_models:
            logger.warning(f"Current active model {current_ollama_model} (Ollama) not found in available models. Clearing active_model_name.")
            _global_state["active_model_name"] = None # Will fall back to default or require setting
            current_ollama_model = None # For the message
        
        # Update global state with Ollama-specific available models
        _global_state["ollama_available_models"] = available_models # Store specifically for ollama
        
        return {
            "status": "success",
            "ollama_available_models": available_models, # Specifically Ollama models
            "model_details": model_details, # Details for Ollama models
            "current_ollama_model_for_provider": current_ollama_model, # If provider is ollama
            "recommended_small_ollama_models": recommendations["small_models"],
            "recommended_medium_ollama_models": recommendations["medium_models"],
            "message": message,
            "model_selected": model_selected
        }

    @mcp.tool()
    def run_mcts(iterations: int = 1, simulations_per_iteration: int = 5, model_name: Optional[str] = None) -> Dict[str, Any]: # Restored def
        """
        Run the MCTS algorithm for the specified number of iterations using the currently active LLM provider and model.
        The model_name parameter here is currently not used, as the model is determined by initialize_mcts or set_active_llm.
        This could be changed to allow overriding the model for a specific run if desired.

        Args:
            iterations: Number of MCTS iterations to run (default: 1)
            simulations_per_iteration: Number of simulations per iteration (default: 5)
            model_name: (Currently not used, model is taken from _global_state set by initialize_mcts or set_active_llm)

        Returns:
            Dictionary with status message about the background run
        """
        global _global_state

        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {"error": "MCTS not initialized. Call initialize_mcts first."}
        
        active_provider = _global_state.get("active_llm_provider")
        active_model = _global_state.get("active_model_name")

        if not active_provider or not active_model:
            return {"error": "Active LLM provider or model not set. Call initialize_mcts or set_active_llm first."}

        # Override config values for this run
        temp_config = mcts.config.copy()
        temp_config["max_iterations"] = iterations
        temp_config["simulations_per_iteration"] = simulations_per_iteration
        mcts.config = temp_config # This updates the config in the MCTS instance

        logger.info(f"Starting MCTS background run with {iterations} iterations, {simulations_per_iteration} simulations per iteration using Provider: {active_provider}, Model: {active_model}...")

        # Update collector status if enabled
        if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
            results_collector.update_run_status(
                _global_state["current_run_id"], 
                "running",
                {
                    "iterations": iterations,
                    "simulations_per_iteration": simulations_per_iteration,
                    "provider": active_provider,
                    "model": active_model,
                    "timestamp": int(datetime.datetime.now().timestamp())
                }
            )

        # Start MCTS search in a background thread and return immediately
        import threading
        
        def run_mcts_background():
            try:
                # Run the search asynchronously (wrap in run_async)
                async def run_search():
                    await mcts.run_search_iterations(iterations, simulations_per_iteration)
                    return mcts.get_final_results()

                results = run_async(run_search())
                
                # After completion, save state and update results
                if temp_config.get("enable_state_persistence", True) and _global_state["current_chat_id"]:
                    try:
                        _global_state["state_manager"].save_state(_global_state["current_chat_id"], mcts)
                        logger.info(f"Saved state for chat ID: {_global_state['current_chat_id']}")
                    except Exception as e:
                        logger.error(f"Error saving state: {e}")

                # Find best node and tags
                best_node = mcts.find_best_final_node()
                tags = best_node.descriptive_tags if best_node else []

                # Prepare results
                result_dict = {
                    "status": "completed",
                    "best_score": results.best_score,
                    "best_solution": results.best_solution_content,
                    "tags": tags,
                    "iterations_completed": mcts.iterations_completed,
                    "simulations_completed": mcts.simulations_completed,
                    "provider": _global_state.get("active_llm_provider"),
                    "model": _global_state.get("active_model_name"),
                }
                
                # Save results to collector if enabled
                if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
                    results_collector.save_run_results(_global_state["current_run_id"], result_dict)
                    logger.info(f"Saved results for run ID: {_global_state['current_run_id']}")
                    
            except Exception as e:
                logger.error(f"Error in background MCTS run: {e}")
                
                # Update collector with failure if enabled
                if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
                    results_collector.update_run_status(
                        _global_state["current_run_id"], 
                        "failed",
                        {"error": str(e), "timestamp": int(datetime.datetime.now().timestamp())}
                    )
        
        # Start the background thread
        background_thread = threading.Thread(target=run_mcts_background)
        background_thread.daemon = True  # Allow the thread to exit when the main process exits
        background_thread.start()
        
        # Return immediately with a status message
        return {
            "status": "started",
            "message": f"MCTS process started in background with {iterations} iterations and {simulations_per_iteration} simulations per iteration.",
            "provider": _global_state.get("active_llm_provider"),
            "model": _global_state.get("active_model_name"),
            "run_id": _global_state.get("current_run_id"),
            "results_path": f"/home/ty/Repositories/ai_workspace/mcts-mcp-server/results/{_global_state.get('active_llm_provider')}_{_global_state.get('active_model_name')}_{_global_state.get('current_run_id')}", # Adjusted path
            "background_thread_id": background_thread.ident
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
                "iterations_completed": mcts.iterations_completed,
                "provider": _global_state.get("active_llm_provider"),
                "model": _global_state.get("active_model_name"),
            }

        try:
            synthesis_result = run_async(synth())
            
            # Update results in collector if enabled
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
                results_collector.update_run_status(
                    _global_state["current_run_id"],
                    "completed",
                    {"synthesis": synthesis_result.get("synthesis")}
                )
                synthesis_result["run_id"] = _global_state["current_run_id"]
            
            return synthesis_result
            
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
        # Add active LLM provider and model info
        config = {k: v for k, v in _global_state["config"].items() if not k.startswith("_")}
        config.update({
            "active_llm_provider": _global_state.get("active_llm_provider"),
            "active_model_name": _global_state.get("active_model_name"),
            "ollama_python_package_available": OLLAMA_PYTHON_PACKAGE_AVAILABLE, # For info
            "ollama_available_models": _global_state.get("ollama_available_models", []),
            "collect_results": _global_state.get("collect_results"),
            "current_run_id": _global_state.get("current_run_id")
        })
        return config

    @mcp.tool()
    def update_config(config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the MCTS configuration. Provider and model are updated via set_active_llm.

        Args:
            config_updates: Dictionary with MCTS configuration keys and values to update.
                            To change LLM provider or model, use `set_active_llm` tool.

        Returns:
            Dictionary with the updated configuration.
        """
        global _global_state

        logger.info(f"Updating MCTS config with: {config_updates}")
        
        # Provider and model name changes should be handled by set_active_llm
        if "active_llm_provider" in config_updates or "active_model_name" in config_updates:
            logger.warning("Use 'set_active_llm' tool to change LLM provider or model name. These keys will be ignored in update_config.")
            config_updates.pop("active_llm_provider", None)
            config_updates.pop("active_model_name", None)
            config_updates.pop("ollama_model", None) # old key

        if "collect_results" in config_updates:
            _global_state["collect_results"] = bool(config_updates.pop("collect_results"))

        # Update regular MCTS config
        cfg = _global_state["config"].copy()
        cfg.update(config_updates) # Apply remaining valid config updates
        _global_state["config"] = cfg

        mcts = _global_state.get("mcts_instance")
        if mcts:
            mcts.config = cfg # Update config in existing MCTS instance

        # Return current effective config using get_config()
        return get_config()

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
                "approach_types": getattr(mcts, "approach_types", []),
                "active_llm_provider": _global_state.get("active_llm_provider"),
                "active_model_name": _global_state.get("active_model_name"),
                "collected_results": _global_state.get("collect_results"),
                "run_id": _global_state.get("current_run_id")
            }
        except Exception as e:
            logger.error(f"Error getting MCTS status: {e}")
            return {
                "initialized": True,
                "error": f"Error getting MCTS status: {str(e)}",
                "chat_id": _global_state.get("current_chat_id")
            }
            
    @mcp.tool()
    def run_model_comparison(question: str, iterations: int = 2, simulations_per_iteration: int = 10) -> Dict[str, Any]:
        """
        Run MCTS with the same question across multiple models for comparison.
        
        Args:
            question: The question to analyze with MCTS
            iterations: Number of MCTS iterations per model (default: 2)
            simulations_per_iteration: Simulations per iteration (default: 10)
            
        Returns:
            Dictionary with the run IDs for each model
        """
        # Use OLLAMA_PYTHON_PACKAGE_AVAILABLE to check if the package needed for check_available_models is there
        if not OLLAMA_PYTHON_PACKAGE_AVAILABLE: # Check if the ollama python lib is there
            return {"error": "Ollama python package not available. Cannot run Ollama model comparison."}
            
        if not COLLECTOR_AVAILABLE:
            return {"error": "Results collector is not available. Cannot track comparison results."}
            
        # Refresh available models
        models = check_available_models()
        
        # Filter to only include our preferred small models for faster comparison using constants from ollama_utils
        # Import SMALL_MODELS if needed here, or pass them to this function, or make get_recommended_models more flexible
        # For now, assuming get_recommended_models in ollama_utils handles this logic
        recommendations = get_recommended_models(models)
        comparison_models = recommendations["small_models"] # Example: Use recommended small models
        
        if not comparison_models:
            # If no "small" models, maybe try medium or any available? For now, error out.
            return {"error": f"No suitable Ollama models found from recommended small list for comparison. Available: {models}"}
            
        # Set up comparison config
        config = _global_state["config"].copy()
        config.update({
            "max_iterations": iterations,
            "simulations_per_iteration": simulations_per_iteration
        })
        
        # Start comparison
        run_ids = results_collector.compare_models(
            question=question,
            models=comparison_models,
            config=config,
            iterations=iterations,
            simulations_per_iter=simulations_per_iteration
        )
        
        return {
            "status": "started",
            "question": question,
            "iterations": iterations,
            "simulations_per_iteration": simulations_per_iteration,
            "models": comparison_models,
            "run_ids": run_ids
        }
