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
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from llm_adapter import DirectMcpLLMAdapter

# Try several import strategies to ensure we can import the Ollama adapter
import sys
import os
import importlib.util

# Add all possible import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try different import strategies
OLLAMA_AVAILABLE = False
OllamaAdapter = None

# Strategy 1: Direct module import
try:
    from ollama_adapter import OllamaAdapter
    OLLAMA_AVAILABLE = True
    print("Successfully imported OllamaAdapter (direct)")
except ImportError as e:
    print(f"Failed direct import: {e}")
    
    # Strategy 2: Package import
    try:
        from mcts_mcp_server.ollama_adapter import OllamaAdapter
        OLLAMA_AVAILABLE = True
        print("Successfully imported OllamaAdapter (package)")
    except ImportError as e:
        print(f"Failed package import: {e}")
        
        # Strategy 3: Manual module loading
        try:
            adapter_path = os.path.join(current_dir, "ollama_adapter.py")
            if os.path.exists(adapter_path):
                spec = importlib.util.spec_from_file_location("ollama_adapter", adapter_path)
                ollama_adapter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ollama_adapter)
                OllamaAdapter = ollama_adapter.OllamaAdapter
                OLLAMA_AVAILABLE = True
                print("Successfully imported OllamaAdapter (manual load)")
            else:
                print(f"Adapter file not found at {adapter_path}")
        except Exception as e:
            print(f"Failed manual import: {e}")

# Force Ollama availability check
if OLLAMA_AVAILABLE:
    try:
        import ollama
        # Test if the package works
        ollama_version = getattr(ollama, "__version__", "unknown")
        print(f"Ollama package version: {ollama_version}")
    except ImportError as e:
        print(f"Ollama package not available: {e}")
        OLLAMA_AVAILABLE = False
    except Exception as e:
        print(f"Error testing Ollama package: {e}")

# Import from the MCTS core implementation
from mcts_core import (
    MCTS, StateManager, DEFAULT_CONFIG, truncate_text
)

# Import the results collector
try:
    from results_collector import collector as results_collector
    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False
    results_collector = None

logger = logging.getLogger(".tools")

# Model preferences by size for more flexible selection
SMALL_MODELS = ["qwen3:0.6b", "deepseek-r1:1.5b", "cogito:latest", "phi3:mini", "tinyllama", "phi2:2b", "qwen2:1.5b"]
MEDIUM_MODELS = ["mistral:7b", "llama3:8b", "gemma:7b", "mistral-nemo:7b"]
DEFAULT_MODEL = "qwen3:0.6b"  # Initial default but will be dynamically set

# Global state to maintain between tool calls
_global_state = {
    "mcts_instance": None,
    "config": None,
    "state_manager": None,
    "current_chat_id": None,
    "ollama_model": DEFAULT_MODEL,  # Default to a small, fast model
    "collect_results": COLLECTOR_AVAILABLE,  # Flag to collect results
    "current_run_id": None,  # Current run ID for results collection
    "available_models": []  # Will be populated with actual models from Ollama
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

def check_available_models():
    """Check which Ollama models are available locally."""
    global _global_state
    
    if not OLLAMA_AVAILABLE:
        logger.warning("Ollama is not available, can't check models")
        return []
    
    # Method 1: Try subprocess call first (most reliable)
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        # Skip the header line if present
        if len(lines) > 1 and "NAME" in lines[0] and "ID" in lines[0]:
            lines = lines[1:]
        
        # Extract model names
        models = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if parts:
                model_name = parts[0]
                if ':' not in model_name:
                    model_name += ':latest'
                models.append(model_name)
        
        if models:
            logger.info(f"Available Ollama models via subprocess: {models}")
            # Update global state
            _global_state["available_models"] = models
            
            # Intelligently select a default model
            select_default_model(models)
            return models
    except Exception as e:
        logger.warning(f"Subprocess method failed: {e}")
    
    # Method 2: Fallback to HTTP API
    try:
        import httpx
        client = httpx.Client(base_url="http://localhost:11434", timeout=5.0)
        response = client.get("/api/tags")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name") for m in models if m.get("name")]
            
            if model_names:
                logger.info(f"Available Ollama models via HTTP API: {model_names}")
                # Update global state
                _global_state["available_models"] = model_names
                
                # Intelligently select a default model
                select_default_model(model_names)
                return model_names
        else:
            logger.warning(f"Failed to get models from Ollama API: {response.status_code}")
    except Exception as e:
        logger.warning(f"HTTP API method failed: {e}")
    
    # Method 3: Try ollama package with fixed handling for the current API format
    try:
        import ollama
        # Get models directly with error handling
        try:
            models_data = ollama.list()
            logger.info(f"Ollama package response type: {type(models_data)}")
            
            # Handle different response types
            model_names = []
            
            # Current Ollama package format (tested with version 0.4.8)
            if hasattr(models_data, 'models') and hasattr(models_data.models, '__iter__'):
                # Process Pydantic models from the modern API
                for model in models_data.models:
                    if hasattr(model, 'model'):
                        model_name = model.model
                        model_names.append(model_name)
                        logger.info(f"Found model via 'model' attribute: {model_name}")
                    elif hasattr(model, 'name'):
                        model_name = model.name  
                        model_names.append(model_name)
                        logger.info(f"Found model via 'name' attribute: {model_name}")
            
            # Older dictionary format
            elif isinstance(models_data, dict) and "models" in models_data:
                for model in models_data["models"]:
                    if isinstance(model, dict) and "name" in model:
                        model_names.append(model["name"])
            
            # Direct list format
            elif isinstance(models_data, list):
                for model in models_data:
                    if isinstance(model, dict) and "name" in model:
                        model_names.append(model["name"])
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
                    else:
                        # Last resort - convert to string
                        model_names.append(str(model))
            
            if model_names:
                logger.info(f"Ollama package found {len(model_names)} models: {model_names}")
                _global_state["available_models"] = model_names
                select_default_model(model_names)
                return model_names
            else:
                logger.warning("Ollama package returned data but no models could be extracted")
        except Exception as e:
            logger.warning(f"Error processing Ollama response: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    except ImportError as e:
        logger.warning(f"Ollama package import failed: {e}")
    except Exception as e:
        logger.warning(f"Ollama package method unexpected error: {e}")
    
    # If we get here, all methods failed
    return []


def select_default_model(models):
    """Select the best default model from available models."""
    global _global_state
    
    # First try small models
    for model in SMALL_MODELS:
        if model in models:
            _global_state["ollama_model"] = model
            logger.info(f"Selected small model: {model}")
            return
            
    # Then try medium models
    for model in MEDIUM_MODELS:
        if model in models:
            _global_state["ollama_model"] = model
            logger.info(f"Selected medium model: {model}")
            return
    
    # Fall back to any model
    if models:
        _global_state["ollama_model"] = models[0]
        logger.info(f"Selected first available model: {models[0]}")

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
    
    # Check available models
    check_available_models()


    @mcp.tool()
    def initialize_mcts(question: str, chat_id: str, model_name: Optional[str] = None, config_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize the MCTS system with a new question.

        Args:
            question: The question or text to analyze
            chat_id: Unique identifier for the chat session
            model_name: Optional specific Ollama model to use
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

            # Initialize the LLM adapter - ALWAYS use Ollama
            logger.info("Initializing LLM adapter...")
            
            # Get available Ollama models
            available_models = check_available_models()
            
            # If user specified a model, try to use it
            if model_name:
                if model_name in available_models:
                    _global_state["ollama_model"] = model_name
                    logger.info(f"Using user-specified model: {model_name}")
                else:
                    # Try to find a model with the same base name
                    model_base = model_name.split(':')[0]
                    matching_models = [m for m in available_models if m.startswith(model_base + ':')]
                    
                    if matching_models:
                        model_name = matching_models[0]
                        _global_state["ollama_model"] = model_name
                        logger.info(f"Found similar model: {model_name}")
                    else:
                        logger.warning(f"Model '{model_name}' not found. Using default model selection.")
            
            # Make sure we have a selected model
            model_name = _global_state["ollama_model"]
            if not available_models or model_name not in available_models:
                # If we have models but current selection isn't valid, pick a new one
                if available_models:
                    select_default_model(available_models)
                    model_name = _global_state["ollama_model"]
                    logger.info(f"Selected model not available, using {model_name}")
            
            # Always try to use Ollama first
            logger.info(f"Using OllamaAdapter with model {model_name}")
            try:
                llm_adapter = OllamaAdapter(model_name=model_name, mcp_server=mcp)
            except Exception as e:
                # Only use the fallback adapter if Ollama fails completely
                logger.error(f"Failed to initialize Ollama adapter: {e}")
                logger.info("Using DirectMcpLLMAdapter as fallback")
                llm_adapter = DirectMcpLLMAdapter(mcp)

            # Test the adapter with a simple query
            try:
                async def test_adapter():
                    test_result = await llm_adapter.get_completion(None, [{"role": "user", "content": "Test message"}])
                    return test_result

                run_async(test_adapter())
                logger.info("LLM adapter working properly")
            except Exception as e:
                logger.warning(f"Error testing LLM adapter: {e}. Using default LocalInferenceLLMAdapter.")
                from llm_adapter import LocalInferenceLLMAdapter
                llm_adapter = LocalInferenceLLMAdapter()

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
            
            # Start collecting results if enabled
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE:
                current_run_id = results_collector.start_run(
                    model_name=_global_state["ollama_model"],  # Always use Ollama model
                    question=question,
                    config=cfg
                )
                _global_state["current_run_id"] = current_run_id
                logger.info(f"Started collecting results for run ID: {current_run_id}")

            # Return success and initial analysis
            return {
                "status": "initialized",
                "question": question,
                "chat_id": chat_id,
                "initial_analysis": initial_analysis,
                "loaded_state": loaded_state is not None,
                "adapter_type": "ollama",  # Always use Ollama
                "model": _global_state["ollama_model"],
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")},  # Filter internal config
                "run_id": _global_state.get("current_run_id")
            }
        except Exception as e:
            logger.error(f"Error in initialize_mcts {e}")
            return {"error": f"Failed to initialize MCTS: {str(e)}"}

    @mcp.tool()
    def set_ollama_model(model_name: str) -> Dict[str, Any]:
        """
        Set the Ollama model to use for future MCTS runs.

        Args:
            model_name: Name of the Ollama model (e.g., "qwen3:0.6b", "deepseek-r1:1.5b", etc.)

        Returns:
            Status message
        """
        global _global_state

        if not OLLAMA_AVAILABLE:
            return {"error": "Ollama support is not available. Make sure ollama package is installed."}
        
        # Refresh available models
        available_models = check_available_models()
        
        # Check if model is available
        if model_name not in available_models:
            # Try partial matches (just the model name without version specification)
            model_base = model_name.split(':')[0]
            matching_models = [m for m in available_models if m.startswith(model_base + ':')]
            
            if matching_models:
                # Found models with the same base name
                model_name = matching_models[0]
                _global_state["ollama_model"] = model_name
                
                return {
                    "status": "success",
                    "message": f"Model '{model_name}' selected from available models with base name '{model_base}'.",
                    "available_similar_models": matching_models
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Model '{model_name}' is not available. Available models: {available_models}. You may need to pull it with 'ollama pull {model_name}'.",
                    "available_models": available_models
                }

        _global_state["ollama_model"] = model_name
        
        return {
            "status": "success", 
            "message": f"Set Ollama model to {model_name}. It will be used in the next MCTS initialization."
        }

    @mcp.tool()
    def list_ollama_models() -> Dict[str, Any]:
        """
        List all available Ollama models.

        Returns:
            Dictionary with available models and their details
        """
        # Force direct command line call for reliability but with better error handling
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            # Skip the header line if present
            if len(lines) > 1 and "NAME" in lines[0] and "ID" in lines[0]:
                lines = lines[1:]
            
            # Extract model names
            available_models = []
            model_details = []
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:  # We need at least NAME, ID, and SIZE
                    model_name = parts[0]
                    model_id = parts[1]
                    model_size = parts[2]
                    
                    if ':' not in model_name:
                        model_name += ':latest'
                    
                    available_models.append(model_name)
                    model_details.append({
                        "name": model_name,
                        "id": model_id,
                        "size": model_size
                    })
            
            # Update global state
            if available_models:
                _global_state["available_models"] = available_models
                # Select a default model if needed
                if not _global_state["ollama_model"] or _global_state["ollama_model"] not in available_models:
                    select_default_model(available_models)
                    
                return {
                    "status": "success",
                    "available_models": available_models,
                    "model_details": model_details,
                    "current_model": _global_state["ollama_model"],
                    "recommended_small_models": SMALL_MODELS,
                    "recommended_medium_models": MEDIUM_MODELS
                }
        except Exception as e:
            logger.warning(f"Command-line list failed: {e}")
            
        # Fall back to check_available_models as a second attempt
        available_models = check_available_models()
        
        # Get more detailed model information when possible
        model_details = []
        try:
            import subprocess
            import json
            
            # Try using ollama show command to get detailed info
            for model in available_models:
                try:
                    result = subprocess.run(['ollama', 'show', model, '--json'], 
                                           capture_output=True, text=True, check=False)
                    if result.returncode == 0 and result.stdout.strip():
                        details = json.loads(result.stdout)
                        model_details.append({
                            "name": model,
                            "parameter_size": details.get("parameter_size", "unknown"),
                            "quantization": details.get("quantization_level", "unknown"),
                            "family": details.get("family", "unknown"),
                            "size_mb": round(details.get("size", 0) / (1024 * 1024), 1)
                        })
                except Exception as e:
                    logger.warning(f"Error getting details for model {model}: {e}")
        except Exception as e:
            logger.warning(f"Error getting detailed model information: {e}")
        
        return {
            "status": "success",
            "available_models": available_models,
            "model_details": model_details,
            "current_model": _global_state["ollama_model"],
            "recommended_small_models": SMALL_MODELS,
            "recommended_medium_models": MEDIUM_MODELS
        }

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

        # Update collector status if enabled
        if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
            results_collector.update_run_status(
                _global_state["current_run_id"], 
                "running",
                {
                    "iterations": iterations,
                    "simulations_per_iteration": simulations_per_iteration,
                    "timestamp": int(datetime.datetime.now().timestamp())
                }
            )

        # Run MCTS (synchronously)
        async def run_search():
            await mcts.run_search_iterations(iterations, simulations_per_iteration)
            return mcts.get_final_results()

        try:
            results = run_async(run_search())
        except Exception as e:
            logger.error(f"Error running MCTS: {e}")
            
            # Update collector with failure if enabled
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
                results_collector.update_run_status(
                    _global_state["current_run_id"], 
                    "failed",
                    {"error": str(e), "timestamp": int(datetime.datetime.now().timestamp())}
                )
                
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

        # Prepare results
        result_dict = {
            "status": "completed",
            "best_score": results.best_score,
            "best_solution": results.best_solution_content,
            "tags": tags,
            "iterations_completed": mcts.iterations_completed,
            "simulations_completed": mcts.simulations_completed,
            "model": _global_state["ollama_model"],  # Always use Ollama model
        }
        
        # Save results to collector if enabled
        if _global_state["collect_results"] and COLLECTOR_AVAILABLE and _global_state.get("current_run_id"):
            results_collector.save_run_results(_global_state["current_run_id"], result_dict)
            result_dict["run_id"] = _global_state["current_run_id"]

        return result_dict

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
                "model": _global_state["ollama_model"],  # Always use Ollama model
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
        # Add Ollama-specific config
        config = {k: v for k, v in _global_state["config"].items() if not k.startswith("_")}
        config.update({
            "ollama_model": _global_state["ollama_model"],
            "available_models": _global_state["available_models"],
            "collect_results": _global_state["collect_results"],
            "current_run_id": _global_state.get("current_run_id")
        })
        return config

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
        
        if "ollama_model" in config_updates:
            model_name = config_updates.pop("ollama_model")
            
            # Check if model is available
            if not _global_state["available_models"]:
                check_available_models()
                
            if model_name in _global_state["available_models"] or not _global_state["available_models"]:
                _global_state["ollama_model"] = model_name
            else:
                logger.warning(f"Model {model_name} not available, keeping current model {_global_state['ollama_model']}")
        
        if "collect_results" in config_updates:
            _global_state["collect_results"] = bool(config_updates.pop("collect_results"))

        # Update regular MCTS config
        cfg = _global_state["config"].copy()
        cfg.update(config_updates)
        _global_state["config"] = cfg

        # If MCTS instance exists, update its config
        mcts = _global_state.get("mcts_instance")
        if mcts:
            mcts.config = cfg

        # Return filtered config (without private items)
        config = {k: v for k, v in cfg.items() if not k.startswith("_")}
        # Add Ollama-specific config
        config.update({
            "ollama_model": _global_state["ollama_model"],
            "ollama_available": OLLAMA_AVAILABLE,
            "available_models": _global_state["available_models"],
            "collect_results": _global_state["collect_results"],
            "current_run_id": _global_state.get("current_run_id")
        })
        return config

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
                "adapter_type": "ollama",  # Always use Ollama
                "model": _global_state["ollama_model"],  # Always use Ollama model
                "collected_results": _global_state["collect_results"],
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
        if not OLLAMA_AVAILABLE:
            return {"error": "Ollama is not available. Cannot run model comparison."}
            
        if not COLLECTOR_AVAILABLE:
            return {"error": "Results collector is not available. Cannot track comparison results."}
            
        # Refresh available models
        models = check_available_models()
        
        # Filter to only include our preferred small models for faster comparison
        preferred_models = ["qwen3:0.6b", "deepseek-r1:1.5b", "cogito:latest"]
        comparison_models = [m for m in models if any(sm in m for sm in preferred_models)]
        
        if not comparison_models:
            return {"error": f"No suitable models found. Please pull at least one of: {preferred_models}"}
            
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
