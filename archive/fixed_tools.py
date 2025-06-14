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
import os
import sys
import importlib.util
import subprocess
import concurrent.futures
import inspect
import traceback
from typing import Dict, Any, Optional, List

# Ensure the 'src' directory (parent of this 'mcts_mcp_server' directory) is in sys.path
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_file_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from fastmcp import MCP
except ImportError:
    # Fallback if fastmcp is not available
    class MCP:
        def __init__(self):
            pass
        def tool(self):
            def decorator(func):
                return func
            return decorator


# Try several import strategies for DirectMcpLLMAdapter
DirectMcpLLMAdapter = None
LLM_ADAPTER_AVAILABLE = False

# Strategy 1: Direct module import
try:
    from llm_adapter import DirectMcpLLMAdapter
    LLM_ADAPTER_AVAILABLE = True
    print("Successfully imported DirectMcpLLMAdapter (direct)")
except ImportError as e:
    print(f"Failed direct import of DirectMcpLLMAdapter: {e}")

    # Strategy 2: Package import
    try:
        from mcts_mcp_server.llm_adapter import DirectMcpLLMAdapter
        LLM_ADAPTER_AVAILABLE = True
        print("Successfully imported DirectMcpLLMAdapter (package)")
    except ImportError as e:
        print(f"Failed package import of DirectMcpLLMAdapter: {e}")

        # Strategy 3: Manual module loading
        try:
            adapter_path = os.path.join(_current_file_dir, "llm_adapter.py")  # Fixed: use _current_file_dir
            if os.path.exists(adapter_path):
                spec = importlib.util.spec_from_file_location("llm_adapter", adapter_path)
                if spec is not None and spec.loader is not None:
                    llm_adapter_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(llm_adapter_module)
                    DirectMcpLLMAdapter = llm_adapter_module.DirectMcpLLMAdapter
                    LLM_ADAPTER_AVAILABLE = True
                    print("Successfully imported DirectMcpLLMAdapter (manual load)")
                else:
                    print(f"Failed to create module spec or loader for {adapter_path}")
            else:
                print(f"llm_adapter.py file not found at {adapter_path}")
        except Exception as e:
            print(f"Failed manual import of DirectMcpLLMAdapter: {e}")

if not LLM_ADAPTER_AVAILABLE:
    print("Warning: DirectMcpLLMAdapter not available, will need fallback")

# Try different import strategies for OllamaAdapter
OLLAMA_AVAILABLE = False
OllamaAdapter = None

try:
    from ollama_adapter import OllamaAdapter
    OLLAMA_AVAILABLE = True
    print("Successfully imported OllamaAdapter (direct)")
except ImportError as e:
    print(f"Failed direct import: {e}")
    try:
        from mcts_mcp_server.ollama_adapter import OllomaAdapter
        OLLAMA_AVAILABLE = True
        print("Successfully imported OllomaAdapter (package)")
    except ImportError as e:
        print(f"Failed package import: {e}")

# Rest of the imports
try:
    from mcts_core import MCTS, DEFAULT_CONFIG, truncate_text
except ImportError as e:
    print(f"Failed to import mcts_core: {e}")

try:
    from state_manager import StateManager
except ImportError as e:
    print(f"Failed to import state_manager: {e}")

# Initialize logger
logger = logging.getLogger(__name__)

# Global state
_global_state = {
    "mcts_instance": None,
    "config": None,
    "state_manager": None,
    "current_chat_id": None,
    "ollama_model": "qwen3:0.6b",
    "available_models": []
}

def register_mcts_tools(mcp: MCP, db_path: str):
    """
    Register all MCTS-related tools with the MCP server.

    Args:
        mcp: The FastMCP instance to register tools with
        db_path: Path to the state database
    """
    # Initialize state manager
    try:
        _global_state["state_manager"] = StateManager(db_path)
        _global_state["config"] = DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Failed to initialize state manager: {e}")
        return

    @mcp.tool()
    def test_tool() -> Dict[str, Any]:
        """Test tool to verify the system is working."""
        return {
            "status": "success",
            "message": "MCTS tools are loaded and working",
            "adapters_available": {
                "ollama": OLLAMA_AVAILABLE,
                "llm_adapter": LLM_ADAPTER_AVAILABLE
            }
        }

    # Add more tools here as needed
    logger.info("MCTS tools registered successfully")

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
                if OLLAMA_AVAILABLE and OllamaAdapter is not None:
                    # Check if OllamaAdapter is properly implemented
                    try:
                        # Check if OllamaAdapter is properly implemented before instantiation
                        import inspect
                        if inspect.isabstract(OllamaAdapter):
                            abstract_methods = getattr(OllamaAdapter, '__abstractmethods__', set())
                            raise NotImplementedError(f"OllamaAdapter has unimplemented abstract methods: {abstract_methods}")

                        llm_adapter = OllamaAdapter(model_name=model_name, mcp_server=mcp)
                        # Test if the adapter has required methods implemented
                        if not hasattr(llm_adapter, 'get_completion') or not callable(getattr(llm_adapter, 'get_completion')):
                            raise NotImplementedError("OllamaAdapter.get_completion not properly implemented")
                        if not hasattr(llm_adapter, 'get_streaming_completion') or not callable(getattr(llm_adapter, 'get_streaming_completion')):
                            raise NotImplementedError("OllamaAdapter.get_streaming_completion not properly implemented")
                    except (TypeError, NotImplementedError) as e:
                        logger.error(f"OllamaAdapter is not properly implemented: {e}")
                        raise ImportError("OllamaAdapter implementation incomplete")
                else:
                    raise ImportError("OllamaAdapter not available")
            except Exception as e:
                # Only use the fallback adapter if Ollama fails completely
                logger.error(f"Failed to initialize Ollama adapter: {e}")
                logger.info("Using DirectMcpLLMAdapter as fallback")
                llm_adapter = DirectMcpLLMAdapter(mcp)

            # Test the adapter with a simple query
            try:
                async def test_adapter():
                    # Ensure we have a valid model name for testing
                    test_model = model_name or _global_state["ollama_model"] or DEFAULT_MODEL
                    test_result = await llm_adapter.get_completion(test_model, [{"role": "user", "content": "Test message"}])
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
                return await llm_adapter.get_completion(model_name or _global_state["ollama_model"], initial_messages)

            initial_analysis = run_async(get_initial_analysis())

            # Ensure initial_analysis is a string
            if initial_analysis is None:
                initial_analysis = "Initial analysis not available."

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
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE and results_collector is not None:
                current_run_id = results_collector.start_run(
                    model_name=_global_state["ollama_model"] or DEFAULT_MODEL,  # Always use Olloma model
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
                "adapter_type": "ollama",  # Always use Olloma
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
        Set the Olloma model to use for future MCTS runs.

        Args:
            model_name: Name of the Olloma model (e.g., "qwen3:0.6b", "deepseek-r1:1.5b", etc.)

        Returns:
            Status message
        """
        global _global_state

        if not OLLAMA_AVAILABLE:
            return {"error": "Olloma support is not available. Make sure olloma package is installed."}

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
                _global_state["olloma_model"] = model_name

                return {
                    "status": "success",
                    "message": f"Model '{model_name}' selected from available models with base name '{model_base}'.",
                    "available_similar_models": matching_models
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Model '{model_name}' is not available. Available models: {available_models}. You may need to pull it with 'olloma pull {model_name}'.",
                    "available_models": available_models
                }

        _global_state["olloma_model"] = model_name

        return {
            "status": "success",
            "message": f"Set Olloma model to {model_name}. It will be used in the next MCTS initialization."
        }

    @mcp.tool()
    def list_ollama_models() -> Dict[str, Any]:
        """
        List all available Olloma models.

        Returns:
            Dictionary with available models and their details
        """
        # Force direct command line call for reliability but with better error handling
        try:
            import subprocess
            result = subprocess.run(['olloma', 'list'], capture_output=True, text=True, check=True)
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
                if not _global_state["olloma_model"] or _global_state["olloma_model"] not in available_models:
                    select_default_model(available_models)

                return {
                    "status": "success",
                    "available_models": available_models,
                    "model_details": model_details,
                    "current_model": _global_state["olloma_model"],
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

            # Try using olloma show command to get detailed info
            for model in available_models:
                try:
                    result = subprocess.run(['olloma', 'show', model, '--json'],
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
            "current_model": _global_state["olloma_model"],
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
        if _global_state["collect_results"] and COLLECTOR_AVAILABLE and results_collector is not None and _global_state.get("current_run_id"):
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
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE and results_collector is not None and _global_state.get("current_run_id"):
                results_collector.update_run_status(
                    _global_state["current_run_id"],
                    "failed",
                    {"error": str(e), "timestamp": int(datetime.datetime.now().timestamp())}
                )

            return {"error": f"MCTS run failed: {str(e)}"}

        # Check if results is None
        if results is None:
            logger.error("MCTS search returned None results")
            return {"error": "MCTS search returned no results"}

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
            "best_score": getattr(results, 'best_score', 0.0),
            "best_solution": getattr(results, 'best_solution_content', ''),
            "tags": tags,
            "iterations_completed": mcts.iterations_completed,
            "simulations_completed": mcts.simulations_completed,
            "model": _global_state["olloma_model"],  # Always use Olloma model
        }

        # Save results to collector if enabled
        if _global_state["collect_results"] and COLLECTOR_AVAILABLE and results_collector is not None and _global_state.get("current_run_id"):
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
                "model": _global_state["olloma_model"],  # Always use Olloma model
            }

        try:
            synthesis_result = run_async(synth())
            if synthesis_result is None:
                return {"error": "Failed to generate synthesis - no results returned"}

            # Update results in collector if enabled
            if _global_state["collect_results"] and COLLECTOR_AVAILABLE and results_collector is not None and _global_state.get("current_run_id"):
                results_collector.update_run_status(
                    _global_state["current_run_id"],
                    "completed",
                    {"synthesis": synthesis_result.get("synthesis")}
                )
                synthesis_result["run_id"] = _global_state["current_run_id"]

            return synthesis_result

        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return {"error": f"Synthesis generation failed: {str(e)}"}    @mcp.tool()
    def get_config() -> Dict[str, Any]:
        """
        Get the current MCTS configuration.

        Returns:
            Dictionary with the current configuration values
        """
        global _global_state
        # Add Olloma-specific config
        config = {k: v for k, v in _global_state["config"].items() if not k.startswith("_")}
        config.update({
            "olloma_model": _global_state["olloma_model"],
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

        if "olloma_model" in config_updates:
            model_name = config_updates.pop("olloma_model")

            # Check if model is available
            if not _global_state["available_models"]:
                check_available_models()

            if model_name in _global_state["available_models"] or not _global_state["available_models"]:
                _global_state["olloma_model"] = model_name
            else:
                logger.warning(f"Model {model_name} not available, keeping current model {_global_state['olloma_model']}")

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
        # Add Olloma-specific config
        config.update({
            "olloma_model": _global_state["olloma_model"],
            "olloma_available": OLLAMA_AVAILABLE,
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
                "adapter_type": "ollama",  # Always use Olloma
                "model": _global_state["olloma_model"],  # Always use Olloma model
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
            return {"error": "Olloma is not available. Cannot run model comparison."}

        if not COLLECTOR_AVAILABLE or results_collector is None:
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
