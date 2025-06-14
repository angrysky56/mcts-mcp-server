#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Tools for MCTS with deferred initialization
====================================================

This module provides fast MCP server startup by deferring heavy
operations until they're actually needed.
"""
import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional
import threading

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Global state to maintain between tool calls
_global_state = {
    "mcts_instance": None,
    "config": None,
    "state_manager": None,
    "current_chat_id": None,
    "active_llm_provider": None,
    "active_model_name": None,
    "collect_results": False,
    "current_run_id": None,
    "ollama_available_models": [],
    "background_loop": None,
    "background_thread": None,
    "initialized": False
}

def lazy_init():
    """Initialize heavy components only when needed."""
    global _global_state
    
    if _global_state["initialized"]:
        return
    
    try:
        print("Lazy loading MCTS components...", file=sys.stderr)
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load config
        from .mcts_config import DEFAULT_CONFIG
        _global_state["config"] = DEFAULT_CONFIG.copy()
        
        # Set default provider from environment
        _global_state["active_llm_provider"] = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
        _global_state["active_model_name"] = os.getenv("DEFAULT_MODEL_NAME")
        
        # Initialize state manager
        from .state_manager import StateManager
        db_path = os.path.expanduser("~/.mcts_mcp_server/state.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _global_state["state_manager"] = StateManager(db_path)
        
        _global_state["initialized"] = True
        print("MCTS components loaded", file=sys.stderr)
        
    except Exception as e:
        print(f"Lazy init error: {e}", file=sys.stderr)
        logger.error(f"Lazy initialization failed: {e}")

def get_or_create_background_loop():
    """Get or create a background event loop (lazy)."""
    global _global_state
    
    if _global_state["background_loop"] is None or _global_state["background_thread"] is None:
        loop_created = threading.Event()
        loop_container = {"loop": None}
        
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
        
        # Wait for loop to be created
        if not loop_created.wait(timeout=3.0):
            raise RuntimeError("Failed to create background event loop")
    
    return _global_state["background_loop"]

def run_in_background_loop(coro):
    """Run a coroutine in the background event loop."""
    loop = get_or_create_background_loop()
    
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=300)
    else:
        raise RuntimeError("Background event loop is not running")

def register_mcts_tools(mcp: FastMCP, db_path: str):
    """
    Register all MCTS-related tools with minimal startup delay.
    Heavy initialization is deferred until tools are actually used.
    """
    global _global_state
    
    print("Registering MCTS tools (fast mode)...", file=sys.stderr)
    
    # Store db_path for lazy initialization
    _global_state["db_path"] = db_path
    
    @mcp.tool()
    def initialize_mcts(question: str, chat_id: str, provider_name: Optional[str] = None,
                       model_name: Optional[str] = None, config_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the MCTS system with lazy loading."""
        global _global_state
        
        try:
            # Trigger lazy initialization
            lazy_init()
            
            logger.info(f"Initializing MCTS for chat ID: {chat_id}")
            
            # Determine target provider and model
            target_provider = provider_name or _global_state["active_llm_provider"] or "ollama"
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
                from .ollama_adapter import OllamaAdapter
                if not target_model:
                    target_model = OllamaAdapter.DEFAULT_MODEL
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
                return {"error": f"Failed to generate initial analysis: {str(e)}", "status": "error"}
            
            # Create MCTS instance
            from .mcts_core import MCTS
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
            
        except Exception as e:
            logger.error(f"Error in initialize_mcts: {e}")
            return {"error": f"Failed to initialize MCTS: {str(e)}", "status": "error"}
    
    @mcp.tool()
    def set_active_llm(provider_name: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Set the active LLM provider and model."""
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
    def list_ollama_models() -> Dict[str, Any]:
        """List all available Ollama models (with lazy loading)."""
        try:
            # Check if Ollama server is running (quick check)
            import httpx
            with httpx.Client(base_url="http://localhost:11434", timeout=2.0) as client:
                response = client.get("/")
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": "Ollama server not responding. Please ensure Ollama is running."
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cannot connect to Ollama server: {str(e)}"
            }
        
        # Now do the heavy model checking
        try:
            from .ollama_utils import check_available_models, get_recommended_models
            available_models = check_available_models()
            
            if not available_models:
                return {
                    "status": "error",
                    "message": "No Ollama models found. Try 'ollama pull MODEL_NAME' to download a model."
                }
            
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
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing Ollama models: {str(e)}"
            }
    
    @mcp.tool()
    def run_mcts(iterations: int = 1, simulations_per_iteration: int = 5, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the MCTS algorithm."""
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
            """Run MCTS in background thread."""
            try:
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
    def generate_synthesis() -> Dict[str, Any]:
        """Generate a final synthesis of the MCTS results."""
        global _global_state
        
        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {"error": "MCTS not initialized. Call initialize_mcts first."}
        
        try:
            async def synth():
                llm_adapter = mcts.llm
                path_nodes = mcts.get_best_path_nodes()
                
                from .utils import truncate_text
                
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
            
            synthesis_result = run_in_background_loop(synth())
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return {"error": f"Synthesis generation failed: {str(e)}"}
    
    @mcp.tool()
    def get_config() -> Dict[str, Any]:
        """Get the current MCTS configuration."""
        global _global_state
        
        # Trigger lazy init if needed
        if not _global_state["initialized"]:
            lazy_init()
        
        config = {k: v for k, v in _global_state["config"].items() if not k.startswith("_")}
        config.update({
            "active_llm_provider": _global_state.get("active_llm_provider"),
            "active_model_name": _global_state.get("active_model_name"),
            "ollama_available_models": _global_state.get("ollama_available_models", []),
            "current_run_id": _global_state.get("current_run_id")
        })
        return config
    
    @mcp.tool()
    def update_config(config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update the MCTS configuration."""
        global _global_state
        
        # Trigger lazy init if needed
        if not _global_state["initialized"]:
            lazy_init()
        
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
    def get_mcts_status() -> Dict[str, Any]:
        """Get the current status of the MCTS system."""
        global _global_state
        
        mcts = _global_state.get("mcts_instance")
        if not mcts:
            return {
                "initialized": False,
                "message": "MCTS not initialized. Call initialize_mcts first."
            }
        
        try:
            from .utils import truncate_text
            
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
                "error": f"Error getting MCTS status: {str(e)}",
                "chat_id": _global_state.get("current_chat_id")
            }
    
    print("MCTS tools registered successfully", file=sys.stderr)
