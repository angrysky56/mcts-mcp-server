#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Utilities for MCTS
=========================

This module provides utility functions and constants for interacting with Ollama.
"""
import logging
import os
import sys
import importlib.util
import subprocess
import httpx # Used by check_available_models
import json # Used by check_available_models
from typing import List, Dict, Any # Optional was unused

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Ollama Adapter Import Logic (copied from tools.py) ---
OLLAMA_AVAILABLE = False
OllamaAdapter = None # Placeholder

# Add all possible import paths
current_dir_ollama_utils = os.path.dirname(os.path.abspath(__file__))
if current_dir_ollama_utils not in sys.path:
    sys.path.append(current_dir_ollama_utils)

# Strategy 1: Direct module import
try:
    from ollama_adapter import OllamaAdapter
    OLLAMA_AVAILABLE = True
    logger.info("Successfully imported OllamaAdapter (direct) in ollama_utils")
except ImportError as e:
    logger.debug(f"Failed direct import of OllamaAdapter in ollama_utils: {e}")
    # Strategy 2: Package import (assuming ollama_adapter might be in the same package)
    try:
        from .ollama_adapter import OllamaAdapter # Relative import if in the same package
        OLLAMA_AVAILABLE = True
        logger.info("Successfully imported OllamaAdapter (package relative) in ollama_utils")
    except ImportError as e_pkg:
        logger.debug(f"Failed package relative import of OllamaAdapter in ollama_utils: {e_pkg}")
        # Strategy 3: Manual module loading (if ollama_adapter.py is alongside this file)
        try:
            adapter_path = os.path.join(current_dir_ollama_utils, "ollama_adapter.py")
            if os.path.exists(adapter_path):
                spec = importlib.util.spec_from_file_location("ollama_adapter", adapter_path)
                if spec and spec.loader:
                    ollama_adapter_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ollama_adapter_module)
                    OllamaAdapter = ollama_adapter_module.OllamaAdapter
                    OLLAMA_AVAILABLE = True
                    logger.info("Successfully imported OllamaAdapter (manual load) in ollama_utils")
                else:
                    logger.debug(f"Manual load spec invalid for {adapter_path}")
            else:
                logger.debug(f"Adapter file not found at {adapter_path} for manual load")
        except Exception as e_man:
            logger.debug(f"Failed manual import of OllamaAdapter in ollama_utils: {e_man}")

if not OLLAMA_AVAILABLE:
    logger.warning("OllamaAdapter could not be imported in ollama_utils. Ollama features will be unavailable.")
    class OllamaAdapterPlaceholder: # Define a placeholder if import fails
        def __init__(self, *args, **kwargs):
            logger.error("OllamaAdapterPlaceholder is being used because OllamaAdapter failed to load.")
            raise ImportError("OllamaAdapter could not be loaded.")
    OllamaAdapter = OllamaAdapterPlaceholder

# Force Ollama availability check (runtime check for ollama package itself)
if OLLAMA_AVAILABLE:
    try:
        import ollama
        ollama_version = getattr(ollama, "__version__", "unknown")
        logger.info(f"Ollama package version: {ollama_version} (checked in ollama_utils)")
    except ImportError as e:
        logger.warning(f"Ollama package not available despite adapter import: {e} (in ollama_utils)")
        OLLAMA_AVAILABLE = False # Correctly set if 'import ollama' fails
    except Exception as e:
        logger.warning(f"Error testing Ollama package: {e} (in ollama_utils)")
        OLLAMA_AVAILABLE = False


# --- Model Constants (copied from tools.py) ---
SMALL_MODELS = ["qwen3:0.6b", "deepseek-r1:1.5b", "cogito:latest", "phi3:mini", "tinyllama", "phi2:2b", "qwen2:1.5b"]
MEDIUM_MODELS = ["mistral:7b", "llama3:8b", "gemma:7b", "mistral-nemo:7b"]
DEFAULT_MODEL = "qwen3:0.6b"

# --- Functions (copied and adapted from tools.py) ---

def check_available_models() -> List[str]:
    """Check which Ollama models are available locally. Returns a list of model names."""
    if not OLLAMA_AVAILABLE:
        logger.warning("Ollama is not available, can't check models (ollama_utils)")
        return []

    try:
        client = httpx.Client(base_url="http://localhost:11434", timeout=3.0)
        response = client.get("/")
        if response.status_code != 200:
            logger.error(f"Ollama server health check failed: {response.status_code} (ollama_utils)")
            return []
        logger.info("Ollama server is running (ollama_utils)")
    except Exception as e:
        logger.error(f"Ollama server health check failed: {e}. Server might not be running. (ollama_utils)")
        return []

    available_models: List[str] = []

    # Method 1: Subprocess
    try:
        cmd = ['ollama.exe', 'list'] if sys.platform == 'win32' else ['ollama', 'list']
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1 and "NAME" in lines[0].upper() and "ID" in lines[0].upper(): # Make header check case-insensitive
                lines = lines[1:]

            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    if ':' not in model_name: model_name += ':latest'
                    available_models.append(model_name)
            if available_models:
                logger.info(f"Available Ollama models via subprocess: {available_models} (ollama_utils)")
                return available_models
        else:
            logger.warning(f"Ollama list command failed (code {result.returncode}): {result.stderr} (ollama_utils)")
    except Exception as e:
        logger.warning(f"Subprocess 'ollama list' failed: {e} (ollama_utils)")

    # Method 2: HTTP API
    try:
        client = httpx.Client(base_url="http://localhost:11434", timeout=5.0)
        response = client.get("/api/tags")
        if response.status_code == 200:
            data = response.json()
            models_data = data.get("models", [])
            api_models = [m.get("name") for m in models_data if m.get("name")]
            if api_models:
                logger.info(f"Available Ollama models via HTTP API: {api_models} (ollama_utils)")
                return api_models
        else:
            logger.warning(f"Failed to get models from Ollama API: {response.status_code} (ollama_utils)")
    except Exception as e:
        logger.warning(f"HTTP API for Ollama models failed: {e} (ollama_utils)")

    # Method 3: Ollama package (if subprocess and API failed)
    # This is often redundant if OLLAMA_AVAILABLE is true, but acts as another fallback.
    if OLLAMA_AVAILABLE: # Check again, as it might have been set to False by earlier checks
        try:
            import ollama
            models_response = ollama.list() # This might return a more complex object

            package_models = []
            if hasattr(models_response, 'models') and hasattr(models_response.models, '__iter__'): # Modern ollama package
                for model_obj in models_response.models:
                    if hasattr(model_obj, 'model') and isinstance(model_obj.model, str): # New Pydantic field
                        package_models.append(model_obj.model)
                    elif hasattr(model_obj, 'name') and isinstance(model_obj.name, str): # Older field
                        package_models.append(model_obj.name)
            elif isinstance(models_response, dict) and "models" in models_response: # Older dict format
                 for model_dict in models_response["models"]:
                    if isinstance(model_dict, dict) and "name" in model_dict:
                        package_models.append(model_dict["name"])

            if package_models:
                logger.info(f"Available Ollama models via ollama package: {package_models} (ollama_utils)")
                return package_models
        except Exception as e:
            logger.warning(f"Ollama package 'list()' method failed: {e} (ollama_utils)")

    logger.warning("All methods to list Ollama models failed or returned no models. (ollama_utils)")
    return []


def get_recommended_models(models: List[str]) -> Dict[str, List[str]]:
    """Get a list of recommended models from available models, categorized by size."""
    small_recs = [model for model in SMALL_MODELS if model in models]
    medium_recs = [model for model in MEDIUM_MODELS if model in models]
    other_models = [m for m in models if m not in small_recs and m not in medium_recs]

    return {
        "small_models": small_recs,
        "medium_models": medium_recs,
        "other_models": other_models,
        "all_models": models # Return all detected models as well
    }
