#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Utilities for MCTS
=========================

This module provides utility functions and constants for interacting with Ollama.
"""
import logging
<<<<<<< HEAD
import sys
import subprocess
import httpx # Used by check_available_models
from typing import List, Dict # Optional was unused
=======
import os
import sys
import importlib.util
import subprocess
import httpx # Used by check_available_models
import json # Used by check_available_models
from typing import List, Dict, Any # Optional was unused
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)

# Setup logger for this module
logger = logging.getLogger(__name__)

# Check if the 'ollama' Python package is installed.
# This is different from OllamaAdapter availability.
OLLAMA_PYTHON_PACKAGE_AVAILABLE = False
try:
    import ollama # type: ignore
    OLLAMA_PYTHON_PACKAGE_AVAILABLE = True
    logger.info(f"Ollama python package version: {getattr(ollama, '__version__', 'unknown')}")
except ImportError:
    logger.info("Ollama python package not found. Some features of check_available_models might be limited.")
except Exception as e:
    logger.warning(f"Error importing or checking ollama package version: {e}")


# --- Model Constants for get_recommended_models ---
SMALL_MODELS = ["qwen3:0.6b", "deepseek-r1:1.5b", "cogito:latest", "phi3:mini", "tinyllama", "phi2:2b", "qwen2:1.5b"]
MEDIUM_MODELS = ["mistral:7b", "llama3:8b", "gemma:7b", "mistral-nemo:7b"]
# DEFAULT_MODEL for an adapter is now defined in the adapter itself.

# --- Functions ---

def check_available_models() -> List[str]:
    """Check which Ollama models are available locally. Returns a list of model names."""
    # This function no longer relies on a global OLLAMA_AVAILABLE specific to the adapter,
    # but can use OLLAMA_PYTHON_PACKAGE_AVAILABLE for its 'ollama' package dependent part.
    # The primary check is if the Ollama server is running.
    # This function no longer relies on a global OLLAMA_AVAILABLE specific to the adapter,
    # but can use OLLAMA_PYTHON_PACKAGE_AVAILABLE for its 'ollama' package dependent part.
    # The primary check is if the Ollama server is running.

    try:
        # Use httpx for the initial server health check, as it's a direct dependency of this file.
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
<<<<<<< HEAD
                if not line.strip():
                    continue
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    if ':' not in model_name:
                        model_name += ':latest'
=======
                if not line.strip(): continue
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    if ':' not in model_name: model_name += ':latest'
>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
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
    if OLLAMA_PYTHON_PACKAGE_AVAILABLE:
        try:
            # This import is already tried at the top, but to be safe if logic changes:
            import ollama # type: ignore
            models_response = ollama.list()

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

<<<<<<< HEAD
=======

>>>>>>> fbff161 (feat: Convert project to uv with pyproject.toml)
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
