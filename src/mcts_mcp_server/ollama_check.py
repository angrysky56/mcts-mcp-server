#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Model Check
===============

Simple script to test Ollama model detection.
"""
import sys
import logging
import subprocess
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ollama_check")

def check_models_subprocess():
    """Check available models using subprocess to call 'ollama list'."""
    try:
        # Run 'ollama list' and capture output
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        logger.info(f"Subprocess output: {result.stdout}")
        
        # Process the output
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Just the header line
            logger.info("No models found in subprocess output")
            return []
        
        # Skip the header line if present
        if "NAME" in lines[0] and "ID" in lines[0]:
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
        
        logger.info(f"Models found via subprocess: {models}")
        return models
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling 'ollama list': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in subprocess check: {e}")
        return []

def check_models_httpx():
    """Check available models using direct HTTP API call."""
    try:
        import httpx
        client = httpx.Client(base_url="http://localhost:11434", timeout=5.0)
        response = client.get("/api/tags")
        
        logger.info(f"HTTPX status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name") for m in models if m.get("name")]
            logger.info(f"Models found via HTTP API: {model_names}")
            return model_names
        else:
            logger.warning(f"Failed to get models via HTTP: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error checking models via HTTP: {e}")
        return []

def check_models_ollama_package():
    """Check available models using ollama Python package."""
    try:
        import ollama
        models_data = ollama.list()
        logger.info(f"Ollama package response: {models_data}")
        
        if isinstance(models_data, dict) and "models" in models_data:
            model_names = [m.get("name") for m in models_data["models"] if m.get("name")]
            logger.info(f"Models found via ollama package: {model_names}")
            return model_names
        else:
            logger.warning("Unexpected response format from ollama package")
            return []
    except Exception as e:
        logger.error(f"Error checking models via ollama package: {e}")
        return []

def main():
    """Test all methods of getting Ollama models."""
    logger.info("Testing Ollama model detection")
    
    logger.info("--- Method 1: Subprocess ---")
    subprocess_models = check_models_subprocess()
    
    logger.info("--- Method 2: HTTP API ---")
    httpx_models = check_models_httpx()
    
    logger.info("--- Method 3: Ollama Package ---")
    package_models = check_models_ollama_package()
    
    # Combine all results
    all_models = list(set(subprocess_models + httpx_models + package_models))
    
    logger.info(f"Combined unique models: {all_models}")
    
    # Output JSON result for easier parsing
    result = {
        "subprocess_models": subprocess_models,
        "httpx_models": httpx_models, 
        "package_models": package_models,
        "all_models": all_models
    }
    
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
