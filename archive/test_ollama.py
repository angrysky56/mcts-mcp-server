#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to diagnose Ollama model detection issues
"""
import os
import sys
import subprocess
import json
import logging

# Add MCTS MCP Server to PYTHONPATH
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_ollama")

def test_subprocess_method():
    """Test listing models via subprocess call."""
    try:
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
        
        logger.info(f"Subprocess method found {len(models)} models: {models}")
        return models
    except Exception as e:
        logger.error(f"Subprocess method failed: {e}")
        return []

def test_httpx_method():
    """Test listing models via HTTP API."""
    try:
        # Try to import httpx
        import httpx
        
        client = httpx.Client(base_url="http://localhost:11434", timeout=5.0)
        response = client.get("/api/tags")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name") for m in models if m.get("name")]
            
            logger.info(f"HTTPX method found {len(model_names)} models: {model_names}")
            return model_names
        else:
            logger.error(f"HTTPX request failed with status code {response.status_code}")
            return []
    except ImportError:
        logger.error("HTTPX not installed. Cannot test HTTP API method.")
        return []
    except Exception as e:
        logger.error(f"HTTPX method failed: {e}")
        return []

def test_ollama_package():
    """Test listing models via ollama Python package."""
    try:
        # Try to import ollama
        import ollama
        
        # Log the ollama version
        ollama_version = getattr(ollama, "__version__", "unknown")
        logger.info(f"Ollama package version: {ollama_version}")
        
        # Test the list() function
        models_data = ollama.list()
        logger.info(f"Ollama package response type: {type(models_data)}")
        logger.info(f"Ollama package response: {models_data}")
        
        # Try different parsing methods based on the response format
        model_names = []
        
        # Method 1: Object with models attribute (newer API)
        if hasattr(models_data, 'models'):
            logger.info("Response has 'models' attribute")
            if isinstance(models_data.models, list):
                logger.info("models attribute is a list")
                for model in models_data.models:
                    logger.info(f"Model object type: {type(model)}")
                    logger.info(f"Model object attributes: {dir(model)}")
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                        logger.info(f"Added model name from 'model' attribute: {model.model}")
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
                        logger.info(f"Added model name from 'name' attribute: {model.name}")
        
        # Method 2: Dictionary format (older API)
        elif isinstance(models_data, dict):
            logger.info("Response is a dictionary")
            if "models" in models_data:
                logger.info("Dictionary has 'models' key")
                for m in models_data["models"]:
                    if isinstance(m, dict) and "name" in m:
                        model_names.append(m["name"])
                        logger.info(f"Added model name from dictionary: {m['name']}")
        
        # Method 3: List format
        elif isinstance(models_data, list):
            logger.info("Response is a list")
            for m in models_data:
                if isinstance(m, dict) and "name" in m:
                    model_names.append(m["name"])
                    logger.info(f"Added model name from list item: {m['name']}")
                elif hasattr(m, 'name'):
                    model_names.append(m.name)
                    logger.info(f"Added model name from list item attribute: {m.name}")
                else:
                    # Last resort, convert to string
                    model_names.append(str(m))
                    logger.info(f"Added model name as string: {str(m)}")
        
        logger.info(f"Ollama package method found {len(model_names)} models: {model_names}")
        return model_names
    except ImportError:
        logger.error("Ollama package not installed. Cannot test ollama API method.")
        return []
    except Exception as e:
        logger.error(f"Ollama package method failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def main():
    """Run all test methods and report results."""
    logger.info("====== Testing Ollama Model Detection ======")
    
    # Test subprocess method
    logger.info("--- Testing Subprocess Method ---")
    subprocess_models = test_subprocess_method()
    
    # Test HTTPX method
    logger.info("--- Testing HTTPX Method ---")
    httpx_models = test_httpx_method()
    
    # Test ollama package
    logger.info("--- Testing Ollama Package Method ---")
    package_models = test_ollama_package()
    
    # Print results
    print("\n====== RESULTS ======")
    print(f"Subprocess Method: {len(subprocess_models)} models")
    print(f"HTTPX Method: {len(httpx_models)} models")
    print(f"Ollama Package Method: {len(package_models)} models")
    
    # Check for consistency
    if subprocess_models and httpx_models and package_models:
        if set(subprocess_models) == set(httpx_models) == set(package_models):
            print("\n✅ All methods detected the same models")
        else:
            print("\n⚠️ Methods detected different sets of models")
            
            # Find differences
            all_models = set(subprocess_models + httpx_models + package_models)
            for model in all_models:
                in_subprocess = model in subprocess_models
                in_httpx = model in httpx_models
                in_package = model in package_models
                
                if not (in_subprocess and in_httpx and in_package):
                    print(f"  - '{model}': subprocess={in_subprocess}, httpx={in_httpx}, package={in_package}")
    else:
        print("\n⚠️ Some methods failed to detect models")
    
    # Output data for debugging
    result = {
        "subprocess_models": subprocess_models,
        "httpx_models": httpx_models,
        "package_models": package_models
    }
    
    print("\nDetailed Results:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
