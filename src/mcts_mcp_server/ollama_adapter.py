#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama LLM Adapter
==================

This module defines the OllamaAdapter class for interacting with local Ollama models.
"""
import logging
import os
import httpx # For direct HTTP calls if ollama package is problematic
import json
from typing import AsyncGenerator, List, Dict, Any, Optional, Union

from .base_llm_adapter import BaseLLMAdapter

# Attempt to import the official ollama package
OLLAMA_PACKAGE_AVAILABLE = False
ollama_module = None # Use a different name to avoid confusion with the type hint
try:
    import ollama as ollama_module # type: ignore
    OLLAMA_PACKAGE_AVAILABLE = True
except ImportError:
    pass # Handled in constructor

# Define the type for self.client
# Use Any for the ollama client type if it's an optional dependency
ClientType = Union[httpx.AsyncClient, Any]

class OllamaAdapter(BaseLLMAdapter):
    """
    LLM Adapter for local Ollama models.
    """
    DEFAULT_MODEL = "cogito:latest" # A common default, can be overridden

    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None, **kwargs):
        super().__init__(**kwargs) # Pass kwargs like api_key (though not used by Ollama)

        self.model_name = model_name or self.DEFAULT_MODEL
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.logger = logging.getLogger(__name__)
        self._client_type = None # To track if 'ollama' or 'httpx' is used
        self.client: ClientType # Type hint for self.client

        if OLLAMA_PACKAGE_AVAILABLE and ollama_module is not None: # Check ollama_module is not None
            try:
                self.client = ollama_module.AsyncClient(host=self.host)
                self._client_type = "ollama"
                self.logger.info(f"Initialized OllamaAdapter with model: {self.model_name} using 'ollama' package via host: {self.host}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ollama.AsyncClient (host: {self.host}): {e}. Falling back to httpx.")
                self.client = httpx.AsyncClient(base_url=self.host, timeout=60.0) # httpx fallback
                self._client_type = "httpx"
        else:
            self.logger.info(f"Ollama package not found. Initializing OllamaAdapter with model: {self.model_name} using 'httpx' for host: {self.host}")
            self.client = httpx.AsyncClient(base_url=self.host, timeout=60.0)
            self._client_type = "httpx"

        # Quick health check using httpx as it's always available here
        try:
            # This is a synchronous check for simplicity during init
            # In a fully async setup, this might be deferred or handled differently
            # Using httpx directly for the health check regardless of client type for simplicity here.
            health_check_url = f"{self.host.rstrip('/')}/"
            response = httpx.get(health_check_url)
            response.raise_for_status()
            self.logger.info(f"Ollama server health check successful for {health_check_url}")
        except Exception as e:
            self.logger.error(f"Ollama server at {self.host} is not responding: {e}")
            # We don't raise here, but get_completion/streaming will fail.

    async def get_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> str: # Removed default for model
        target_model = model if model is not None else self.model_name # Check for None explicitly
        self.logger.debug(f"Ollama get_completion using model: {target_model}, client: {self._client_type}, messages: {messages}, kwargs: {kwargs}")

        if self._client_type == "ollama" and OLLAMA_PACKAGE_AVAILABLE and ollama_module is not None: # Add ollama_module is not None
            try:
                # Ensure client is the correct type for type checker, though it should be
                if not isinstance(self.client, ollama_module.AsyncClient): # Use ollama_module
                    raise TypeError("Ollama client not initialized correctly for 'ollama' package.")

                response = await self.client.chat(
                    model=target_model,
                    messages=messages, # type: ignore
                    options=kwargs.get("options"),
                )
                return response['message']['content']
            except Exception as e:
                self.logger.error(f"Ollama package API error in get_completion: {e}", exc_info=True)
                return f"Error: Ollama package request failed - {type(e).__name__}: {e}"
        else: # Fallback or primary httpx usage
            try:
                if not isinstance(self.client, httpx.AsyncClient):
                     raise TypeError("HTTPX client not initialized correctly.")
                payload = {
                    "model": target_model,
                    "messages": messages,
                    "stream": False,
                    "options": kwargs.get("options")
                }
                response = await self.client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data['message']['content']
            except httpx.HTTPStatusError as e:
                self.logger.error(f"Ollama HTTP API error in get_completion: {e.response.text}", exc_info=True)
                return f"Error: Ollama HTTP API request failed - {e.response.status_code}: {e.response.text}"
            except Exception as e:
                self.logger.error(f"Unexpected error in Ollama (httpx) get_completion: {e}", exc_info=True)
                return f"Error: Unexpected error during Ollama (httpx) request - {type(e).__name__}: {e}"

    async def get_streaming_completion(self, model: Optional[str], messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]: # Removed default for model
        target_model = model if model is not None else self.model_name # Check for None explicitly
        self.logger.debug(f"Ollama get_streaming_completion using model: {target_model}, client: {self._client_type}, messages: {messages}, kwargs: {kwargs}")

        if self._client_type == "ollama" and OLLAMA_PACKAGE_AVAILABLE and ollama_module is not None: # Add ollama_module is not None
            try:
                if not isinstance(self.client, ollama_module.AsyncClient): # Use ollama_module
                    raise TypeError("Ollama client not initialized correctly for 'ollama' package.")

                async for part in await self.client.chat(
                    model=target_model,
                    messages=messages, # type: ignore
                    stream=True,
                    options=kwargs.get("options")
                ):
                    yield part['message']['content']
            except Exception as e:
                self.logger.error(f"Ollama package API error in get_streaming_completion: {e}", exc_info=True)
                yield f"Error: Ollama package streaming request failed - {type(e).__name__}: {e}"
        else: # Fallback or primary httpx usage
            try:
                if not isinstance(self.client, httpx.AsyncClient):
                     raise TypeError("HTTPX client not initialized correctly.")
                payload = {
                    "model": target_model,
                    "messages": messages,
                    "stream": True,
                    "options": kwargs.get("options")
                }
                async with self.client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("message") and data["message"].get("content"):
                                    yield data["message"]["content"]
                                if data.get("done") and data.get("error"): # Check for stream error part
                                    self.logger.error(f"Ollama stream error part: {data.get('error')}")
                                    yield f"Error: {data.get('error')}"
                            except json.JSONDecodeError:
                                self.logger.warning(f"Ollama stream: Could not decode JSON line: {line}")
            except httpx.HTTPStatusError as e:
                self.logger.error(f"Ollama HTTP API error in get_streaming_completion: {e.response.text}", exc_info=True)
                yield f"Error: Ollama HTTP API streaming request failed - {e.response.status_code}: {e.response.text}"
            except Exception as e:
                self.logger.error(f"Unexpected error in Ollama (httpx) get_streaming_completion: {e}", exc_info=True)
                yield f"Error: Unexpected error during Ollama (httpx) streaming request - {type(e).__name__}: {e}"

# Example usage
async def _test_ollama_adapter():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # This test assumes Ollama server is running and the model is available.
    # Example: ollama pull cogito:latest
    # You might need to set OLLAMA_HOST if it's not localhost:11434
    # os.environ["OLLAMA_HOST"] = "http://your_ollama_host:port"

    try:
        adapter = OllamaAdapter(model_name="cogito:latest")

        logger.info("Testing get_completion with OllamaAdapter...")
        messages = [{"role": "user", "content": "Why is the sky blue?"}]
        response = await adapter.get_completion(model=None, messages=messages)
        logger.info(f"Completion response: {response}")
        assert response and "Error:" not in response # Basic check

        logger.info("\nTesting get_streaming_completion with OllamaAdapter...")
        full_streamed_response = ""
        async for chunk in adapter.get_streaming_completion(model=None, messages=messages):
            logger.info(f"Stream chunk: '{chunk}'")
            full_streamed_response += chunk
        logger.info(f"Full streamed response: {full_streamed_response}")
        assert full_streamed_response and "Error:" not in full_streamed_response

        logger.info("\nTesting generate_tags (via BaseLLMAdapter) with OllamaAdapter...")
        tags_text = "The quick brown fox jumps over the lazy dog. This is a test for ollama adapter."
        tags = await adapter.generate_tags(analysis_text=tags_text, config={})
        logger.info(f"Generated tags: {tags}")
        assert tags and "Error:" not in tags[0] if tags else True


        logger.info("OllamaAdapter tests completed successfully (assuming Ollama server is running and model is available).")
    except Exception as e:
        logger.error(f"Error during OllamaAdapter test: {e}", exc_info=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_ollama_adapter())
