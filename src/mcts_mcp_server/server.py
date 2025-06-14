#!/usr/bin/env python3
"""
Simple MCTS MCP Server - Basic Implementation
============================================

A working Monte Carlo Tree Search server using basic MCP server.
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server
server = Server("mcts-server")

# Simple state storage
server_state = {
    "current_question": None,
    "chat_id": None,
    "provider": "gemini",
    "model": "gemini-2.0-flash-lite",
    "iterations_completed": 0,
    "best_score": 0.0,
    "best_analysis": "",
    "initialized": False
}

def get_gemini_client():
    """Get a Gemini client if API key is available."""
    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None

async def call_llm(prompt: str) -> str:
    """Call the configured LLM with a prompt."""
    try:
        client = get_gemini_client()
        if not client:
            return "Error: No Gemini API key configured. Set GEMINI_API_KEY environment variable."

        # Use the async API properly
        response = await client.aio.models.generate_content(
            model=server_state["model"],
            contents=[{
                'role': 'user',
                'parts': [{'text': prompt}]
            }]
        )

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                return text if text is not None else "No response generated."

        return "No response generated."

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error calling LLM: {e!s}"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="initialize_mcts",
            description="Initialize MCTS for a question",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to analyze"},
                    "chat_id": {"type": "string", "description": "Unique identifier for this conversation", "default": "default"},
                    "provider": {"type": "string", "description": "LLM provider", "default": "gemini"},
                    "model": {"type": "string", "description": "Model name (optional)"}
                },
                "required": ["question"]
            }
        ),
        types.Tool(
            name="run_mcts_search",
            description="Run MCTS search iterations",
            inputSchema={
                "type": "object",
                "properties": {
                    "iterations": {"type": "integer", "description": "Number of search iterations (1-10)", "default": 3},
                    "simulations_per_iteration": {"type": "integer", "description": "Simulations per iteration (1-20)", "default": 5}
                }
            }
        ),
        types.Tool(
            name="get_synthesis",
            description="Generate a final synthesis of the MCTS results",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="get_status",
            description="Get the current MCTS status",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="set_provider",
            description="Set the LLM provider and model",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Provider name", "default": "gemini"},
                    "model": {"type": "string", "description": "Model name (optional)"}
                }
            }
        ),
        types.Tool(
            name="list_available_models",
            description="List available models for a provider",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Provider name", "default": "gemini"}
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Handle tool calls."""
    if arguments is None:
        arguments = {}

    try:
        if name == "initialize_mcts":
            result = await initialize_mcts(**arguments)
        elif name == "run_mcts_search":
            result = await run_mcts_search(**arguments)
        elif name == "get_synthesis":
            result = await get_synthesis()
        elif name == "get_status":
            result = get_status()
        elif name == "set_provider":
            result = set_provider(**arguments)
        elif name == "list_available_models":
            result = list_available_models(**arguments)
        else:
            result = {"error": f"Unknown tool: {name}", "status": "error"}

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        error_result = {"error": f"Tool execution failed: {e!s}", "status": "error"}
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]

async def initialize_mcts(
    question: str,
    chat_id: str = "default",
    provider: str = "gemini",
    model: str | None = None
) -> dict[str, Any]:
    """
    Initialize MCTS for a question.

    Args:
        question: The question or topic to analyze
        chat_id: Unique identifier for this conversation session
        provider: LLM provider to use (currently only 'gemini' supported)
        model: Specific model name to use (optional, defaults to gemini-2.0-flash-lite)

    Returns:
        Dict containing initialization status, configuration, and any error messages

    Raises:
        Exception: If initialization fails due to missing API key or other errors
    """
    try:
        # Validate inputs
        if not question.strip():
            return {"error": "Question cannot be empty", "status": "error"}

        if provider.lower() != "gemini":
            return {"error": "Only 'gemini' provider is currently supported", "status": "error"}

        # Check if API key is available
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "error": "GEMINI_API_KEY or GOOGLE_API_KEY environment variable required",
                "status": "error",
                "setup_help": "Set your API key with: export GEMINI_API_KEY='your-key-here'"
            }

        # Update state
        server_state.update({
            "current_question": question,
            "chat_id": chat_id,
            "provider": provider.lower(),
            "model": model or "gemini-2.0-flash-lite",
            "iterations_completed": 0,
            "best_score": 0.0,
            "best_analysis": "",
            "initialized": True
        })

        logger.info(f"Initialized MCTS for question: {question[:50]}...")

        return {
            "status": "initialized",
            "question": question,
            "chat_id": chat_id,
            "provider": server_state["provider"],
            "model": server_state["model"],
            "message": "MCTS initialized successfully. Use run_mcts_search to begin analysis."
        }

    except Exception as e:
        logger.error(f"Error initializing MCTS: {e}")
        return {"error": f"Initialization failed: {e!s}", "status": "error"}

async def run_mcts_search(
    iterations: int = 3,
    simulations_per_iteration: int = 5
) -> dict[str, Any]:
    """
    Run MCTS search iterations to explore different analytical approaches.

    Args:
        iterations: Number of search iterations to run (1-10, clamped to range)
        simulations_per_iteration: Number of simulations per iteration (1-20, clamped to range)

    Returns:
        Dict containing search results including best analysis, scores, and statistics

    Raises:
        Exception: If search fails due to LLM errors or other issues
    """
    if not server_state["initialized"]:
        return {"error": "MCTS not initialized. Call initialize_mcts first.", "status": "error"}

    # Validate parameters
    iterations = max(1, min(10, iterations))
    simulations_per_iteration = max(1, min(20, simulations_per_iteration))

    try:
        question = server_state["current_question"]

        # Generate multiple analysis approaches
        analyses = []

        for i in range(iterations):
            logger.info(f"Running iteration {i+1}/{iterations}")

            for j in range(simulations_per_iteration):
                # Create a prompt for this simulation
                if i == 0 and j == 0:
                    # Initial analysis
                    prompt = f"""Provide a thoughtful analysis of this question: {question}

Focus on being insightful, comprehensive, and offering unique perspectives."""
                else:
                    # Varied approaches for subsequent simulations
                    approaches = [
                        "from a practical perspective",
                        "considering potential counterarguments",
                        "examining underlying assumptions",
                        "exploring alternative solutions",
                        "analyzing long-term implications",
                        "considering different stakeholder viewpoints",
                        "examining historical context",
                        "thinking about implementation challenges"
                    ]
                    approach = approaches[(i * simulations_per_iteration + j) % len(approaches)]

                    prompt = f"""Analyze this question {approach}: {question}

Previous best analysis (score {server_state['best_score']:.1f}/10):
{server_state['best_analysis'][:200]}...

Provide a different angle or deeper insight."""

                # Get analysis
                analysis = await call_llm(prompt)

                # Score the analysis
                score_prompt = f"""Rate the quality and insight of this analysis on a scale of 1-10:

Question: {question}
Analysis: {analysis}

Consider: depth, originality, practical value, logical consistency.
Respond with just a number from 1-10."""

                score_response = await call_llm(score_prompt)

                # Parse score
                try:
                    import re
                    score_matches = re.findall(r'\b([1-9]|10)\b', score_response)
                    score = float(score_matches[0]) if score_matches else 5.0
                except (ValueError, IndexError, TypeError):
                    score = 5.0

                analyses.append({
                    "iteration": i + 1,
                    "simulation": j + 1,
                    "analysis": analysis,
                    "score": score
                })

                # Update best if this is better
                if score > server_state["best_score"]:
                    server_state["best_score"] = score
                    server_state["best_analysis"] = analysis

                logger.info(f"Simulation {j+1} completed with score: {score:.1f}")

            server_state["iterations_completed"] = i + 1

        # Find the best analysis
        best_analysis = max(analyses, key=lambda x: x["score"])

        return {
            "status": "completed",
            "iterations_completed": iterations,
            "total_simulations": len(analyses),
            "best_score": server_state["best_score"],
            "best_analysis": server_state["best_analysis"],
            "best_from_this_run": best_analysis,
            "all_scores": [a["score"] for a in analyses],
            "average_score": sum(a["score"] for a in analyses) / len(analyses),
            "provider": server_state["provider"],
            "model": server_state["model"]
        }

    except Exception as e:
        logger.error(f"Error during MCTS search: {e}")
        return {"error": f"Search failed: {e!s}", "status": "error"}

async def get_synthesis() -> dict[str, Any]:
    """
    Generate a final synthesis of the MCTS results.

    Creates a comprehensive summary that synthesizes the key insights from the best
    analysis found during the MCTS search process.

    Returns:
        Dict containing the synthesis text, best score, and metadata

    Raises:
        Exception: If synthesis generation fails or MCTS hasn't been run yet
    """
    if not server_state["initialized"]:
        return {"error": "MCTS not initialized. Call initialize_mcts first.", "status": "error"}

    if server_state["best_score"] == 0.0:
        return {"error": "No analysis completed yet. Run run_mcts_search first.", "status": "error"}

    try:
        question = server_state["current_question"]
        best_analysis = server_state["best_analysis"]
        best_score = server_state["best_score"]

        synthesis_prompt = f"""Create a comprehensive synthesis based on this MCTS analysis:

Original Question: {question}

Best Analysis Found (Score: {best_score}/10):
{best_analysis}

Provide a final synthesis that:
1. Summarizes the key insights
2. Highlights the most important findings
3. Offers actionable conclusions
4. Explains why this approach is valuable

Make it clear, comprehensive, and practical."""

        synthesis = await call_llm(synthesis_prompt)

        return {
            "synthesis": synthesis,
            "best_score": best_score,
            "iterations_completed": server_state["iterations_completed"],
            "question": question,
            "provider": server_state["provider"],
            "model": server_state["model"],
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error generating synthesis: {e}")
        return {"error": f"Synthesis failed: {e!s}", "status": "error"}

def get_status() -> dict[str, Any]:
    """
    Get the current MCTS status and configuration.

    Returns comprehensive information about the current state of the MCTS system
    including initialization status, current question, provider settings, and results.

    Returns:
        Dict containing all current status information and configuration
    """
    return {
        "initialized": server_state["initialized"],
        "current_question": server_state["current_question"],
        "chat_id": server_state["chat_id"],
        "provider": server_state["provider"],
        "model": server_state["model"],
        "iterations_completed": server_state["iterations_completed"],
        "best_score": server_state["best_score"],
        "has_analysis": bool(server_state["best_analysis"]),
        "available_providers": ["gemini"],
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    }

def set_provider(provider: str = "gemini", model: str | None = None) -> dict[str, Any]:
    """
    Set the LLM provider and model configuration.

    Args:
        provider: LLM provider name (currently only 'gemini' supported)
        model: Specific model name to use (optional)

    Returns:
        Dict containing success status and new configuration

    Note:
        Currently only supports Gemini provider. Other providers will return an error.
    """
    if provider.lower() != "gemini":
        return {"error": "Only 'gemini' provider is currently supported", "status": "error"}

    server_state["provider"] = provider.lower()
    if model:
        server_state["model"] = model

    return {
        "status": "success",
        "provider": server_state["provider"],
        "model": server_state["model"],
        "message": f"Provider set to {provider}" + (f" with model {model}" if model else "")
    }

def list_available_models(provider: str = "gemini") -> dict[str, Any]:
    """
    List available models for a given provider.

    Args:
        provider: Provider name to list models for (currently only 'gemini' supported)

    Returns:
        Dict containing available models, default model, and current configuration

    Note:
        Model availability depends on the provider. Currently only Gemini models are supported.
    """
    if provider.lower() == "gemini":
        return {
            "provider": "gemini",
            "default_model": "gemini-2.0-flash-lite",
            "available_models": [
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            "current_model": server_state["model"]
        }
    else:
        return {"error": f"Provider '{provider}' not supported", "available_providers": ["gemini"]}

async def main():
    """Main entry point."""
    try:
        logger.info("Starting Simple MCTS MCP Server... Version: 1.0, Default Provider: Gemini, Default Model: gemini-2.0-flash-lite")
        logger.info("Default provider: Gemini")
        logger.info("To use: Set GEMINI_API_KEY environment variable")

        # Run server with stdio
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

def cli_main() -> None:
    """
    Synchronous entry point for the CLI script.

    This function is called by the console script entry point in pyproject.toml
    and properly runs the async main() function.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServer shutdown initiated by user")
    except Exception as e:
        print(f"\n\nServer error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    """
    Execute the MCTS-MCP-Server.

    ### Execution Flow
    1. Initialize server instance
    2. Configure MCP handlers
    3. Start async event loop
    4. Handle graceful shutdown
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServer shutdown initiated by user")
    except Exception as e:
        print(f"\n\nServer error: {e}")
        sys.exit(1)
