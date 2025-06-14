#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Analysis Tools for MCP Server
================================

This module provides MCP tools for analyzing MCTS results in an integrated way.
"""

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from .results_processor import ResultsProcessor

logger = logging.getLogger("mcts_analysis_tools")

# Global state for tools
_global_state: dict[str, ResultsProcessor | str | None] = {
    "processor": None,
    "base_directory": None
}

def register_mcts_analysis_tools(mcp: FastMCP, results_base_dir: Optional[str] = None) -> None:
    """
    Register all MCTS analysis tools with the MCP server.

    Args:
        mcp: The FastMCP instance to register tools with
        results_base_dir: Base directory for MCTS results (uses default if None)

    Note:
        Initializes global state including the ResultsProcessor instance
        All registered tools depend on this initialization
    """
    global _global_state

    # Initialize results processor
    _global_state["processor"] = ResultsProcessor(results_base_dir)
    _global_state["base_directory"] = results_base_dir

    @mcp.tool()
    def list_mcts_runs(count: int = 10, model: Optional[str] = None) -> dict[str, Any]:
        """
        List recent MCTS runs with key metadata.

        Args:
            count: Maximum number of runs to return (default: 10)
            model: Optional model name to filter by

        Returns:
            Dictionary containing:
            - status: Success or error status
            - count: Number of runs found
            - runs: List of run metadata
            - base_directory: Results base directory path

        Note:
            Returns most recent runs first, filtered by model if specified
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            runs = processor.list_runs(count=count, model=model)

            return {
                "status": "success",
                "count": len(runs),
                "runs": runs,
                "base_directory": _global_state["base_directory"]
            }
        except (OSError, ValueError) as e:
            logger.error("Error listing MCTS runs: %s", e)
            return {"error": f"Failed to list MCTS runs: {str(e)}"}

    @mcp.tool()
    def get_mcts_run_details(run_id: str) -> dict[str, Any]:
        """
        Get detailed information about a specific MCTS run.

        Args:
            run_id: The unique run identifier

        Returns:
            Dictionary containing:
            - status: Success or error status
            - details: Comprehensive run information including:
              - run_id, model, question, timestamp, status
              - score, tags, iterations, simulations, duration
              - has_solution: Boolean indicating solution availability
              - progress_count: Number of progress entries
            - path: File system path to run data

        Note:
            Excludes full solution content for efficiency - use get_mcts_solution for that
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            run_details = processor.get_run_details(run_id)
            if not run_details:
                return {"error": f"Run not found: {run_id}"}

            # Clean output for better display
            if "metadata" in run_details:
                metadata = run_details["metadata"]

                # Extract key fields for easier access
                results = {
                    "run_id": run_id,
                    "model": metadata.get("model_name", "Unknown"),
                    "question": metadata.get("question", "Unknown"),
                    "timestamp": metadata.get("timestamp_readable", "Unknown"),
                    "status": metadata.get("status", "Unknown"),
                    "score": metadata.get("results", {}).get("best_score", 0),
                    "tags": metadata.get("results", {}).get("tags", []),
                    "iterations": metadata.get("results", {}).get("iterations_completed", 0),
                    "simulations": metadata.get("results", {}).get("simulations_completed", 0),
                    "duration": metadata.get("duration_seconds", 0),
                }

                # For efficiency, don't include the full solution in this overview
                results["has_solution"] = bool(run_details.get("best_solution", ""))

                # Statistics about progress
                results["progress_count"] = len(run_details.get("progress", []))

                return {
                    "status": "success",
                    "details": results,
                    "path": run_details.get("run_path", "")
                }

            return {
                "status": "success",
                "details": run_details
            }

        except (OSError, ValueError, KeyError) as e:
            logger.error("Error getting MCTS run details: %s", e)
            return {"error": f"Failed to get MCTS run details: {str(e)}"}

    @mcp.tool()
    def get_mcts_solution(run_id: str) -> dict[str, Any]:
        """
        Get the best solution from an MCTS run.

        Args:
            run_id: The unique run identifier

        Returns:
            Dictionary containing:
            - status: Success or error status
            - run_id: Run identifier
            - question: Original question analyzed
            - model: Model used for analysis
            - score: Best score achieved
            - tags: Descriptive tags from analysis
            - solution: Full solution content

        Note:
            Returns the complete best solution text - may be large for complex analyses
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            run_details = processor.get_run_details(run_id)
            if not run_details:
                return {"error": f"Run not found: {run_id}"}

            # Extract solution and metadata
            best_solution = run_details.get("best_solution", "")
            metadata = run_details.get("metadata", {})

            return {
                "status": "success",
                "run_id": run_id,
                "question": metadata.get("question", "Unknown"),
                "model": metadata.get("model_name", "Unknown"),
                "score": metadata.get("results", {}).get("best_score", 0),
                "tags": metadata.get("results", {}).get("tags", []),
                "solution": best_solution
            }

        except (OSError, ValueError, KeyError) as e:
            logger.error("Error getting MCTS solution: %s", e)
            return {"error": f"Failed to get MCTS solution: {str(e)}"}

    @mcp.tool()
    def analyze_mcts_run(run_id: str) -> dict[str, Any]:
        """
        Perform a comprehensive analysis of an MCTS run.

        Args:
            run_id: The unique run identifier

        Returns:
            Dictionary containing:
            - status: Success or error status
            - analysis: Comprehensive analysis results from ResultsProcessor

        Note:
            Provides deep analytical insights including performance patterns,
            approach effectiveness, and search behavior analysis
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            analysis = processor.analyze_run(run_id)

            if "error" in analysis:
                return {"error": analysis["error"]}

            return {
                "status": "success",
                "analysis": analysis
            }

        except (OSError, ValueError) as e:
            logger.error("Error analyzing MCTS run: %s", e)
            return {"error": f"Failed to analyze MCTS run: {str(e)}"}

    @mcp.tool()
    def compare_mcts_runs(run_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Compare multiple MCTS runs to identify similarities and differences.

        Args:
            run_ids: List of run identifiers to compare (2-10 recommended)

        Returns:
            Dictionary containing:
            - status: Success or error status
            - comparison: Detailed comparison results including:
              - Performance differences
              - Approach similarities
              - Score distributions
              - Model effectiveness comparisons

        Note:
            Most effective with 2-5 runs; too many runs may produce overwhelming output
        """
        if run_ids is None:
            run_ids = []

        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            comparison = processor.compare_runs(run_ids)

            if "error" in comparison:
                return {"error": comparison["error"]}

            return {
                "status": "success",
                "comparison": comparison
            }

        except (OSError, ValueError) as e:
            logger.error("Error comparing MCTS runs: %s", e)
            return {"error": f"Failed to compare MCTS runs: {str(e)}"}

    @mcp.tool()
    def get_mcts_insights(run_id: str, max_insights: int = 5) -> dict[str, Any]:
        """
        Extract key insights from an MCTS run.

        Args:
            run_id: The unique run identifier
            max_insights: Maximum number of insights to extract (default: 5)

        Returns:
            Dictionary containing:
            - status: Success or error status
            - run_id: Run identifier
            - question: Original question
            - model: Model used
            - insights: List of key analytical insights

        Note:
            Focuses on the most important findings and patterns from the analysis
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            insights = processor.extract_insights(run_id, max_insights=max_insights)

            if insights and insights[0].startswith("Error:"):
                return {"error": insights[0]}

            # Get basic run info for context
            run_details = processor.get_run_details(run_id)
            metadata = run_details.get("metadata", {}) if run_details else {}

            return {
                "status": "success",
                "run_id": run_id,
                "question": metadata.get("question", "Unknown"),
                "model": metadata.get("model_name", "Unknown"),
                "insights": insights
            }

        except (OSError, ValueError) as e:
            logger.error("Error extracting MCTS insights: %s", e)
            return {"error": f"Failed to extract MCTS insights: {str(e)}"}

    @mcp.tool()
    def suggest_mcts_improvements(run_id: str) -> dict[str, Any]:
        """
        Suggest improvements for MCTS runs based on analysis.

        Args:
            run_id: The unique run identifier

        Returns:
            Dictionary containing:
            - status: Success or error status
            - run_id: Run identifier
            - model: Model used
            - score: Best score achieved
            - current_config: Current MCTS configuration
            - suggestions: List of improvement recommendations

        Note:
            Analyzes performance patterns to suggest configuration changes,
            different approaches, or alternative strategies
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            suggestions = processor.suggest_improvements(run_id)

            if suggestions and suggestions[0].startswith("Error:"):
                return {"error": suggestions[0]}

            # Get basic run info for context
            run_details = processor.get_run_details(run_id)
            metadata = run_details.get("metadata", {}) if run_details else {}
            config = metadata.get("config", {}) if metadata else {}

            return {
                "status": "success",
                "run_id": run_id,
                "model": metadata.get("model_name", "Unknown"),
                "score": metadata.get("results", {}).get("best_score", 0),
                "current_config": config,
                "suggestions": suggestions
            }

        except (OSError, ValueError) as e:
            logger.error("Error suggesting MCTS improvements: %s", e)
            return {"error": f"Failed to suggest MCTS improvements: {str(e)}"}

    @mcp.tool()
    def get_mcts_report(run_id: str, report_format: str = "markdown") -> dict[str, Any]:
        """
        Generate a comprehensive report for an MCTS run.

        Args:
            run_id: The unique run identifier
            report_format: Output format - 'markdown', 'text', or 'html' (default: 'markdown')

        Returns:
            Dictionary containing:
            - status: Success or error status
            - run_id: Run identifier
            - model: Model used
            - format: Report format used
            - report: Formatted report content

        Note:
            Generates publication-ready reports suitable for documentation or sharing
            Markdown format is recommended for flexibility
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            report = processor.generate_report(run_id, format=report_format)

            if report.startswith("Error:"):
                return {"error": report}

            # Get basic run info for context
            run_details = processor.get_run_details(run_id)
            metadata = run_details.get("metadata", {}) if run_details else {}

            return {
                "status": "success",
                "run_id": run_id,
                "model": metadata.get("model_name", "Unknown"),
                "format": report_format,
                "report": report
            }

        except (OSError, ValueError) as e:
            logger.error("Error generating MCTS report: %s", e)
            return {"error": f"Failed to generate MCTS report: {str(e)}"}

    @mcp.tool()
    def get_best_mcts_runs(count: int = 5, min_score: float = 7.0) -> dict[str, Any]:
        """
        Get the best MCTS runs based on score threshold.

        Args:
            count: Maximum number of runs to return (default: 5)
            min_score: Minimum score threshold for inclusion (default: 7.0)

        Returns:
            Dictionary containing:
            - status: Success or error status
            - count: Number of runs found
            - min_score: Score threshold used
            - runs: List of best run analyses with metadata

        Note:
            Useful for identifying successful analysis patterns and high-quality results
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            best_runs = processor.get_best_runs(count=count, min_score=min_score)

            return {
                "status": "success",
                "count": len(best_runs),
                "min_score": min_score,
                "runs": best_runs
            }

        except (OSError, ValueError) as e:
            logger.exception("Error getting best MCTS runs")
            return {"error": f"Failed to get best MCTS runs: {e!s}"}

    @mcp.tool()
    def extract_mcts_conclusions(run_id: str) -> dict[str, Any]:
        """
        Extract actionable conclusions from an MCTS run.

        Args:
            run_id: The unique run identifier

        Returns:
            Dictionary containing:
            - status: Success or error status
            - run_id: Run identifier
            - question: Original question
            - model: Model used
            - conclusions: List of actionable conclusions

        Note:
            Focuses on practical takeaways and actionable insights rather than
            technical analysis details
        """
        processor = _global_state["processor"]
        if not isinstance(processor, ResultsProcessor):
            return {"error": "Results processor not initialized"}

        try:
            # Get run details
            run_details = processor.get_run_details(run_id)
            if not run_details:
                return {"error": f"Run not found: {run_id}"}

            # Extract solution and progress
            best_solution = run_details.get("best_solution", "")
            progress = run_details.get("progress", [])
            metadata = run_details.get("metadata", {})

            # Extract conclusions
            conclusions = processor.extract_conclusions(best_solution, progress)

            return {
                "status": "success",
                "run_id": run_id,
                "question": metadata.get("question", "Unknown"),
                "model": metadata.get("model_name", "Unknown"),
                "conclusions": conclusions
            }

        except (OSError, ValueError) as e:
            logger.error("Error extracting MCTS conclusions: %s", e)
            return {"error": f"Failed to extract MCTS conclusions: {str(e)}"}
            progress = run_details.get("progress", [])
            metadata = run_details.get("metadata", {})

            # Extract conclusions
            conclusions = processor.extract_conclusions(best_solution, progress)

            return {
                "status": "success",
                "run_id": run_id,
                "question": metadata.get("question", "Unknown"),
                "model": metadata.get("model_name", "Unknown"),
                "conclusions": conclusions
            }

        except Exception as e:
            logger.error(f"Error extracting MCTS conclusions: {e}")
            return {"error": f"Failed to extract MCTS conclusions: {str(e)}"}
