#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Analysis Tools for MCP Server
================================

This module provides MCP tools for analyzing MCTS results in an integrated way.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Union

from mcp.server.fastmcp import FastMCP
from .results_processor import ResultsProcessor

logger = logging.getLogger("mcts_analysis_tools")

# Global state for tools
_global_state = {
    "processor": None,
    "base_directory": None
}

def register_mcts_analysis_tools(mcp: FastMCP, results_base_dir: Optional[str] = None):
    """
    Register all MCTS analysis tools with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register tools with
        results_base_dir: Base directory for MCTS results
    """
    global _global_state
    
    # Initialize results processor
    _global_state["processor"] = ResultsProcessor(results_base_dir)
    _global_state["base_directory"] = results_base_dir
    
    @mcp.tool()
    def list_mcts_runs(count: int = 10, model: Optional[str] = None) -> Dict[str, Any]:
        """
        List recent MCTS runs with key metadata.
        
        Args:
            count: Maximum number of runs to return
            model: Optional model name to filter by
            
        Returns:
            Dictionary with list of runs and their metadata
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
            return {"error": "Results processor not initialized"}
        
        try:
            runs = processor.list_runs(count=count, model=model)
            
            return {
                "status": "success",
                "count": len(runs),
                "runs": runs,
                "base_directory": _global_state["base_directory"]
            }
        except Exception as e:
            logger.error(f"Error listing MCTS runs: {e}")
            return {"error": f"Failed to list MCTS runs: {str(e)}"}
    
    @mcp.tool()
    def get_mcts_run_details(run_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific MCTS run.
        
        Args:
            run_id: The run ID to get details for
            
        Returns:
            Dictionary with detailed run information
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
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
            
        except Exception as e:
            logger.error(f"Error getting MCTS run details: {e}")
            return {"error": f"Failed to get MCTS run details: {str(e)}"}
    
    @mcp.tool()
    def get_mcts_solution(run_id: str) -> Dict[str, Any]:
        """
        Get the best solution from an MCTS run.
        
        Args:
            run_id: The run ID to get the solution for
            
        Returns:
            Dictionary with the best solution content and metadata
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
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
            
        except Exception as e:
            logger.error(f"Error getting MCTS solution: {e}")
            return {"error": f"Failed to get MCTS solution: {str(e)}"}
    
    @mcp.tool()
    def analyze_mcts_run(run_id: str) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of an MCTS run.
        
        Args:
            run_id: The run ID to analyze
            
        Returns:
            Dictionary with analysis results
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
            return {"error": "Results processor not initialized"}
        
        try:
            analysis = processor.analyze_run(run_id)
            
            if "error" in analysis:
                return {"error": analysis["error"]}
                
            return {
                "status": "success",
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing MCTS run: {e}")
            return {"error": f"Failed to analyze MCTS run: {str(e)}"}
    
    @mcp.tool()
    def compare_mcts_runs(run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple MCTS runs to identify similarities and differences.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
            return {"error": "Results processor not initialized"}
        
        try:
            comparison = processor.compare_runs(run_ids)
            
            if "error" in comparison:
                return {"error": comparison["error"]}
                
            return {
                "status": "success",
                "comparison": comparison
            }
            
        except Exception as e:
            logger.error(f"Error comparing MCTS runs: {e}")
            return {"error": f"Failed to compare MCTS runs: {str(e)}"}
    
    @mcp.tool()
    def get_mcts_insights(run_id: str, max_insights: int = 5) -> Dict[str, Any]:
        """
        Extract key insights from an MCTS run.
        
        Args:
            run_id: The run ID to extract insights from
            max_insights: Maximum number of insights to extract
            
        Returns:
            Dictionary with key insights
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
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
                "question": metadata.get("question", "Unknown") if metadata else "Unknown",
                "model": metadata.get("model_name", "Unknown") if metadata else "Unknown",
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error extracting MCTS insights: {e}")
            return {"error": f"Failed to extract MCTS insights: {str(e)}"}
    
    @mcp.tool()
    def suggest_mcts_improvements(run_id: str) -> Dict[str, Any]:
        """
        Suggest improvements for MCTS runs based on analysis.
        
        Args:
            run_id: The run ID to analyze
            
        Returns:
            Dictionary with improvement suggestions
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
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
                "model": metadata.get("model_name", "Unknown") if metadata else "Unknown",
                "score": metadata.get("results", {}).get("best_score", 0) if metadata else 0,
                "current_config": config,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error suggesting MCTS improvements: {e}")
            return {"error": f"Failed to suggest MCTS improvements: {str(e)}"}
            
    @mcp.tool()
    def get_mcts_report(run_id: str, format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a comprehensive report for an MCTS run.
        
        Args:
            run_id: The run ID to generate a report for
            format: Output format ('markdown', 'text', or 'html')
            
        Returns:
            Dictionary with the formatted report
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
            return {"error": "Results processor not initialized"}
        
        try:
            report = processor.generate_report(run_id, format=format)
            
            if report.startswith("Error:"):
                return {"error": report}
                
            # Get basic run info for context
            run_details = processor.get_run_details(run_id)
            metadata = run_details.get("metadata", {}) if run_details else {}
            
            return {
                "status": "success",
                "run_id": run_id,
                "model": metadata.get("model_name", "Unknown") if metadata else "Unknown",
                "format": format,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Error generating MCTS report: {e}")
            return {"error": f"Failed to generate MCTS report: {str(e)}"}
    
    @mcp.tool()
    def get_best_mcts_runs(count: int = 5, min_score: float = 7.0) -> Dict[str, Any]:
        """
        Get the best MCTS runs based on score.
        
        Args:
            count: Maximum number of runs to return
            min_score: Minimum score threshold
            
        Returns:
            Dictionary with list of best run analyses
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
            return {"error": "Results processor not initialized"}
        
        try:
            best_runs = processor.get_best_runs(count=count, min_score=min_score)
            
            return {
                "status": "success",
                "count": len(best_runs),
                "min_score": min_score,
                "runs": best_runs
            }
            
        except Exception as e:
            logger.error(f"Error getting best MCTS runs: {e}")
            return {"error": f"Failed to get best MCTS runs: {str(e)}"}
    
    @mcp.tool()
    def extract_mcts_conclusions(run_id: str) -> Dict[str, Any]:
        """
        Extract conclusions from an MCTS run.
        
        Args:
            run_id: The run ID to extract conclusions from
            
        Returns:
            Dictionary with the extracted conclusions
        """
        global _global_state
        
        processor = _global_state["processor"]
        if not processor:
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
            
        except Exception as e:
            logger.error(f"Error extracting MCTS conclusions: {e}")
            return {"error": f"Failed to extract MCTS conclusions: {str(e)}"}
