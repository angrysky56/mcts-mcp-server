#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Results Processor
=====================

This module provides a class for processing, analyzing, and extracting insights
from MCTS run results. It helps identify the most valuable information in MCTS
outputs and present it in a more structured and useful format.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import re
from pathlib import Path

logger = logging.getLogger("mcts_analysis")

class ResultsProcessor:
    """Processes and analyzes MCTS run results to extract key insights."""
    
    def __init__(self, results_base_dir: Optional[str] = None):
        """
        Initialize the results processor.
        
        Args:
            results_base_dir: Base directory for MCTS results. If None, defaults to
                             the standard location.
        """
        if results_base_dir is None:
            # Default to 'results' in the repository root
            repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.results_base_dir = os.path.join(repo_dir, "results")
        else:
            self.results_base_dir = results_base_dir
            
        logger.info(f"Initialized ResultsProcessor with base directory: {self.results_base_dir}")
        
        # Cache for analyzed results
        self._cache = {}
    
    def list_runs(self, count: int = 10, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List recent MCTS runs with key metadata.
        
        Args:
            count: Maximum number of runs to return
            model: Optional model name to filter by
            
        Returns:
            List of run dictionaries with key metadata
        """
        runs = []
        
        # Walk through the results directory
        for model_dir in os.listdir(self.results_base_dir):
            # Skip if filtering by model and not matching
            if model and model != model_dir:
                continue
                
            model_path = os.path.join(self.results_base_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
                
            # Check each run directory
            for run_dir in os.listdir(model_path):
                run_path = os.path.join(model_path, run_dir)
                if not os.path.isdir(run_path):
                    continue
                    
                # Try to load metadata
                metadata_path = os.path.join(run_path, "metadata.json")
                if not os.path.exists(metadata_path):
                    continue
                    
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Extract key information
                    run_info = {
                        "run_id": metadata.get("run_id", run_dir),
                        "model": metadata.get("model_name", model_dir),
                        "question": metadata.get("question", "Unknown"),
                        "timestamp": metadata.get("timestamp", 0),
                        "timestamp_readable": metadata.get("timestamp_readable", "Unknown"),
                        "status": metadata.get("status", "Unknown"),
                        "score": metadata.get("results", {}).get("best_score", 0),
                        "iterations": metadata.get("results", {}).get("iterations_completed", 0),
                        "simulations": metadata.get("results", {}).get("simulations_completed", 0),
                        "tags": metadata.get("results", {}).get("tags", []),
                        "path": run_path
                    }
                    
                    runs.append(run_info)
                except Exception as e:
                    logger.warning(f"Failed to parse metadata from {metadata_path}: {e}")
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        
        # Limit to the requested count
        return runs[:count]
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific run.
        
        Args:
            run_id: Run ID or path to the run directory
            
        Returns:
            Dictionary with detailed run information or None if not found
        """
        # Handle the case where run_id is a path
        if os.path.isdir(run_id):
            run_path = run_id
        else:
            # Search for the run directory
            run_path = None
            for model_dir in os.listdir(self.results_base_dir):
                model_path = os.path.join(self.results_base_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue
                
                potential_path = os.path.join(model_path, run_id)
                if os.path.isdir(potential_path):
                    run_path = potential_path
                    break
                    
            if run_path is None:
                logger.warning(f"Run not found: {run_id}")
                return None
        
        # Try to load metadata
        metadata_path = os.path.join(run_path, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata not found at {metadata_path}")
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Load the best solution
            best_solution = ""
            solution_path = os.path.join(run_path, "best_solution.txt")
            if os.path.exists(solution_path):
                with open(solution_path, 'r') as f:
                    best_solution = f.read()
            
            # Load progress information
            progress = []
            progress_path = os.path.join(run_path, "progress.jsonl")
            if os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                progress.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            
            # Combine everything into a single result
            result = {
                "metadata": metadata,
                "best_solution": best_solution,
                "progress": progress,
                "run_path": run_path,
                "run_id": os.path.basename(run_path)
            }
            
            return result
        except Exception as e:
            logger.warning(f"Failed to load run details from {run_path}: {e}")
            return None
    
    def extract_key_concepts(self, solution_text: str) -> List[str]:
        """
        Extract key concepts from a solution text.
        
        Args:
            solution_text: The solution text to analyze
            
        Returns:
            List of key concepts extracted from the text
        """
        # Look for sections explicitly labeled as key concepts
        key_concepts_match = re.search(r'Key Concepts:(.+?)($|(?:\n\n))', solution_text, re.DOTALL)
        if key_concepts_match:
            # Extract and clean concepts
            concepts_text = key_concepts_match.group(1)
            concepts = [c.strip().strip('-*•') for c in concepts_text.strip().split('\n') 
                       if c.strip() and not c.strip().startswith('#')]
            return [c for c in concepts if c]
        
        # Fallback: Look for bulleted or numbered lists
        bullet_matches = re.findall(r'(?:^|\n)[ \t]*[-•*][ \t]*(.*?)(?:$|\n)', solution_text)
        if bullet_matches:
            return [m.strip() for m in bullet_matches if m.strip()]
        
        # Last resort: Split paragraphs and take short ones as potential concepts
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', solution_text) if p.strip()]
        return [p for p in paragraphs if len(p) < 100 and len(p.split()) < 15][:5]
    
    def extract_key_arguments(self, solution_text: str) -> Dict[str, List[str]]:
        """
        Extract key arguments for and against from solution text.
        
        Args:
            solution_text: The solution text to analyze
            
        Returns:
            Dictionary with 'for' and 'against' keys mapping to lists of arguments
        """
        arguments = {"for": [], "against": []}
        
        # Look for "Arguments For" section
        for_match = re.search(r'(?:Key )?Arguments For.*?:(.+?)(?:\n\n|\n(?:Arguments|Against))', 
                             solution_text, re.DOTALL | re.IGNORECASE)
        if for_match:
            for_text = for_match.group(1).strip()
            # Extract bullet points
            for_args = [a.strip().strip('-*•') for a in re.findall(r'(?:^|\n)[ \t]*[-•*\d\.][ \t]*(.*?)(?:$|\n)', for_text)]
            arguments["for"] = [a for a in for_args if a]
        
        # Look for "Arguments Against" section
        against_match = re.search(r'(?:Key )?Arguments Against.*?:(.+?)(?:\n\n|\n(?:[A-Z]))', 
                                solution_text, re.DOTALL | re.IGNORECASE)
        if against_match:
            against_text = against_match.group(1).strip()
            # Extract bullet points
            against_args = [a.strip().strip('-*•') for a in re.findall(r'(?:^|\n)[ \t]*[-•*\d\.][ \t]*(.*?)(?:$|\n)', against_text)]
            arguments["against"] = [a for a in against_args if a]
        
        return arguments
    
    def extract_conclusions(self, solution_text: str, progress: List[Dict[str, Any]]) -> List[str]:
        """
        Extract conclusions from a solution and progress syntheses.
        
        Args:
            solution_text: The best solution text
            progress: Progress information including syntheses
            
        Returns:
            List of key conclusions
        """
        conclusions = []
        
        # Extract any section labeled "Conclusion" or at the end of the text
        conclusion_match = re.search(r'(?:^|\n)Conclusion:?\s*(.*?)(?:$|\n\n)', solution_text, re.DOTALL | re.IGNORECASE)
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            conclusions.append(conclusion_text)
        else:
            # Try to extract the last paragraph as a potential conclusion
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', solution_text) if p.strip()]
            if paragraphs and not paragraphs[-1].startswith('#') and len(paragraphs[-1]) > 50:
                conclusions.append(paragraphs[-1])
        
        # Extract syntheses from progress
        for entry in progress:
            if "synthesis" in entry:
                # Take the last paragraph of each synthesis as a conclusion
                synthesis_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', entry["synthesis"]) if p.strip()]
                if synthesis_paragraphs:
                    conclusions.append(synthesis_paragraphs[-1])
        
        # Remove duplicates and very similar conclusions
        unique_conclusions = []
        for c in conclusions:
            if not any(self._text_similarity(c, uc) > 0.7 for uc in unique_conclusions):
                unique_conclusions.append(c)
        
        return unique_conclusions
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize and tokenize
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def analyze_run(self, run_id: str) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a run's results.
        
        Args:
            run_id: The run ID to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Check cache first
        if run_id in self._cache:
            return self._cache[run_id]
        
        # Get the run details
        run_details = self.get_run_details(run_id)
        if not run_details:
            return {"error": f"Run not found: {run_id}"}
        
        # Extract key information
        best_solution = run_details.get("best_solution", "")
        progress = run_details.get("progress", [])
        metadata = run_details.get("metadata", {})
        
        # Extract key insights
        key_concepts = self.extract_key_concepts(best_solution)
        key_arguments = self.extract_key_arguments(best_solution)
        conclusions = self.extract_conclusions(best_solution, progress)
        
        # Extract tags
        tags = metadata.get("results", {}).get("tags", [])
        
        # Prepare the analysis results
        analysis = {
            "run_id": run_id,
            "question": metadata.get("question", "Unknown"),
            "model": metadata.get("model_name", "Unknown"),
            "timestamp": metadata.get("timestamp_readable", "Unknown"),
            "duration": metadata.get("duration_seconds", 0),
            "status": metadata.get("status", "Unknown"),
            "best_score": metadata.get("results", {}).get("best_score", 0),
            "tags": tags,
            "key_concepts": key_concepts,
            "arguments_for": key_arguments["for"],
            "arguments_against": key_arguments["against"],
            "conclusions": conclusions,
            "path": run_details.get("run_path", "")
        }
        
        # Cache the results
        self._cache[run_id] = analysis
        
        return analysis
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple runs to identify similarities and differences.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Analyze each run
        analyses = [self.analyze_run(run_id) for run_id in run_ids]
        analyses = [a for a in analyses if "error" not in a]
        
        if not analyses:
            return {"error": "No valid runs to compare"}
        
        # Extract shared and unique concepts
        all_concepts = [set(a.get("key_concepts", [])) for a in analyses]
        shared_concepts = set.intersection(*all_concepts) if all_concepts else set()
        unique_concepts = {}
        
        for i, a in enumerate(analyses):
            run_id = a.get("run_id", f"run_{i}")
            unique = all_concepts[i] - set.union(*[c for j, c in enumerate(all_concepts) if j != i])
            if unique:
                unique_concepts[run_id] = list(unique)
        
        # Get mean score
        scores = [a.get("best_score", 0) for a in analyses]
        mean_score = sum(scores) / len(scores) if scores else 0
        
        # Find common arguments
        all_args_for = [set(a.get("arguments_for", [])) for a in analyses]
        all_args_against = [set(a.get("arguments_against", [])) for a in analyses]
        
        shared_args_for = set.intersection(*all_args_for) if all_args_for else set()
        shared_args_against = set.intersection(*all_args_against) if all_args_against else set()
        
        # Prepare comparison results
        comparison = {
            "runs_compared": run_ids,
            "models": [a.get("model", "Unknown") for a in analyses],
            "mean_score": mean_score,
            "shared_concepts": list(shared_concepts),
            "unique_concepts": unique_concepts,
            "shared_arguments_for": list(shared_args_for),
            "shared_arguments_against": list(shared_args_against),
            "best_run": max(analyses, key=lambda a: a.get("best_score", 0)).get("run_id") if analyses else None
        }
        
        return comparison
    
    def get_best_runs(self, count: int = 5, min_score: float = 7.0) -> List[Dict[str, Any]]:
        """
        Get the best MCTS runs based on score.
        
        Args:
            count: Maximum number of runs to return
            min_score: Minimum score threshold
            
        Returns:
            List of best run analyses
        """
        # List all runs
        all_runs = self.list_runs(count=100)  # Get more than we need to filter
        
        # Filter by minimum score
        qualifying_runs = [r for r in all_runs if r.get("score", 0) >= min_score]
        
        # Sort by score (highest first)
        qualifying_runs.sort(key=lambda r: r.get("score", 0), reverse=True)
        
        # Analyze the top runs
        return [self.analyze_run(r.get("run_id")) for r in qualifying_runs[:count]]
    
    def generate_report(self, run_id: str, format: str = "markdown") -> str:
        """
        Generate a comprehensive report for a run.
        
        Args:
            run_id: Run ID to generate report for
            format: Output format ('markdown', 'text', or 'html')
            
        Returns:
            Formatted report as a string
        """
        # Analyze the run
        analysis = self.analyze_run(run_id)
        
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        # Get the run details for additional information
        run_details = self.get_run_details(run_id)
        if not run_details:
            return f"Error: Run details not found for {run_id}"
        
        # Generate the report based on the format
        if format == "markdown":
            return self._generate_markdown_report(analysis, run_details)
        elif format == "text":
            return self._generate_text_report(analysis, run_details)
        elif format == "html":
            return self._generate_html_report(analysis, run_details)
        else:
            return f"Unsupported format: {format}"
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], run_details: Dict[str, Any]) -> str:
        """Generate a markdown report."""
        report = []
        
        # Header
        report.append(f"# MCTS Analysis Report: {analysis['run_id']}")
        report.append("")
        
        # Basic information
        report.append("## Basic Information")
        report.append("")
        report.append(f"- **Question:** {analysis['question']}")
        report.append(f"- **Model:** {analysis['model']}")
        report.append(f"- **Date:** {analysis['timestamp']}")
        report.append(f"- **Duration:** {analysis['duration']} seconds")
        report.append(f"- **Score:** {analysis['best_score']}")
        if analysis['tags']:
            report.append(f"- **Tags:** {', '.join(analysis['tags'])}")
        report.append("")
        
        # Key concepts
        if analysis.get('key_concepts'):
            report.append("## Key Concepts")
            report.append("")
            for concept in analysis['key_concepts']:
                report.append(f"- {concept}")
            report.append("")
        
        # Key arguments
        if analysis.get('arguments_for') or analysis.get('arguments_against'):
            report.append("## Key Arguments")
            report.append("")
            
            if analysis.get('arguments_for'):
                report.append("### Arguments For")
                report.append("")
                for arg in analysis['arguments_for']:
                    report.append(f"- {arg}")
                report.append("")
            
            if analysis.get('arguments_against'):
                report.append("### Arguments Against")
                report.append("")
                for arg in analysis['arguments_against']:
                    report.append(f"- {arg}")
                report.append("")
        
        # Conclusions
        if analysis.get('conclusions'):
            report.append("## Key Conclusions")
            report.append("")
            for conclusion in analysis['conclusions']:
                report.append(f"> {conclusion}")
                report.append("")
        
        # Best solution
        best_solution = run_details.get('best_solution', '')
        if best_solution:
            report.append("## Best Solution")
            report.append("")
            report.append("```")
            report.append(best_solution)
            report.append("```")
        
        return "\n".join(report)
    
    def _generate_text_report(self, analysis: Dict[str, Any], run_details: Dict[str, Any]) -> str:
        """Generate a plain text report."""
        report = []
        
        # Header
        report.append(f"MCTS Analysis Report: {analysis['run_id']}")
        report.append("=" * 80)
        report.append("")
        
        # Basic information
        report.append("Basic Information:")
        report.append(f"  Question: {analysis['question']}")
        report.append(f"  Model: {analysis['model']}")
        report.append(f"  Date: {analysis['timestamp']}")
        report.append(f"  Duration: {analysis['duration']} seconds")
        report.append(f"  Score: {analysis['best_score']}")
        if analysis['tags']:
            report.append(f"  Tags: {', '.join(analysis['tags'])}")
        report.append("")
        
        # Key concepts
        if analysis.get('key_concepts'):
            report.append("Key Concepts:")
            for concept in analysis['key_concepts']:
                report.append(f"  * {concept}")
            report.append("")
        
        # Key arguments
        if analysis.get('arguments_for') or analysis.get('arguments_against'):
            report.append("Key Arguments:")
            
            if analysis.get('arguments_for'):
                report.append("  Arguments For:")
                for arg in analysis['arguments_for']:
                    report.append(f"    * {arg}")
                report.append("")
            
            if analysis.get('arguments_against'):
                report.append("  Arguments Against:")
                for arg in analysis['arguments_against']:
                    report.append(f"    * {arg}")
                report.append("")
        
        # Conclusions
        if analysis.get('conclusions'):
            report.append("Key Conclusions:")
            for conclusion in analysis['conclusions']:
                report.append(f"  {conclusion}")
                report.append("")
        
        # Best solution
        best_solution = run_details.get('best_solution', '')
        if best_solution:
            report.append("Best Solution:")
            report.append("-" * 80)
            report.append(best_solution)
            report.append("-" * 80)
        
        return "\n".join(report)
    
    def _generate_html_report(self, analysis: Dict[str, Any], run_details: Dict[str, Any]) -> str:
        """Generate an HTML report."""
        # For now, we'll convert the markdown to basic HTML
        md_report = self._generate_markdown_report(analysis, run_details)
        
        # Convert headers
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', md_report, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # Convert lists
        html = re.sub(r'^- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\n\g<0></ul>', html, flags=re.DOTALL)
        
        # Convert blockquotes
        html = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
        
        # Convert code blocks
        html = re.sub(r'```\n(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        
        # Convert line breaks
        html = re.sub(r'\n\n', r'<br><br>', html)
        
        # Wrap in basic HTML structure
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCTS Analysis Report: {analysis['run_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        blockquote {{ background-color: #f9f9f9; border-left: 5px solid #ccc; padding: 10px 20px; margin: 20px 0; }}
        pre {{ background-color: #f5f5f5; padding: 15px; overflow-x: auto; }}
        ul {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
        
        return html
    
    def extract_insights(self, run_id: str, max_insights: int = 5) -> List[str]:
        """
        Extract key insights from a run's results.
        
        Args:
            run_id: The run ID to analyze
            max_insights: Maximum number of insights to extract
            
        Returns:
            List of key insights as strings
        """
        # Analyze the run
        analysis = self.analyze_run(run_id)
        
        if "error" in analysis:
            return [f"Error: {analysis['error']}"]
        
        insights = []
        
        # Add conclusions as insights
        for conclusion in analysis.get('conclusions', [])[:max_insights]:
            if conclusion and not any(self._text_similarity(conclusion, i) > 0.7 for i in insights):
                insights.append(conclusion)
        
        # Add key arguments as insights if we need more
        if len(insights) < max_insights:
            for_args = analysis.get('arguments_for', [])
            against_args = analysis.get('arguments_against', [])
            
            # Interleave arguments for and against
            all_args = []
            for i in range(max(len(for_args), len(against_args))):
                if i < len(for_args):
                    all_args.append(("For: " + for_args[i]) if for_args[i].startswith("For: ") else for_args[i])
                if i < len(against_args):
                    all_args.append(("Against: " + against_args[i]) if against_args[i].startswith("Against: ") else against_args[i])
            
            # Add arguments as insights
            for arg in all_args:
                if len(insights) >= max_insights:
                    break
                if not any(self._text_similarity(arg, i) > 0.7 for i in insights):
                    insights.append(arg)
        
        # Add key concepts as insights if we still need more
        if len(insights) < max_insights:
            for concept in analysis.get('key_concepts', []):
                if len(insights) >= max_insights:
                    break
                if not any(self._text_similarity(concept, i) > 0.7 for i in insights):
                    insights.append(concept)
        
        return insights
    
    def suggest_improvements(self, run_id: str) -> List[str]:
        """
        Suggest improvements for MCTS runs based on analysis.
        
        Args:
            run_id: The run ID to analyze
            
        Returns:
            List of improvement suggestions
        """
        # Analyze the run
        analysis = self.analyze_run(run_id)
        
        if "error" in analysis:
            return [f"Error: {analysis['error']}"]
        
        suggestions = []
        
        # Check if we've got enough iterations
        iterations = analysis.get('iterations', 0)
        if iterations < 2:
            suggestions.append(f"Increase iterations from {iterations} to at least 2 for more thorough exploration")
        
        # Check score
        score = analysis.get('best_score', 0)
        if score < 7.0:
            suggestions.append(f"Current score is {score}, which is relatively low. Try using a more sophisticated model or adjusting exploration parameters")
        
        # Check for diverse approaches
        if len(analysis.get('key_concepts', [])) < 3:
            suggestions.append("Limited key concepts identified. Consider increasing exploration weight parameter for more diverse thinking")
        
        # Check for balanced arguments
        if len(analysis.get('arguments_for', [])) > 0 and len(analysis.get('arguments_against', [])) == 0:
            suggestions.append("Arguments are one-sided (only 'for' arguments). Consider using a balanced prompt approach to get both sides")
        elif len(analysis.get('arguments_against', [])) > 0 and len(analysis.get('arguments_for', [])) == 0:
            suggestions.append("Arguments are one-sided (only 'against' arguments). Consider using a balanced prompt approach to get both sides")
        
        # Check for bayesian parameters if score is low
        if score < 8.0:
            suggestions.append("Try adjusting the prior parameters (beta_prior_alpha/beta) to improve the bandit algorithm performance")
        
        # Default suggestion
        if not suggestions:
            suggestions.append("The MCTS run looks good and achieved a reasonable score. For even better results, try increasing iterations or using a more capable model")
        
        return suggestions
