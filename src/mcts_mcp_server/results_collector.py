#!/usr/bin/env python3
"""
Results Collector for MCTS with Ollama
=====================================

This module manages the collection, saving and comparison of
MCTS runs with different Ollama models.
"""
import asyncio
import concurrent.futures
import datetime
import json
import logging
import os
import threading
import time
from collections.abc import Coroutine
from typing import Any


def run_async_safe(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine safely in a synchronous context.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Note:
        Handles event loop detection and creates new threads if necessary
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, run in a new thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coro)

logger = logging.getLogger("results_collector")

class ResultsCollector:
    """
    Manages the collection and storage of MCTS run results.

    Provides functionality to track multiple MCTS runs, save results to disk,
    and compare performance across different models.
    """

    def __init__(self, base_directory: str | None = None) -> None:
        """
        Initialize the results collector.

        Args:
            base_directory: Base directory to store results. If None, defaults to
                            'results' in the current working directory.
        """
        if base_directory is None:
            # Default to 'results' in the repository root
            repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.base_directory = os.path.join(repo_dir, "results")
        else:
            self.base_directory = base_directory

        # Create the base directory if it doesn't exist
        os.makedirs(self.base_directory, exist_ok=True)

        # Create model-specific directories
        self.model_directories = {}

        # Track runs and their status
        self.runs = {}
        self.runs_lock = threading.Lock()

        # Active threads for background processes
        self.active_threads = []

        logger.info(f"Initialized ResultsCollector with base directory: {self.base_directory}")

    def _get_model_directory(self, model_name: str) -> str:
        """
        Get the directory for a specific model, creating it if necessary.

        Args:
            model_name: Name of the model to get directory for

        Returns:
            Path to the model's directory

        Note:
            Creates the directory structure if it doesn't exist
        """
        if model_name not in self.model_directories:
            model_dir = os.path.join(self.base_directory, model_name)
            os.makedirs(model_dir, exist_ok=True)
            self.model_directories[model_name] = model_dir
        return self.model_directories[model_name]

    def start_run(self, model_name: str, question: str, config: dict[str, Any]) -> str:
        """
        Start tracking a new MCTS run.

        Args:
            model_name: Name of the Ollama model
            question: The question or prompt being analyzed
            config: The MCTS configuration

        Returns:
            The unique run ID for tracking this run

        Note:
            Creates run directory and saves initial metadata
        """
        # Generate a unique run ID
        timestamp = int(time.time())
        run_id = f"{model_name}_{timestamp}"

        # Create run directory
        model_dir = self._get_model_directory(model_name)
        run_dir = os.path.join(model_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Store run metadata
        run_info = {
            "run_id": run_id,
            "model_name": model_name,
            "question": question,
            "config": config,
            "timestamp": timestamp,
            "timestamp_readable": datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "status": "started",
            "completion_time": None,
            "duration_seconds": None,
            "results": None
        }

        # Save the metadata
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(run_info, f, indent=2)

        # Track this run in memory
        with self.runs_lock:
            self.runs[run_id] = run_info

        logger.info(f"Started MCTS run: {run_id} with model {model_name}")
        return run_id

    def update_run_status(self, run_id: str, status: str, progress: dict[str, Any] | None = None) -> None:
        """
        Update the status of a run.

        Args:
            run_id: The run ID to update
            status: New status (e.g., 'running', 'completed', 'failed')
            progress: Optional progress information to append

        Note:
            Updates both in-memory tracking and saves to disk
            Automatically calculates duration for completed/failed runs
        """
        with self.runs_lock:
            if run_id not in self.runs:
                logger.warning(f"Attempted to update unknown run: {run_id}")
                return

            # Update the run information
            self.runs[run_id]["status"] = status

            if progress:
                if "progress" not in self.runs[run_id]:
                    self.runs[run_id]["progress"] = []
                self.runs[run_id]["progress"].append(progress)

            # If complete, update completion time and duration
            if status == "completed" or status == "failed":
                now = int(time.time())
                self.runs[run_id]["completion_time"] = now
                self.runs[run_id]["completion_time_readable"] = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
                self.runs[run_id]["duration_seconds"] = now - self.runs[run_id]["timestamp"]

            # Save updated metadata
            model_name = self.runs[run_id]["model_name"]
            model_dir = self._get_model_directory(model_name)
            run_dir = os.path.join(model_dir, run_id)
            metadata_path = os.path.join(run_dir, "metadata.json")

            with open(metadata_path, 'w') as f:
                json.dump(self.runs[run_id], f, indent=2)

            # If there's progress, also save to a separate file for easier tracking
            if progress:
                progress_path = os.path.join(run_dir, "progress.jsonl")
                with open(progress_path, 'a') as f:
                    f.write(json.dumps(progress) + "\n")

        logger.info(f"Updated run {run_id} status to {status}")

    def save_run_results(self, run_id: str, results: dict[str, Any]) -> None:
        """
        Save the final results of a run.

        Args:
            run_id: The run ID to save results for
            results: The results data to save

        Note:
            Saves results in multiple formats:
            - metadata.json: Full run information
            - results.json: Just the results data
            - best_solution.txt: Best solution text (if available)
        """
        with self.runs_lock:
            if run_id not in self.runs:
                logger.warning(f"Attempted to save results for unknown run: {run_id}")
                return

            # Update run information with results
            self.runs[run_id]["results"] = results
            self.runs[run_id]["status"] = "completed"

            now = int(time.time())
            if not self.runs[run_id].get("completion_time"):
                self.runs[run_id]["completion_time"] = now
                self.runs[run_id]["completion_time_readable"] = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
                self.runs[run_id]["duration_seconds"] = now - self.runs[run_id]["timestamp"]

            # Save updated metadata
            model_name = self.runs[run_id]["model_name"]
            model_dir = self._get_model_directory(model_name)
            run_dir = os.path.join(model_dir, run_id)

            # Save metadata
            metadata_path = os.path.join(run_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.runs[run_id], f, indent=2)

            # Save results separately
            results_path = os.path.join(run_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            # If there's a best solution, save it separately
            if "best_solution" in results:
                solution_path = os.path.join(run_dir, "best_solution.txt")
                with open(solution_path, 'w') as f:
                    f.write(results["best_solution"])

        logger.info(f"Saved results for run {run_id}")

    def list_runs(self, model_name: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        """
        List runs, optionally filtered by model or status.

        Args:
            model_name: Optional model name to filter by
            status: Optional status to filter by ('started', 'running', 'completed', 'failed')

        Returns:
            List of run dictionaries sorted by timestamp (most recent first)

        Note:
            Returns copies of run data to prevent external modification
        """
        with self.runs_lock:
            # Start with all runs
            result = list(self.runs.values())

            # Apply filters
            if model_name:
                result = [r for r in result if r["model_name"] == model_name]
            if status:
                result = [r for r in result if r["status"] == status]

            # Sort by timestamp (most recent first)
            result.sort(key=lambda r: r["timestamp"], reverse=True)

            return result

    def get_run_details(self, run_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific run.

        Args:
            run_id: The run ID to get details for

        Returns:
            Detailed information about the run, or None if run not found

        Note:
            Returns a copy of the run data to prevent external modification
        """
        with self.runs_lock:
            return self.runs.get(run_id)

    def compare_models(self, question: str, models: list[str], config: dict[str, Any],
                      iterations: int, simulations_per_iter: int) -> dict[str, str]:
        """
        Run MCTS with the same question across multiple models for comparison.

        Args:
            question: The question to analyze
            models: List of model names to compare
            config: Base configuration for MCTS
            iterations: Number of MCTS iterations
            simulations_per_iter: Simulations per iteration

        Returns:
            Dictionary mapping model names to their corresponding run IDs

        Note:
            Only starts tracking runs - actual MCTS execution should be handled by caller
            Staggers run starts by 1 second to avoid resource conflicts
        """
        run_ids = {}

        for model in models:
            # Start tracking the run but don't execute MCTS here
            # The actual MCTS execution should be handled by the caller
            run_id = self.start_run(model, question, config)
            run_ids[model] = run_id

            # Wait briefly between model starts to stagger them
            time.sleep(1)

        return run_ids

    def get_model_comparison(self, question: str) -> dict[str, list[dict[str, Any]]]:
        """
        Get comparison data for all models that have run the same question.

        Args:
            question: The question to find comparisons for

        Returns:
            Dictionary mapping model names to lists of their runs for that question

        Note:
            Useful for analyzing performance differences across models
        """
        with self.runs_lock:
            comparison = {}
            for run in self.runs.values():
                if run["question"] == question:
                    model = run["model_name"]
                    if model not in comparison:
                        comparison[model] = []
                    comparison[model].append(run)

            # Sort runs within each model by timestamp
            for model_runs in comparison.values():
                model_runs.sort(key=lambda r: r["timestamp"], reverse=True)

            return comparison

    def get_summary_stats(self) -> dict[str, Any]:
        """
        Get summary statistics across all runs.

        Returns:
            Dictionary containing summary statistics including:
            - total_runs: Total number of runs
            - models_used: List of unique models used
            - status_counts: Count of runs by status
            - average_duration: Average run duration in seconds
            - success_rate: Percentage of successful completions
        """
        with self.runs_lock:
            total_runs = len(self.runs)
            if total_runs == 0:
                return {
                    "total_runs": 0,
                    "models_used": [],
                    "status_counts": {},
                    "average_duration": 0,
                    "success_rate": 0
                }

            models_used = list({run["model_name"] for run in self.runs.values()})
            status_counts = {}
            durations = []
            completed_runs = 0

            for run in self.runs.values():
                status = run["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

                if run.get("duration_seconds"):
                    durations.append(run["duration_seconds"])

                if status == "completed":
                    completed_runs += 1

            avg_duration = sum(durations) / len(durations) if durations else 0
            success_rate = (completed_runs / total_runs) * 100 if total_runs > 0 else 0

            return {
                "total_runs": total_runs,
                "models_used": models_used,
                "status_counts": status_counts,
                "average_duration": avg_duration,
                "success_rate": success_rate
            }

    def cleanup_old_runs(self, days_old: int = 30) -> int:
        """
        Clean up runs older than specified number of days.

        Args:
            days_old: Number of days after which to consider runs old

        Returns:
            Number of runs cleaned up

        Note:
            Removes both in-memory tracking and disk files
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0

        with self.runs_lock:
            runs_to_remove = []

            for run_id, run_data in self.runs.items():
                if run_data["timestamp"] < cutoff_time:
                    runs_to_remove.append(run_id)

                    # Remove disk files
                    model_dir = self._get_model_directory(run_data["model_name"])
                    run_dir = os.path.join(model_dir, run_id)

                    try:
                        import shutil
                        if os.path.exists(run_dir):
                            shutil.rmtree(run_dir)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove run directory {run_dir}: {e}")

            # Remove from memory
            for run_id in runs_to_remove:
                del self.runs[run_id]

        logger.info(f"Cleaned up {cleaned_count} old runs (older than {days_old} days)")
        return cleaned_count

# Create a global instance
collector = ResultsCollector()
