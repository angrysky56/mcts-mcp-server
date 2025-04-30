#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Collector for MCTS with Ollama
=====================================

This module manages the collection, saving and comparison of
MCTS runs with different Ollama models.
"""
import os
import json
import time
import datetime
import logging
from typing import Dict, Any, List, Optional
import threading

logger = logging.getLogger("results_collector")

class ResultsCollector:
    """Manages the collection and storage of MCTS run results."""
    
    def __init__(self, base_directory: str = None):
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
        """Get the directory for a specific model, creating it if necessary."""
        if model_name not in self.model_directories:
            model_dir = os.path.join(self.base_directory, model_name)
            os.makedirs(model_dir, exist_ok=True)
            self.model_directories[model_name] = model_dir
        return self.model_directories[model_name]
    
    def start_run(self, model_name: str, question: str, config: Dict[str, Any]) -> str:
        """
        Start tracking a new MCTS run.
        
        Args:
            model_name: Name of the Ollama model
            question: The question or prompt being analyzed
            config: The MCTS configuration
            
        Returns:
            The unique run ID
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
    
    def update_run_status(self, run_id: str, status: str, progress: Optional[Dict[str, Any]] = None):
        """
        Update the status of a run.
        
        Args:
            run_id: The run ID
            status: New status (e.g., 'running', 'completed', 'failed')
            progress: Optional progress information
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
    
    def save_run_results(self, run_id: str, results: Dict[str, Any]):
        """
        Save the final results of a run.
        
        Args:
            run_id: The run ID
            results: The results data to save
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
    
    def list_runs(self, model_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List runs, optionally filtered by model or status.
        
        Args:
            model_name: Optional model name to filter by
            status: Optional status to filter by
            
        Returns:
            List of run dictionaries
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
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific run.
        
        Args:
            run_id: The run ID
            
        Returns:
            The run dictionary or None if not found
        """
        with self.runs_lock:
            return self.runs.get(run_id)
    
    def run_mcts_async(self, model_name: str, question: str, config: Dict[str, Any], 
                      iterations: int, simulations_per_iter: int, callback=None):
        """
        Run MCTS in a background thread and save results.
        
        Args:
            model_name: Name of the Ollama model to use
            question: The question to analyze
            config: Configuration for MCTS
            iterations: Number of MCTS iterations
            simulations_per_iter: Simulations per iteration
            callback: Optional callback function called with results when done
        """
        # Start tracking the run
        run_id = self.start_run(model_name, question, config)
        
        # Define the worker function
        def worker():
            try:
                # Import here to avoid circular imports
                from tools import _global_state
                
                # Create a copy of the config with our specific settings
                run_config = config.copy()
                run_config.update({
                    "use_ollama": True,
                    "ollama_model": model_name,
                    "max_iterations": iterations,
                    "simulations_per_iteration": simulations_per_iter
                })
                
                # Initialize MCTS
                from tools import run_async
                
                # Update to running status
                self.update_run_status(run_id, "running")
                
                # This would have the actual MCTS implementation
                # For now we'll just simulate it
                # In a real implementation, you would call:
                # 1. initialize_mcts
                # 2. run_mcts
                # 3. generate_synthesis
                
                # Simulate MCTS running
                total_steps = iterations * simulations_per_iter
                for i in range(total_steps):
                    # Sleep to simulate work
                    time.sleep(0.1)
                    
                    # Update progress every few steps
                    if i % 5 == 0 or i == total_steps - 1:
                        progress = {
                            "step": i + 1,
                            "total_steps": total_steps,
                            "percent_complete": round((i + 1) / total_steps * 100, 1),
                            "timestamp": int(time.time())
                        }
                        self.update_run_status(run_id, "running", progress)
                
                # Generate simulated results
                results = {
                    "status": "completed",
                    "best_score": 8.5,
                    "best_solution": "This is a simulated result for the MCTS run.",
                    "tags": ["analysis", "framework", "perspective"],
                    "iterations_completed": iterations,
                    "simulations_completed": iterations * simulations_per_iter
                }
                
                # Save results
                self.save_run_results(run_id, results)
                
                # Call the callback if provided
                if callback:
                    callback(run_id, results)
                
            except Exception as e:
                logger.error(f"Error in MCTS run {run_id}: {e}")
                self.update_run_status(run_id, "failed", {"error": str(e)})
                
                # Call the callback with error if provided
                if callback:
                    callback(run_id, {"status": "failed", "error": str(e)})
        
        # Start the worker thread
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        # Track the thread
        self.active_threads.append(thread)
        
        # Clean up completed threads
        self.active_threads = [t for t in self.active_threads if t.is_alive()]
        
        return run_id
    
    def compare_models(self, question: str, models: List[str], config: Dict[str, Any],
                      iterations: int, simulations_per_iter: int):
        """
        Run MCTS with the same question across multiple models for comparison.
        
        Args:
            question: The question to analyze
            models: List of model names to compare
            config: Base configuration for MCTS
            iterations: Number of MCTS iterations
            simulations_per_iter: Simulations per iteration
            
        Returns:
            Dictionary mapping model names to run IDs
        """
        run_ids = {}
        
        for model in models:
            run_id = self.run_mcts_async(model, question, config, iterations, simulations_per_iter)
            run_ids[model] = run_id
            
            # Wait briefly between model starts to stagger them
            time.sleep(1)
        
        return run_ids

# Create a global instance
collector = ResultsCollector()
