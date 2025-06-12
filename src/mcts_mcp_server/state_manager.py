#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Manager for MCTS
======================

This module defines the StateManager class for saving and loading MCTS state.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING

# DEFAULT_CONFIG is now in mcts_config
from .mcts_config import DEFAULT_CONFIG
# truncate_text is now in utils
from .utils import truncate_text

if TYPE_CHECKING:
    from .mcts_core import MCTS

# Setup logger for this module
logger = logging.getLogger(__name__)

class StateManager:
    """Handles saving and loading MCTS state to/from a SQLite database."""

    def __init__(self, db_file_path: str):
        self.db_file = db_file_path
        self._ensure_db_and_table()

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """Establishes connection to the SQLite DB."""
        conn = None
        try:
            db_dir = os.path.dirname(self.db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created directory for database: {db_dir}")
            conn = sqlite3.connect(self.db_file, timeout=10) # Add timeout
            conn.execute("PRAGMA journal_mode=WAL;")  # Improve concurrency
            conn.execute("PRAGMA busy_timeout = 5000;") # Wait 5s if locked
            return conn
        except sqlite3.Error as e:
            logger.error(f"SQLite error connecting to {self.db_file}: {e}", exc_info=True)
            if conn: conn.close()
            return None
        except Exception as e:
            logger.error(f"Unexpected error during DB connection setup {self.db_file}: {e}", exc_info=True)
            if conn: conn.close()
            return None

    def _ensure_db_and_table(self):
        """Ensures the database file and the state table exist."""
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot ensure DB table exists: Failed to get DB connection.")
            return
        try:
            with conn: # Use context manager for commit/rollback
                 conn.execute("""
                    CREATE TABLE IF NOT EXISTS mcts_state (
                        chat_id TEXT PRIMARY KEY,
                        last_state_json TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                 """)
                 # Optional: Index for faster lookups
                 conn.execute("CREATE INDEX IF NOT EXISTS idx_mcts_state_timestamp ON mcts_state (timestamp);")
            logger.debug(f"Database table 'mcts_state' ensured in {self.db_file}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating table in {self.db_file}: {e}", exc_info=True)
        finally:
            if conn: conn.close()

    def save_state(self, chat_id: str, mcts_instance: "MCTS"):
        """Serializes MCTS state and saves it to the database."""
        if not chat_id:
            logger.warning("Cannot save state: chat_id is missing.")
            return
        if not mcts_instance:
            logger.warning("Cannot save state: MCTS instance is invalid.")
            return

        state_json = self._serialize_mcts_state(mcts_instance)
        if state_json == "{}":
             logger.warning("Serialization produced empty state, not saving.")
             return

        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot save state: Failed to get DB connection.")
            return
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO mcts_state (chat_id, last_state_json, timestamp) VALUES (?, ?, ?)",
                    (chat_id, state_json, datetime.now())
                )
            logger.info(f"Saved MCTS state for chat_id: {chat_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving state for chat_id {chat_id}: {e}", exc_info=True)
        finally:
            if conn: conn.close()

    def load_state(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Loads and deserializes the latest MCTS state for a chat_id."""
        if not chat_id:
            logger.warning("Cannot load state: chat_id is missing.")
            return None

        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot load state: Failed to get DB connection.")
            return None

        state_dict = None
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT last_state_json FROM mcts_state WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (chat_id,)
                )
                result = cursor.fetchone()
            if result and result[0]:
                try:
                    loaded_data = json.loads(result[0])
                    # Basic validation (can be extended)
                    if isinstance(loaded_data, dict) and loaded_data.get("version") == "0.8.0":
                         state_dict = loaded_data
                         logger.info(f"Loaded and validated MCTS state for chat_id: {chat_id}")
                    else:
                         logger.warning(f"Loaded state for {chat_id} is invalid or wrong version ({loaded_data.get('version')}). Discarding.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding loaded state JSON for chat_id {chat_id}: {e}")
            else:
                 logger.info(f"No previous MCTS state found for chat_id: {chat_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error loading state for chat_id {chat_id}: {e}", exc_info=True)
        finally:
            if conn: conn.close()
        return state_dict

    def _serialize_mcts_state(self, mcts_instance: "MCTS") -> str:
        """Extracts selective state from MCTS and returns as JSON string."""
        if not mcts_instance or not mcts_instance.root:
            logger.warning("Attempted to serialize empty or invalid MCTS instance.")
            return "{}"

        state = {"version": "0.8.0"} # Version state format

        try:
            cfg = mcts_instance.config
            # Basic info
            state["best_score"] = mcts_instance.best_score
            best_node = mcts_instance.find_best_final_node()
            state["best_solution_summary"] = truncate_text(mcts_instance.best_solution, 400)
            state["best_solution_content"] = str(mcts_instance.best_solution) # Save full best content
            state["best_node_tags"] = best_node.descriptive_tags if best_node else []

            # Learned Priors (if Bayesian)
            if cfg.get("use_bayesian_evaluation"):
                alphas = getattr(mcts_instance, "approach_alphas", {})
                betas = getattr(mcts_instance, "approach_betas", {})
                if isinstance(alphas, dict) and isinstance(betas, dict):
                    state["approach_priors"] = {
                        "alpha": {k: round(v, 4) for k, v in alphas.items()},
                        "beta": {k: round(v, 4) for k, v in betas.items()}
                    }
                else: state["approach_priors"] = None

            # Unfit Markers (simple approach based on score/visits)
            unfit_markers = []
            score_thresh = cfg.get("unfit_score_threshold", DEFAULT_CONFIG["unfit_score_threshold"])
            visit_thresh = cfg.get("unfit_visit_threshold", DEFAULT_CONFIG["unfit_visit_threshold"])

            # Collect all nodes first (BFS)
            all_nodes = []
            queue = [mcts_instance.root] if mcts_instance.root else []
            # Corrected: Initialize visited_ids properly
            visited_ids = {mcts_instance.root.id} if mcts_instance.root and mcts_instance.root.id else set()
            while queue:
                 current = queue.pop(0)
                 if not current: continue
                 all_nodes.append(current) # Add node itself
                 for child in current.children:
                      if child and child.id not in visited_ids:
                           visited_ids.add(child.id)
                           queue.append(child)

            # Now check collected nodes for unfitness
            for node_item in all_nodes: # Renamed to avoid conflict with 'node' module
                 if node_item.visits >= visit_thresh and node_item.get_average_score() < score_thresh:
                      marker = {
                          "id": node_item.id,
                          "sequence": node_item.sequence,
                          "summary": truncate_text(node_item.thought or node_item.content, 80),
                          "reason": f"Low score ({node_item.get_average_score():.1f} < {score_thresh}) after {node_item.visits} visits",
                          "tags": node_item.descriptive_tags,
                      }
                      unfit_markers.append(marker)
            state["unfit_markers"] = unfit_markers[:10] # Limit number saved

            return json.dumps(state)

        except Exception as e:
            logger.error(f"Error during MCTS state serialization: {e}", exc_info=True)
            return "{}" # Return empty JSON on error
