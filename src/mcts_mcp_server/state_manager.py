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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

# DEFAULT_CONFIG is now in mcts_config
from .mcts_config import DEFAULT_CONFIG, STATE_FORMAT_VERSION

# truncate_text is now in utils
from .utils import truncate_text

if TYPE_CHECKING:
    from .mcts_core import MCTS

# Setup logger for this module
logger = logging.getLogger(__name__)

class StateManager:
    """
    Handles saving and loading MCTS state to/from a SQLite database.

    Provides persistent storage for MCTS runs, allowing continuation of analysis
    across sessions and preservation of learned approach preferences.
    """

    def __init__(self, db_file_path: str) -> None:
        """
        Initialize the StateManager with a database file path.

        Args:
            db_file_path: Path to the SQLite database file

        Note:
            Creates the database and required tables if they don't exist
        """
        self.db_file = db_file_path
        self._ensure_db_and_table()

    def _get_db_connection(self) -> sqlite3.Connection | None:
        """
        Establish connection to the SQLite database with optimized settings.

        Returns:
            SQLite connection object, or None if connection failed

        Note:
            Configures WAL mode for better concurrency and sets timeouts
        """
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
            if conn:
                conn.close()
            return None
        except Exception as e:
            logger.error(f"Unexpected error during DB connection setup {self.db_file}: {e}", exc_info=True)
            if conn:
                conn.close()
            return None

    def _ensure_db_and_table(self) -> None:
        """
        Ensure the database file and required tables exist.

        Creates:
            - mcts_state table with chat_id, state_json, and timestamp columns
            - Index on timestamp for faster queries

        Note:
            Safe to call multiple times - uses CREATE TABLE IF NOT EXISTS
        """
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot ensure DB table exists: Failed to get DB connection.")
            return
        try:
            with conn:  # Use context manager for commit/rollback
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
            if conn:
                conn.close()

    def save_state(self, chat_id: str, mcts_instance: "MCTS") -> None:
        """
        Serialize and save MCTS state to the database.

        Args:
            chat_id: Unique identifier for the conversation/session
            mcts_instance: The MCTS instance to serialize

        Note:
            Saves selective state including best solutions, approach priors,
            and unfit markers. Uses INSERT OR REPLACE for upsert behavior.
        """
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
            if conn:
                conn.close()

    def load_state(self, chat_id: str) -> dict[str, Any] | None:
        """
        Load and deserialize the latest MCTS state for a chat session.

        Args:
            chat_id: Unique identifier for the conversation/session

        Returns:
            Dictionary containing the deserialized state, or None if not found/invalid

        Note:
            Validates state format version and returns None for incompatible versions
        """
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
                    if isinstance(loaded_data, dict) and loaded_data.get("version") == STATE_FORMAT_VERSION:
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
            if conn:
                conn.close()
        return state_dict

    def _serialize_mcts_state(self, mcts_instance: "MCTS") -> str:
        """
        Extract selective state from MCTS instance and serialize to JSON.

        Args:
            mcts_instance: The MCTS instance to serialize

        Returns:
            JSON string containing serialized state, or "{}" if serialization fails

        Serialized Data:
            - version: State format version for compatibility
            - best_score: Highest score achieved
            - best_solution_summary: Truncated best solution text
            - best_solution_content: Full best solution content
            - best_node_tags: Tags from the best node
            - approach_priors: Bayesian priors for different approaches (if enabled)
            - unfit_markers: Nodes marked as unfit for future avoidance

        Note:
            Limits unfit markers to 10 entries and uses BFS to traverse tree safely
        """
        if not mcts_instance or not mcts_instance.root:
            logger.warning("Attempted to serialize empty or invalid MCTS instance.")
            return "{}"

        state = {"version": STATE_FORMAT_VERSION} # Version state format

        try:
            cfg = mcts_instance.config
            # Basic info
            state["best_score"] = str(mcts_instance.best_score)
            best_node = mcts_instance.find_best_final_node()
            state["best_solution_summary"] = truncate_text(mcts_instance.best_solution, 400)
            state["best_solution_content"] = str(mcts_instance.best_solution) # Save full best content
            state["best_node_tags"] = json.dumps(best_node.descriptive_tags) if best_node else "[]"

            # Learned Priors (if Bayesian)
            if cfg.get("use_bayesian_evaluation"):
                alphas = getattr(mcts_instance, "approach_alphas", {})
                betas = getattr(mcts_instance, "approach_betas", {})
                if isinstance(alphas, dict) and isinstance(betas, dict):
                    state["approach_priors"] = json.dumps({
                        "alpha": {k: round(v, 4) for k, v in alphas.items()},
                        "beta": {k: round(v, 4) for k, v in betas.items()}
                    })
                else:
                    state["approach_priors"] = "{}"

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
                 if not current:
                     continue
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
                          "tags": json.dumps(node_item.descriptive_tags),
                      }
                      unfit_markers.append(marker)
            state["unfit_markers"] = json.dumps(unfit_markers[:10]) # Limit number saved and convert to JSON string

            return json.dumps(state)

        except Exception as e:
            logger.error(f"Error during MCTS state serialization: {e}", exc_info=True)
            return "{}" # Return empty JSON on error

    def delete_state(self, chat_id: str) -> bool:
        """
        Delete stored state for a specific chat session.

        Args:
            chat_id: Unique identifier for the conversation/session

        Returns:
            True if state was deleted, False if not found or error occurred
        """
        if not chat_id:
            logger.warning("Cannot delete state: chat_id is missing.")
            return False

        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot delete state: Failed to get DB connection.")
            return False

        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM mcts_state WHERE chat_id = ?", (chat_id,))
                deleted_count = cursor.rowcount

            if deleted_count > 0:
                logger.info(f"Deleted MCTS state for chat_id: {chat_id}")
                return True
            else:
                logger.info(f"No state found to delete for chat_id: {chat_id}")
                return False

        except sqlite3.Error as e:
            logger.error(f"SQLite error deleting state for chat_id {chat_id}: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()

    def list_stored_sessions(self) -> list[dict[str, Any]]:
        """
        List all stored chat sessions with their metadata.

        Returns:
            List of dictionaries containing chat_id, timestamp, and state summary
        """
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot list sessions: Failed to get DB connection.")
            return []

        sessions = []
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT chat_id, last_state_json, timestamp
                    FROM mcts_state
                    ORDER BY timestamp DESC
                """)
                results = cursor.fetchall()

            for chat_id, state_json, timestamp in results:
                session_info = {
                    "chat_id": chat_id,
                    "timestamp": timestamp,
                    "timestamp_readable": datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
                }

                # Try to extract summary info from state
                try:
                    state_data = json.loads(state_json) if state_json else {}
                    session_info.update({
                        "best_score": state_data.get("best_score", "Unknown"),
                        "version": state_data.get("version", "Unknown"),
                        "has_priors": bool(state_data.get("approach_priors")),
                        "unfit_count": len(json.loads(state_data.get("unfit_markers", "[]")))
                    })
                except (json.JSONDecodeError, Exception):
                    session_info.update({
                        "best_score": "Error",
                        "version": "Error",
                        "has_priors": False,
                        "unfit_count": 0
                    })

                sessions.append(session_info)

        except sqlite3.Error as e:
            logger.error(f"SQLite error listing sessions: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

        return sessions

    def cleanup_old_states(self, days_old: int = 30) -> int:
        """
        Clean up stored states older than specified number of days.

        Args:
            days_old: Number of days after which to consider states old

        Returns:
            Number of states cleaned up
        """
        conn = self._get_db_connection()
        if not conn:
            logger.error("Cannot cleanup states: Failed to get DB connection.")
            return 0

        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)

            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM mcts_state WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount

            logger.info(f"Cleaned up {deleted_count} old states (older than {days_old} days)")
            return deleted_count

        except sqlite3.Error as e:
            logger.error(f"SQLite error during cleanup: {e}", exc_info=True)
            return 0
        finally:
            if conn:
                conn.close()
