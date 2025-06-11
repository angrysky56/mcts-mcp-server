#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS MCP Server Entry Point
===========================

This script initializes and runs the MCP server that
exposes the MCTS (Monte Carlo Tree Search) functionality.
It includes graceful shutdown and process management.
"""
import logging
import os
import sys
import signal
import time
import psutil
# from typing import Optional # Optional was not used

# Add debug print for Python environment
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Add the project directory to the Python path for more reliable imports
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
    print(f"Added {project_dir} to Python path")

# Also add the src directory
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added {src_dir} to Python path")

from mcp.server.fastmcp import FastMCP
# Use proper package import
from .tools import register_mcts_tools # Changed to relative import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcts_mcp_server")

def cleanup_on_exit(signum, frame):
    """Clean up resources when the server is terminated."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # Python will automatically close file handles and clean up resources
    # This is primarily to log that we're shutting down
    sys.exit(0)

def main():
    """Initialize and run the MCP server"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_on_exit)
    signal.signal(signal.SIGTERM, cleanup_on_exit)

    # Create a PID file to track this instance
    pid_file = os.path.expanduser("~/.mcts_mcp_server/mcts_server.pid")
    pid_dir = os.path.dirname(pid_file)
    os.makedirs(pid_dir, exist_ok=True)

    # Check if server is already running
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            # Check if process with this PID exists
            if psutil.pid_exists(old_pid):
                process = psutil.Process(old_pid)
                # Check if it's actually our server process
                if "server.py" in " ".join(process.cmdline()):
                    logger.warning(f"MCTS server already running with PID {old_pid}")
                    logger.warning("Terminating old instance before starting a new one")
                    os.kill(old_pid, signal.SIGTERM)
                    # Wait a bit for the process to terminate
                    time.sleep(2)
        except (ValueError, ProcessLookupError, psutil.NoSuchProcess) as e:
            logger.warning(f"Error checking existing process: {e}")

    # Write our PID to the file
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    try:
        # Initialize the MCP server
        logger.info("Initializing MCTS MCP Server...")
        mcp = FastMCP("MCTSServer")

        # Create state persistence directory
        db_path = os.path.expanduser("~/.mcts_mcp_server/state.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Register MCTS tools with the server
        register_mcts_tools(mcp, db_path)

        # Start the server
        logger.info("Starting MCP server (STDIO transport)...")
        mcp.run()
    finally:
        # Clean up PID file when we exit
        try:
            if os.path.exists(pid_file):
                os.unlink(pid_file)
        except Exception as e:
            logger.error(f"Error removing PID file: {e}")

if __name__ == "__main__":
    main()
