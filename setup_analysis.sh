#!/bin/bash
# Script to set up and deploy the improved MCTS system with analysis tools

# Set up error handling
set -e
echo "Setting up improved MCTS system with analysis tools..."

# Create analysis_tools directory if it doesn't exist
mkdir -p ./src/mcts_mcp_server/analysis_tools

# Check if we're in the correct directory
if [ ! -f "src/mcts_mcp_server/tools.py" ]; then
    echo "Error: Please run this script from the mcts-mcp-server root directory"
    exit 1
fi

# Install required dependencies
echo "Installing required dependencies..."
pip install rich pathlib

# Backup original tools.py file
echo "Backing up original tools.py file..."
cp "src/mcts_mcp_server/tools.py" "src/mcts_mcp_server/tools.py.bak.$(date +%Y%m%d%H%M%S)"
# Update tools.py with new version
echo "Updating tools.py with new version..."
if [ -f "src/mcts_mcp_server/tools.py.update" ]; then
    cp src/mcts_mcp_server/tools.py.update src/mcts_mcp_server/tools.py
    echo "tools.py updated successfully."
else
    echo "Error: tools.py.update not found. Please run the setup script first."
    exit 1
fi

# Create __init__.py in analysis_tools directory
echo "Creating analysis_tools/__init__.py..."
cat > src/mcts_mcp_server/analysis_tools/__init__.py << 'EOF'
"""
MCTS Analysis Tools
=================

This module provides tools for analyzing and visualizing MCTS results.
"""

from .results_processor import ResultsProcessor
from .mcts_tools import register_mcts_analysis_tools
EOF

# Check if results_processor.py exists
if [ ! -f "src/mcts_mcp_server/analysis_tools/results_processor.py" ]; then
    echo "Error: results_processor.py not found. Please run the setup script first."
    exit 1
fi

# Check if mcts_tools.py exists
if [ ! -f "src/mcts_mcp_server/analysis_tools/mcts_tools.py" ]; then
    echo "Error: mcts_tools.py not found. Please run the setup script first."
    exit 1
fi

echo "Setup complete!"
echo "To use the new analysis tools, restart the MCP server."
echo ""
echo "Available new tools:"
echo "- list_mcts_runs: List recent MCTS runs"
echo "- get_mcts_run_details: Get details about a specific run"
echo "- get_mcts_solution: Get the best solution from a run"
echo "- analyze_mcts_run: Analyze a run to extract key insights"
echo "- get_mcts_insights: Extract key insights from a run"
echo "- get_mcts_report: Generate a comprehensive report"
echo "- get_best_mcts_runs: Get the best runs based on score"
echo "- suggest_mcts_improvements: Get suggestions for improvement"
echo "- compare_mcts_runs: Compare multiple runs"
echo ""
echo "Example usage:"
echo "1. list_mcts_runs()  # List all runs"
echo "2. get_mcts_insights(run_id='cogito:latest_1745979984')  # Get key insights"
echo "3. get_mcts_report(run_id='cogito:latest_1745979984', format='markdown')  # Generate a report"
