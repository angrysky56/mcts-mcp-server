#!/bin/bash
# Setup script for MCTS MCP Server using UV (Astral UV)

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Setting up MCTS MCP Server with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV not installed. Installing UV..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
    # Reload shell to use uv
    source ~/.bashrc
fi

# Create and activate virtual environment using UV
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment with UV..."
    uv venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment (for this script)
source .venv/bin/activate

# Install dependencies with UV
echo "Installing dependencies with UV..."
uv pip install -r requirements.txt

# Create state directory
echo "Creating state directory..."
mkdir -p ~/.mcts_mcp_server

echo "Setup complete!"
echo ""
echo "To use with Claude Desktop:"
echo "1. Copy the content from claude_desktop_config.json"
echo "2. Add it to your Claude Desktop configuration (~/.claude/claude_desktop_config.json)"
echo "3. Make sure to update paths if needed"
echo ""
echo "You may need to restart Claude Desktop after updating the configuration."
