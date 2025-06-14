#!/bin/bash
# MCTS MCP Server Setup Script
# Simple wrapper around the Python setup script

set -e

echo "🚀 MCTS MCP Server Setup"
echo "========================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Then restart your terminal and run this script again."
    exit 1
fi

echo "✅ Found uv"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "✅ Project structure verified"

# Run the Python setup script
echo "🔧 Running setup..."
uv run python setup.py

echo ""
echo "🎉 Setup complete!"
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Add claude_desktop_config.json to Claude Desktop"
echo "3. Restart Claude Desktop"
