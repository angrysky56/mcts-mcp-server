#!/bin/bash
# MCTS MCP Server Setup Script for Unix-like systems (Linux/macOS)
# This script calls the cross-platform Python setup script

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting MCTS MCP Server setup..."
echo "Platform: $(uname -s) $(uname -r)"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.10+ and try again"
    echo ""
    echo "Installation instructions:"
    echo "• Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip"
    echo "• CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "• macOS: brew install python@3.11 or download from python.org"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ is required. Found $python_version"
    echo "Please upgrade Python and try again"
    exit 1
fi

echo "✅ Python $python_version found"

# Run the cross-platform setup script
echo "🔧 Running setup script..."
python3 setup.py

echo ""
echo "🎉 Setup script completed!"
echo ""
echo "Next steps:"
echo "1. Edit .env file to add your API keys"
echo "2. Add claude_desktop_config.json contents to your Claude Desktop config"
echo "3. Restart Claude Desktop"
echo ""
echo "For detailed instructions, see README.md"
