#!/bin/bash
# MCTS MCP Server Setup Script
# Cross-platform setup script that calls the Python setup script

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 MCTS MCP Server Setup"
echo "========================================"
echo "Platform: $(uname -s) $(uname -r)"
echo "Date: $(date)"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is available
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    # Check if it's Python 3
    python_version=$(python -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
    if [ "$python_version" = "3" ]; then
        PYTHON_CMD="python"
    else
        echo "❌ Python 3 is required but found Python 2"
        echo "Please install Python 3.10+ and try again"
        exit 1
    fi
else
    echo "❌ Python 3 is not installed or not in PATH"
    echo ""
    echo "Installation instructions:"
    echo "• Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "• CentOS/RHEL/Fedora: sudo dnf install python3 python3-pip"
    echo "• macOS: brew install python@3.11 or download from python.org"
    echo "• Arch Linux: sudo pacman -S python python-pip"
    exit 1
fi

# Check Python version
python_version=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ is required. Found $python_version"
    echo "Please upgrade Python and try again"
    exit 1
fi

echo "✅ Python $python_version found"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found. Please run this script from the mcts-mcp-server directory"
    exit 1
fi

if [ ! -f "setup.py" ]; then
    echo "❌ setup.py not found. The setup script is missing"
    exit 1
fi

# Run the cross-platform setup script
echo ""
echo "🔧 Running cross-platform setup script..."
echo "========================================"
$PYTHON_CMD setup.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Quick start:"
    echo "1. Edit .env file: nano .env"
    echo "2. Add your API keys to the .env file"
    echo "3. Copy claude_desktop_config.json contents to Claude Desktop"
    echo "4. Restart Claude Desktop"
    echo ""
    echo "For detailed instructions, see README.md"
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    echo "If you need help, please check the README.md or create an issue."
    exit 1
fi
