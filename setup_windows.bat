@echo off
REM MCTS MCP Server Setup Script for Windows
REM Simple wrapper around the Python setup script

echo 🚀 MCTS MCP Server Setup
echo ========================

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ uv not found. Please install uv first:
    echo    pip install uv
    echo    Or visit: https://docs.astral.sh/uv/getting-started/installation/
    echo    Then run this script again.
    echo.
    pause
    exit /b 1
)

echo ✅ Found uv

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo ❌ pyproject.toml not found
    echo Please run this script from the project root directory
    echo.
    pause
    exit /b 1
)

echo ✅ Project structure verified

REM Run the Python setup script
echo 🔧 Running setup...
uv run python setup.py

if %errorlevel% neq 0 (
    echo ❌ Setup failed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup complete!
echo Next steps:
echo 1. Edit .env and add your API keys
echo 2. Add claude_desktop_config.json to Claude Desktop
echo 3. Restart Claude Desktop
echo.
pause
