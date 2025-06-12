@echo off
REM MCTS MCP Server Setup Script for Windows
REM This script calls the cross-platform Python setup script

echo 🚀 Starting MCTS MCP Server setup...
echo Platform: Windows

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org and try again
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% found

REM Run the cross-platform setup script
echo 🔧 Running setup script...
python setup.py

if %errorlevel% neq 0 (
    echo ❌ Setup failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo 🎉 Setup script completed!
echo.
echo Next steps:
echo 1. Edit .env file to add your API keys
echo 2. Add claude_desktop_config.json contents to your Claude Desktop config
echo 3. Restart Claude Desktop
echo.
echo For detailed instructions, see README.md
echo.
pause
