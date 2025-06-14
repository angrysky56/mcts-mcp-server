#!/usr/bin/env python3
"""
MCTS MCP Server Setup Script
============================

Simple setup script using uv for the MCTS MCP Server.
"""
# ruff: noqa: T201
# Setup scripts legitimately need print statements for user feedback

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and return the result."""
    try:
        # Using shell=False and list of strings for security
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            shell=False
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"❌ Command failed: {' '.join(cmd)}\n")
        if e.stderr:
            sys.stderr.write(f"   Error: {e.stderr}\n")
        raise

def check_uv() -> bool:
    """Check if uv is installed."""
    return shutil.which("uv") is not None

def setup_project() -> None:
    """Set up the project using uv."""
    project_dir = Path(__file__).parent.resolve()

    print("🔧 Setting up MCTS MCP Server...")
    print(f"📁 Project directory: {project_dir}")

    if not check_uv():
        print("❌ uv not found. Please install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   Or visit: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

    print("✅ Found uv")

    # Sync project dependencies (creates venv and installs everything)
    print("📦 Installing dependencies...")
    run_command(["uv", "sync"], cwd=project_dir)
    print("✅ Dependencies installed")

    # Create .env file if it doesn't exist
    env_file = project_dir / ".env"
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# MCTS MCP Server Environment Configuration

# OpenAI API Key
OPENAI_API_KEY="your_openai_api_key_here"

# Anthropic API Key
ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Google Gemini API Key
GEMINI_API_KEY="your_gemini_api_key_here"

# Default LLM Provider ("ollama", "openai", "anthropic", "gemini")
DEFAULT_LLM_PROVIDER="ollama"

# Default Model Name
DEFAULT_MODEL_NAME="qwen3:latest"
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")

    # Create Claude Desktop config
    print("🔧 Generating Claude Desktop config...")
    claude_config = {
        "mcpServers": {
            "mcts-mcp-server": {
                "command": "uv",
                "args": [
                    "--directory", str(project_dir),
                    "run", "mcts-mcp-server"
                ]
            }
        }
    }

    config_file = project_dir / "claude_desktop_config.json"
    with config_file.open("w") as f:
        json.dump(claude_config, f, indent=2)
    print("✅ Claude Desktop config generated")

    # Test installation
    print("🧪 Testing installation...")
    try:
        run_command(["uv", "run", "python", "-c",
                    "import mcts_mcp_server; print('✅ Package imported successfully')"],
                   cwd=project_dir)
    except subprocess.CalledProcessError:
        print("❌ Installation test failed")
        sys.exit(1)

    print_success_message(project_dir)

def print_success_message(project_dir: Path) -> None:
    """Print setup completion message."""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)

    print("\n📋 Next Steps:")
    print(f"1. Edit {project_dir / '.env'} and add your API keys")
    print("2. Add the Claude Desktop config:")

    if platform.system() == "Windows":
        config_path = "%APPDATA%\\Claude\\claude_desktop_config.json"
    elif platform.system() == "Darwin":
        config_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
    else:
        config_path = "~/.config/claude/claude_desktop_config.json"

    print(f"   Copy contents of claude_desktop_config.json to: {config_path}")
    print("3. Restart Claude Desktop")
    print("4. Test with: uv run mcts-mcp-server")

    print("\n📚 Documentation:")
    print("• README.md - Project overview")
    print("• USAGE_GUIDE.md - Detailed usage instructions")

def main() -> None:
    """Main setup function."""
    try:
        setup_project()
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
