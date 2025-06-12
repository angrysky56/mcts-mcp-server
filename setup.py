#!/usr/bin/env python3
"""
MCTS MCP Server Setup Script
============================

Cross-platform setup script for the MCTS MCP Server.
Works on Windows, macOS, and Linux.
"""

import os
import sys
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# ANSI color codes for cross-platform terminal colors
class Colors:
    if platform.system() == "Windows":
        # For Windows, we'll try to enable ANSI colors or fall back to plain text
        try:
            import colorama
            colorama.init()
            GREEN = '\033[92m'
            RED = '\033[91m'
            YELLOW = '\033[93m'
            BLUE = '\033[94m'
            RESET = '\033[0m'
            BOLD = '\033[1m'
        except ImportError:
            GREEN = RED = YELLOW = BLUE = RESET = BOLD = ''
    else:
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

def print_colored(message: str, color: str = '') -> None:
    """Print a colored message."""
    print(f"{color}{message}{Colors.RESET}")

def print_header(message: str) -> None:
    """Print a header message."""
    print_colored(f"\n{Colors.BOLD}{'='*60}", Colors.BLUE)
    print_colored(f"{message}", Colors.BLUE + Colors.BOLD)
    print_colored(f"{'='*60}\n", Colors.BLUE)

def print_success(message: str) -> None:
    """Print a success message."""
    print_colored(f"✅ {message}", Colors.GREEN)

def print_warning(message: str) -> None:
    """Print a warning message."""
    print_colored(f"⚠️  {message}", Colors.YELLOW)

def print_error(message: str) -> None:
    """Print an error message."""
    print_colored(f"❌ {message}", Colors.RED)

def print_info(message: str) -> None:
    """Print an info message."""
    print_colored(f"ℹ️  {message}", Colors.BLUE)

def run_command(command: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(command)}")
        print_error(f"Error: {e.stderr}")
        raise

def check_python_version() -> bool:
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    required_major, required_minor = 3, 10
    
    if version.major < required_major or (version.major == required_major and version.minor < required_minor):
        print_error(f"Python {required_major}.{required_minor}+ is required. Found {version.major}.{version.minor}.{version.micro}")
        return False
    
    print_success(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def find_uv() -> Optional[Path]:
    """Find the uv executable."""
    # Check common locations
    uv_paths = [
        shutil.which("uv"),  # In PATH
        Path.home() / ".cargo" / "bin" / "uv",
        Path.home() / ".local" / "bin" / "uv",
    ]
    
    # Windows executable extension
    if platform.system() == "Windows":
        uv_paths.extend([
            Path.home() / ".cargo" / "bin" / "uv.exe",
            Path.home() / ".local" / "bin" / "uv.exe",
        ])
    
    for path in uv_paths:
        if path and Path(path).exists() and Path(path).is_file():
            return Path(path)
    
    return None

def install_uv() -> Path:
    """Install uv package manager."""
    print_info("Installing uv package manager...")
    
    system = platform.system()
    
    if system in ["Linux", "Darwin"]:  # Darwin is macOS
        # Use the official install script
        install_cmd = [
            "curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"
        ]
        try:
            # Use shell=True for pipe command
            subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print_error("Failed to install uv using curl. Trying alternative method...")
            # Fallback to pip install
            run_command([sys.executable, "-m", "pip", "install", "uv"])
    
    elif system == "Windows":
        try:
            # Try PowerShell install script
            ps_cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
            subprocess.run(ps_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print_warning("PowerShell install failed. Trying pip install...")
            run_command([sys.executable, "-m", "pip", "install", "uv"])
    
    else:
        print_warning(f"Unknown system {system}. Trying pip install...")
        run_command([sys.executable, "-m", "pip", "install", "uv"])
    
    # Find uv after installation
    uv_path = find_uv()
    if not uv_path:
        # Try to find it in common locations after install
        common_paths = [
            Path.home() / ".cargo" / "bin" / "uv",
            Path.home() / ".local" / "bin" / "uv",
        ]
        if platform.system() == "Windows":
            common_paths.extend([
                Path.home() / ".cargo" / "bin" / "uv.exe",
                Path.home() / ".local" / "bin" / "uv.exe",
            ])
        
        for path in common_paths:
            if path.exists():
                uv_path = path
                break
    
    if not uv_path:
        raise RuntimeError("Failed to find uv after installation")
    
    print_success(f"uv installed successfully at: {uv_path}")
    return uv_path

def setup_virtual_environment(uv_path: Path, project_dir: Path) -> Path:
    """Set up the virtual environment."""
    venv_dir = project_dir / ".venv"
    
    if venv_dir.exists():
        print_info("Virtual environment already exists")
    else:
        print_info("Creating virtual environment...")
        run_command([str(uv_path), "venv", ".venv"], cwd=project_dir)
        print_success("Virtual environment created")
    
    return venv_dir

def install_dependencies(uv_path: Path, project_dir: Path) -> None:
    """Install project dependencies."""
    print_info("Installing project dependencies...")
    
    # Install main dependencies
    run_command([str(uv_path), "pip", "install", "."], cwd=project_dir)
    print_success("Main dependencies installed")
    
    # Install development dependencies
    try:
        run_command([str(uv_path), "pip", "install", ".[dev]"], cwd=project_dir)
        print_success("Development dependencies installed")
    except subprocess.CalledProcessError:
        print_warning("Development dependencies failed to install (this is optional)")
    
    # Ensure google-genai is installed (in case it's not in pyproject.toml)
    try:
        run_command([str(uv_path), "pip", "install", "google-genai>=1.20.0"], cwd=project_dir)
        print_success("Google Gemini package installed")
    except subprocess.CalledProcessError:
        print_warning("Failed to install google-genai package")

def setup_environment_file(project_dir: Path) -> None:
    """Set up the .env file from .env.example."""
    env_file = project_dir / ".env"
    env_example = project_dir / ".env.example"
    
    if env_file.exists():
        print_info(".env file already exists")
        return
    
    if not env_example.exists():
        print_warning(".env.example not found, creating basic .env file")
        # Create a basic .env file
        env_content = '''# MCTS MCP Server Environment Configuration

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
'''
        env_file.write_text(env_content)
    else:
        # Copy from example
        shutil.copy2(env_example, env_file)
    
    print_success(f".env file created at {env_file}")
    print_info("Please edit the .env file to add your API keys")

def create_state_directory() -> None:
    """Create the state directory."""
    state_dir = Path.home() / ".mcts_mcp_server"
    state_dir.mkdir(exist_ok=True)
    print_success(f"State directory created at {state_dir}")

def verify_installation(uv_path: Path, project_dir: Path) -> bool:
    """Verify the installation is working."""
    print_info("Verifying installation...")
    
    try:
        # Test basic import
        result = run_command([
            str(uv_path), "run", "python", "-c", 
            "import mcts_mcp_server; print('✅ MCTS MCP Server package imported successfully')"
        ], cwd=project_dir)
        
        # Test Gemini package
        result = run_command([
            str(uv_path), "run", "python", "-c",
            "import google.genai; print('✅ Google Gemini package available')"
        ], cwd=project_dir)
        
        print_success("Installation verification passed")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error("Installation verification failed")
        print_error(f"Error: {e}")
        return False

def generate_claude_config(project_dir: Path) -> None:
    """Generate Claude Desktop configuration."""
    claude_config = {
        "mcpServers": {
            "mcts-mcp-server": {
                "command": "uv",
                "args": [
                    "--directory",
                    str(project_dir),
                    "run",
                    "mcts-mcp-server"
                ],
                "env": {
                    "UV_PROJECT_ENVIRONMENT": str(project_dir / ".venv")
                }
            }
        }
    }
    
    config_file = project_dir / "claude_desktop_config.json"
    with open(config_file, 'w') as f:
        json.dump(claude_config, f, indent=2)
    
    print_success(f"Claude Desktop config generated at {config_file}")

def print_setup_complete(project_dir: Path) -> None:
    """Print setup completion message with next steps."""
    print_header("Setup Complete! 🎉")
    
    print_colored("Next Steps:", Colors.BOLD)
    print()
    
    print("1. 📝 Configure API Keys:")
    print(f"   Edit {project_dir / '.env'} and add your API keys")
    print()
    
    print("2. 🔧 Configure Claude Desktop:")
    print("   Add the contents of claude_desktop_config.json to your Claude Desktop config:")
    
    system = platform.system()
    if system == "Windows":
        config_path = "%APPDATA%\\Claude\\claude_desktop_config.json"
    elif system == "Darwin":  # macOS
        config_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
    else:  # Linux
        config_path = "~/.config/claude/claude_desktop_config.json"
    
    print(f"   Location: {config_path}")
    print()
    
    print("3. 🔄 Restart Claude Desktop")
    print()
    
    print("4. 🧪 Test the installation:")
    print(f"   cd {project_dir}")
    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("   python -c \"import mcts_mcp_server; print('Success!')\"")
    print()
    
    print_colored("Available LLM Providers:", Colors.BOLD)
    print("• Ollama (local models) - No API key needed")
    print("• OpenAI (GPT models) - Requires OPENAI_API_KEY")
    print("• Anthropic (Claude models) - Requires ANTHROPIC_API_KEY") 
    print("• Google Gemini - Requires GEMINI_API_KEY")
    print()
    
    print_colored("For detailed usage instructions, see:", Colors.BOLD)
    print("• README.md")
    print("• USAGE_GUIDE.md")

def main():
    """Main setup function."""
    print_header("MCTS MCP Server Setup")
    print_info(f"Platform: {platform.system()} {platform.release()}")
    print_info(f"Python: {sys.version}")
    
    # Get project directory
    project_dir = Path(__file__).parent.resolve()
    print_info(f"Project directory: {project_dir}")
    
    try:
        # Check Python version
        if not check_python_version():
            sys.exit(1)
        
        # Check/install uv
        uv_path = find_uv()
        if not uv_path:
            print_warning("uv package manager not found")
            uv_path = install_uv()
        else:
            print_success(f"Found uv at: {uv_path}")
        
        # Setup virtual environment
        setup_virtual_environment(uv_path, project_dir)
        
        # Install dependencies
        install_dependencies(uv_path, project_dir)
        
        # Setup environment file
        setup_environment_file(project_dir)
        
        # Create state directory
        create_state_directory()
        
        # Generate Claude config
        generate_claude_config(project_dir)
        
        # Verify installation
        if not verify_installation(uv_path, project_dir):
            print_error("Setup completed with errors. Please check the installation.")
            sys.exit(1)
        
        # Print completion message
        print_setup_complete(project_dir)
        
    except Exception as e:
        print_error(f"Setup failed: {e}")
        print_info("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
