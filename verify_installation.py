#!/usr/bin/env python3
"""
MCTS MCP Server Installation Verification Script
===============================================

This script verifies that the MCTS MCP Server is properly installed
and configured across different platforms.
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Simple color output
class Colors:
    GREEN = '\033[92m' if platform.system() != "Windows" else ''
    RED = '\033[91m' if platform.system() != "Windows" else ''
    YELLOW = '\033[93m' if platform.system() != "Windows" else ''
    BLUE = '\033[94m' if platform.system() != "Windows" else ''
    RESET = '\033[0m' if platform.system() != "Windows" else ''
    BOLD = '\033[1m' if platform.system() != "Windows" else ''

def print_colored(message: str, color: str = '') -> None:
    """Print a colored message."""
    print(f"{color}{message}{Colors.RESET}")

def print_header(message: str) -> None:
    """Print a header message."""
    print_colored(f"\n{'='*50}", Colors.BLUE)
    print_colored(f"{message}", Colors.BLUE + Colors.BOLD)
    print_colored(f"{'='*50}", Colors.BLUE)

def print_check(description: str, passed: bool, details: str = "") -> None:
    """Print a check result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    color = Colors.GREEN if passed else Colors.RED
    print_colored(f"{status} {description}", color)
    if details:
        print(f"     {details}")

def run_command(command: List[str], cwd: Optional[Path] = None) -> Optional[subprocess.CompletedProcess]:
    """Run a command and return the result, or None if failed."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None

def check_python_version() -> bool:
    """Check Python version."""
    version = sys.version_info
    required_major, required_minor = 3, 10
    
    is_compatible = version.major >= required_major and version.minor >= required_minor
    details = f"Found {version.major}.{version.minor}.{version.micro}, required 3.10+"
    
    print_check("Python version", is_compatible, details)
    return is_compatible

def check_uv_installation() -> bool:
    """Check if uv is installed and accessible."""
    result = run_command(["uv", "--version"])
    
    if result and result.returncode == 0:
        version = result.stdout.strip()
        print_check("uv package manager", True, f"Version: {version}")
        return True
    else:
        print_check("uv package manager", False, "Not found or not working")
        return False

def check_virtual_environment(project_dir: Path) -> bool:
    """Check if virtual environment exists."""
    venv_dir = project_dir / ".venv"
    
    if venv_dir.exists() and venv_dir.is_dir():
        # Check for Python executable in venv
        if platform.system() == "Windows":
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
        
        if python_exe.exists():
            print_check("Virtual environment", True, f"Found at {venv_dir}")
            return True
    
    print_check("Virtual environment", False, "Not found or incomplete")
    return False

def check_dependencies(project_dir: Path) -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    dependencies = {
        "mcp": False,
        "numpy": False,
        "scikit-learn": False,
        "ollama": False,
        "openai": False,
        "anthropic": False,
        "google.genai": False,
        "fastmcp": False
    }
    
    for dep in dependencies:
        result = run_command([
            "uv", "run", "python", "-c", f"import {dep}; print('OK')"
        ], cwd=project_dir)
        
        if result and result.returncode == 0 and "OK" in result.stdout:
            dependencies[dep] = True
    
    # Print results
    all_good = True
    for dep, installed in dependencies.items():
        print_check(f"Package: {dep}", installed)
        if not installed:
            all_good = False
    
    return dependencies

def check_environment_file(project_dir: Path) -> bool:
    """Check if .env file exists and has basic structure."""
    env_file = project_dir / ".env"
    
    if not env_file.exists():
        print_check("Environment file (.env)", False, "File not found")
        return False
    
    try:
        content = env_file.read_text()
        
        # Check for required keys
        required_keys = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        found_keys = []
        
        for key in required_keys:
            if key in content:
                found_keys.append(key)
        
        details = f"Found {len(found_keys)}/{len(required_keys)} API key entries"
        print_check("Environment file (.env)", True, details)
        return True
        
    except Exception as e:
        print_check("Environment file (.env)", False, f"Error reading file: {e}")
        return False

def check_claude_config(project_dir: Path) -> bool:
    """Check if Claude Desktop config file exists."""
    config_file = project_dir / "claude_desktop_config.json"
    
    if not config_file.exists():
        print_check("Claude Desktop config", False, "File not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check structure
        if "mcpServers" in config and "mcts-mcp-server" in config["mcpServers"]:
            print_check("Claude Desktop config", True, "Valid structure found")
            return True
        else:
            print_check("Claude Desktop config", False, "Invalid structure")
            return False
            
    except Exception as e:
        print_check("Claude Desktop config", False, f"Error reading file: {e}")
        return False

def check_basic_functionality(project_dir: Path) -> bool:
    """Test basic import and functionality."""
    test_code = '''
try:
    from mcts_mcp_server import tools
    from mcts_mcp_server.mcts_core import MCTS
    from mcts_mcp_server.ollama_adapter import OllamaAdapter
    print("BASIC_IMPORT_OK")
except Exception as e:
    print(f"IMPORT_ERROR: {e}")
'''
    
    result = run_command([
        "uv", "run", "python", "-c", test_code
    ], cwd=project_dir)
    
    if result and result.returncode == 0 and "BASIC_IMPORT_OK" in result.stdout:
        print_check("Basic functionality", True, "Core modules import successfully")
        return True
    else:
        error_msg = result.stderr if result else "Command failed to run"
        print_check("Basic functionality", False, f"Import failed: {error_msg}")
        return False

def check_server_startup(project_dir: Path) -> bool:
    """Test if the MCP server can start (basic syntax check)."""
    test_code = '''
try:
    # Just test if we can import and create the main objects without running
    import sys
    sys.path.insert(0, "src")
    from mcts_mcp_server.server import create_server
    print("SERVER_SYNTAX_OK")
except Exception as e:
    print(f"SERVER_ERROR: {e}")
'''
    
    result = run_command([
        "uv", "run", "python", "-c", test_code
    ], cwd=project_dir)
    
    if result and result.returncode == 0 and "SERVER_SYNTAX_OK" in result.stdout:
        print_check("Server startup test", True, "Server code is valid")
        return True
    else:
        error_msg = result.stderr if result else "Command failed to run"
        print_check("Server startup test", False, f"Server test failed: {error_msg}")
        return False

def main():
    """Main verification function."""
    print_header("MCTS MCP Server Installation Verification")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print()
    
    # Get project directory
    project_dir = Path(__file__).parent.resolve()
    print(f"Project directory: {project_dir}")
    
    all_checks = []
    
    # Run all checks
    print_header("Basic Requirements")
    all_checks.append(check_python_version())
    all_checks.append(check_uv_installation())
    
    print_header("Project Structure")
    all_checks.append(check_virtual_environment(project_dir))
    all_checks.append(check_environment_file(project_dir))
    all_checks.append(check_claude_config(project_dir))
    
    print_header("Dependencies")
    deps = check_dependencies(project_dir)
    all_checks.append(all(deps.values()))
    
    print_header("Functionality Tests")
    all_checks.append(check_basic_functionality(project_dir))
    all_checks.append(check_server_startup(project_dir))
    
    # Summary
    print_header("Summary")
    passed = sum(all_checks)
    total = len(all_checks)
    
    if passed == total:
        print_colored(f"🎉 All checks passed! ({passed}/{total})", Colors.GREEN + Colors.BOLD)
        print()
        print_colored("Your MCTS MCP Server installation is ready to use!", Colors.GREEN)
        print()
        print("Next steps:")
        print("1. Add your API keys to the .env file")
        print("2. Configure Claude Desktop with the provided config")
        print("3. Restart Claude Desktop")
        print("4. Test the MCTS tools in Claude")
    else:
        print_colored(f"❌ {total - passed} checks failed ({passed}/{total} passed)", Colors.RED + Colors.BOLD)
        print()
        print_colored("Please fix the failed checks and run the verification again.", Colors.RED)
        print()
        print("Common solutions:")
        print("• Run the setup script again: python setup.py")
        print("• Check the README.md for detailed instructions")
        print("• Ensure all dependencies are installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
