#!/usr/bin/env python3
"""
MCTS MCP Server Startup Test
============================

This script tests the server startup time and basic MCP functionality
to help diagnose timeout issues.
"""

import sys
import time
import subprocess
import json
import os
from pathlib import Path

def print_colored(message, color_code=""):
    """Print colored message."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    if color_code in colors:
        print(f"{colors[color_code]}{message}{colors['reset']}")
    else:
        print(message)

def test_quick_import():
    """Test if basic imports work quickly."""
    print("🔍 Testing quick imports...")
    start_time = time.time()
    
    try:
        import mcp
        import fastmcp
        print_colored("  ✅ MCP packages imported", "green")
    except ImportError as e:
        print_colored(f"  ❌ MCP import failed: {e}", "red")
        return False
    
    try:
        # Test the fast tools import
        sys.path.insert(0, "src")
        from mcts_mcp_server.tools_fast import register_mcts_tools
        print_colored("  ✅ Fast tools imported", "green")
    except ImportError as e:
        print_colored(f"  ❌ Fast tools import failed: {e}", "red")
        return False
    
    elapsed = time.time() - start_time
    print(f"  📊 Import time: {elapsed:.2f} seconds")
    
    if elapsed > 5.0:
        print_colored("  ⚠️  Imports are slow (>5s), may cause timeout", "yellow")
    else:
        print_colored("  ✅ Import speed is good", "green")
    
    return True

def test_server_startup():
    """Test server startup time."""
    print("\n🚀 Testing server startup...")
    
    project_dir = Path(__file__).parent
    
    # Test the fast server startup
    start_time = time.time()
    
    try:
        # Just test import and basic creation (don't actually run)
        cmd = [
            "uv", "run", "python", "-c", 
            """
import sys
sys.path.insert(0, 'src')
from mcts_mcp_server.server import main
print('SERVER_IMPORT_OK')
"""
        ]
        
        result = subprocess.run(
            cmd, 
            cwd=project_dir,
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and "SERVER_IMPORT_OK" in result.stdout:
            print_colored("  ✅ Server imports successfully", "green")
            print(f"  📊 Startup preparation time: {elapsed:.2f} seconds")
            
            if elapsed > 10.0:
                print_colored("  ⚠️  Startup is slow (>10s), may cause timeout", "yellow")
            else:
                print_colored("  ✅ Startup speed is good", "green")
            
            return True
        else:
            print_colored(f"  ❌ Server startup test failed", "red")
            print(f"  Output: {result.stdout}")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_colored("  ❌ Server startup test timed out (>30s)", "red")
        return False
    except Exception as e:
        print_colored(f"  ❌ Server startup test error: {e}", "red")
        return False

def test_environment_setup():
    """Test environment configuration."""
    print("\n🔧 Testing environment setup...")
    
    project_dir = Path(__file__).parent
    
    # Check .env file
    env_file = project_dir / ".env"
    if env_file.exists():
        print_colored("  ✅ .env file exists", "green")
    else:
        print_colored("  ⚠️  .env file missing", "yellow")
    
    # Check virtual environment
    venv_dir = project_dir / ".venv"
    if venv_dir.exists():
        print_colored("  ✅ Virtual environment exists", "green")
    else:
        print_colored("  ❌ Virtual environment missing", "red")
        return False
    
    # Check if in fast mode
    fast_mode = os.getenv("MCTS_FAST_MODE", "true").lower() == "true"
    if fast_mode:
        print_colored("  ✅ Fast mode enabled", "green")
    else:
        print_colored("  ⚠️  Fast mode disabled", "yellow")
    
    return True

def test_claude_config():
    """Test Claude Desktop configuration."""
    print("\n📋 Testing Claude Desktop config...")
    
    config_file = Path(__file__).parent / "claude_desktop_config.json"
    
    if not config_file.exists():
        print_colored("  ❌ claude_desktop_config.json not found", "red")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "mcpServers" in config and "mcts-mcp-server" in config["mcpServers"]:
            server_config = config["mcpServers"]["mcts-mcp-server"]
            
            # Check for fast mode setting
            env_config = server_config.get("env", {})
            fast_mode = env_config.get("MCTS_FAST_MODE", "false").lower() == "true"
            
            if fast_mode:
                print_colored("  ✅ Claude config has fast mode enabled", "green")
            else:
                print_colored("  ⚠️  Claude config doesn't have fast mode", "yellow")
                print_colored("      Add: \"MCTS_FAST_MODE\": \"true\" to env section", "blue")
            
            # Check for timeout setting
            timeout = server_config.get("timeout")
            if timeout:
                print_colored(f"  ✅ Timeout configured: {timeout} seconds", "green")
            else:
                print_colored("  ℹ️  No timeout configured (uses default)", "blue")
            
            print_colored("  ✅ Claude config structure is valid", "green")
            return True
        else:
            print_colored("  ❌ Claude config missing MCTS server entry", "red")
            return False
            
    except json.JSONDecodeError:
        print_colored("  ❌ Claude config has invalid JSON", "red")
        return False
    except Exception as e:
        print_colored(f"  ❌ Error reading Claude config: {e}", "red")
        return False

def main():
    """Run all startup tests."""
    print("🧪 MCTS MCP Server Startup Test")
    print("=" * 40)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    tests = [
        test_environment_setup,
        test_quick_import,
        test_server_startup, 
        test_claude_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_colored("🎉 All tests passed! Server should start quickly.", "green")
        print()
        print("Next steps:")
        print("1. Restart Claude Desktop")
        print("2. Test with: get_config()")
        print("3. If still timing out, check TIMEOUT_FIX.md")
    else:
        print_colored("❌ Some tests failed. Check the issues above.", "red")
        print()
        print("Common fixes:")
        print("1. Run: python setup.py")
        print("2. Enable fast mode in Claude config")
        print("3. Check TIMEOUT_FIX.md for solutions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
