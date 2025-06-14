#!/usr/bin/env python3
"""
Simple test script for MCTS MCP Server
=====================================

This script performs basic tests to verify the installation is working.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import mcp
        print("  ✅ MCP package imported")
    except ImportError as e:
        print(f"  ❌ MCP import failed: {e}")
        return False
    
    try:
        import numpy
        print("  ✅ NumPy imported")
    except ImportError:
        print("  ❌ NumPy import failed")
        return False
    
    try:
        import google.genai
        print("  ✅ Google Gemini imported")
    except ImportError:
        print("  ❌ Google Gemini import failed")
        return False
    
    return True

def test_mcts_imports():
    """Test MCTS-specific imports."""
    print("\n🔍 Testing MCTS imports...")
    
    try:
        from mcts_mcp_server.mcts_core import MCTS
        print("  ✅ MCTS core imported")
    except ImportError as e:
        print(f"  ❌ MCTS core import failed: {e}")
        return False
    
    try:
        from mcts_mcp_server.gemini_adapter import GeminiAdapter
        print("  ✅ Gemini adapter imported")
    except ImportError as e:
        print(f"  ❌ Gemini adapter import failed: {e}")
        return False
    
    try:
        from mcts_mcp_server.tools import register_mcts_tools
        print("  ✅ MCTS tools imported")
    except ImportError as e:
        print(f"  ❌ MCTS tools import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment setup."""
    print("\n🔍 Testing environment...")
    
    project_dir = Path(__file__).parent
    
    # Check .env file
    env_file = project_dir / ".env"
    if env_file.exists():
        print("  ✅ .env file exists")
    else:
        print("  ❌ .env file missing")
        return False
    
    # Check virtual environment
    venv_dir = project_dir / ".venv"
    if venv_dir.exists():
        print("  ✅ Virtual environment exists")
    else:
        print("  ❌ Virtual environment missing")
        return False
    
    # Check Claude config
    claude_config = project_dir / "claude_desktop_config.json"
    if claude_config.exists():
        print("  ✅ Claude Desktop config exists")
    else:
        print("  ❌ Claude Desktop config missing")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 MCTS MCP Server - Simple Test")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    tests = [
        test_basic_imports,
        test_mcts_imports,
        test_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation looks good.")
        print("\nNext steps:")
        print("1. Add API keys to .env file")
        print("2. Configure Claude Desktop") 
        print("3. Test with Claude")
        return True
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTry running: python setup.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
