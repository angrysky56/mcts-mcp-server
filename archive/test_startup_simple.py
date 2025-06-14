#!/usr/bin/env python3
"""
Test the MCTS MCP server startup
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        # Test MCP imports
        from mcp.server.fastmcp import FastMCP
        print("✓ MCP imports working")
        
        # Test server import
        from mcts_mcp_server.server import mcp
        print("✓ Server module imports working")
        
        # Test adapter imports
        from mcts_mcp_server.gemini_adapter import GeminiAdapter
        print("✓ Gemini adapter imports working")
        
        from mcts_mcp_server.mcts_config import DEFAULT_CONFIG
        print("✓ Config imports working")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False

def test_server_creation():
    """Test that the server can be created."""
    try:
        from mcts_mcp_server.server import mcp
        print("✓ Server instance created successfully")
        return True
    except Exception as e:
        print(f"✗ Server creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MCTS MCP Server...")
    
    if test_imports() and test_server_creation():
        print("\n🎉 All tests passed! Server should start properly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Check the errors above.")
        sys.exit(1)
