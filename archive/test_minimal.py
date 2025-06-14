#!/usr/bin/env python3
"""
Minimal test for MCTS server imports
"""
try:
    print("Testing FastMCP import...")
    from mcp.server.fastmcp import FastMCP
    print("✓ FastMCP imported")
    
    print("Testing basic modules...")
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Testing config import...")
    from mcts_mcp_server.mcts_config import DEFAULT_CONFIG
    print("✓ Config imported")
    
    print("Testing state manager...")
    from mcts_mcp_server.state_manager import StateManager
    print("✓ State manager imported")
    
    print("Testing gemini adapter...")
    from mcts_mcp_server.gemini_adapter import GeminiAdapter
    print("✓ Gemini adapter imported")
    
    print("Testing server creation...")
    mcp = FastMCP("Test")
    print("✓ MCP server created")
    
    print("\n🎉 All basic imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
