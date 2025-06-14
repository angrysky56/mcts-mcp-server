#!/usr/bin/env python3
"""
Test script to identify the exact issue with the MCTS server
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test if all imports work"""
    try:
        print("Testing imports...")
        import asyncio
        print("✓ asyncio")

        import mcp.server.stdio
        print("✓ mcp.server.stdio")

        import mcp.types as types
        print("✓ mcp.types")

        from mcp.server import Server
        print("✓ mcp.server.Server")

        from google import genai
        print("✓ google.genai")

        print("All imports successful!")
        return True

    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_server_creation():
    """Test basic server creation"""
    try:
        print("\nTesting server creation...")
        sys.path.insert(0, 'src')

        # Import the server module
        from mcts_mcp_server import server
        print("✓ Server module imported")

        # Check if main function exists
        if hasattr(server, 'main'):
            print("✓ main function found")
            print(f"main function type: {type(server.main)}")
        else:
            print("✗ main function not found")

        return True

    except Exception as e:
        print(f"Server creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing MCTS Server Components...")
    print("=" * 50)

    success = True
    success &= test_imports()
    success &= test_server_creation()

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    sys.exit(0 if success else 1)
