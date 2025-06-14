#!/usr/bin/env python3
"""
Test the fixed MCTS server basic functionality
"""
import os
import sys

def test_basic_imports():
    """Test that basic Python functionality works."""
    try:
        import json
        import logging
        print("✓ Basic Python imports working")
        return True
    except Exception as e:
        print(f"✗ Basic import error: {e}")
        return False

def test_environment():
    """Test environment setup."""
    try:
        # Test path
        print(f"✓ Current directory: {os.getcwd()}")
        
        # Test API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            print("✓ Gemini API key found")
        else:
            print("⚠ No Gemini API key found (set GEMINI_API_KEY)")
        
        return True
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False

def test_server_structure():
    """Test that server file exists and has basic structure."""
    try:
        server_path = "src/mcts_mcp_server/server.py"
        if os.path.exists(server_path):
            print(f"✓ Server file exists: {server_path}")
            
            # Check file has basic content
            with open(server_path, 'r') as f:
                content = f.read()
                if "FastMCP" in content and "def main" in content:
                    print("✓ Server file has expected structure")
                    return True
                else:
                    print("✗ Server file missing expected components")
                    return False
        else:
            print(f"✗ Server file not found: {server_path}")
            return False
    except Exception as e:
        print(f"✗ Server structure error: {e}")
        return False

def test_config():
    """Test MCP config file."""
    try:
        config_path = "example_mcp_config.json"
        if os.path.exists(config_path):
            print(f"✓ Example config exists: {config_path}")
            
            # Test JSON validity
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                if "mcpServers" in config:
                    print("✓ Config has valid structure")
                    return True
                else:
                    print("✗ Config missing mcpServers")
                    return False
        else:
            print(f"✗ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Fixed MCTS MCP Server...")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_environment,
        test_server_structure,
        test_config
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Set GEMINI_API_KEY environment variable")
        print("2. Add example_mcp_config.json to Claude Desktop config")
        print("3. Restart Claude Desktop")
        print("4. Use the MCTS tools in Claude")
        return True
    else:
        print(f"\n❌ {len(tests) - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
