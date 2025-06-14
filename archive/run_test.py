#!/usr/bin/env python3
import sys
import os

# Add the src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

# Now import and run your module
if __name__ == "__main__":
    from mcts_mcp_server.gemini_adapter import _test_gemini_adapter
    import asyncio
    asyncio.run(_test_gemini_adapter())