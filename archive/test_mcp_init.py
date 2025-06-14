#!/usr/bin/env python3
"""
Quick MCP Server Test
====================

Test if the MCTS MCP server responds to initialization quickly.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

async def test_mcp_server():
    """Test if the MCP server responds to initialize quickly."""
    
    project_dir = Path(__file__).parent
    
    print("🧪 Testing MCP server initialization speed...")
    
    # Start the server
    server_cmd = [
        "uv", "run", "python", "-m", "mcts_mcp_server.server"
    ]
    
    try:
        # Start server process
        server_process = await asyncio.create_subprocess_exec(
            *server_cmd,
            cwd=project_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        print("📡 Server started, testing initialization...")
        
        # Send MCP initialize message
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        message_json = json.dumps(init_message) + '\n'
        
        # Record start time
        start_time = time.time()
        
        # Send initialize message
        server_process.stdin.write(message_json.encode())
        await server_process.stdin.drain()
        
        # Try to read response with timeout
        try:
            response_data = await asyncio.wait_for(
                server_process.stdout.readline(),
                timeout=10.0  # 10 second timeout
            )
            
            elapsed = time.time() - start_time
            
            if response_data:
                response_text = response_data.decode().strip()
                print(f"✅ Server responded in {elapsed:.2f} seconds")
                print(f"📋 Response: {response_text[:100]}...")
                
                if elapsed < 5.0:
                    print("🎉 SUCCESS: Server responds quickly!")
                    return True
                else:
                    print("⚠️  Server responds but slowly")
                    return False
            else:
                print("❌ No response received")
                return False
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ TIMEOUT: No response after {elapsed:.2f} seconds")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Clean up server process
        try:
            server_process.terminate()
            await asyncio.wait_for(server_process.wait(), timeout=5.0)
        except:
            try:
                server_process.kill()
                await server_process.wait()
            except:
                pass

def main():
    """Run the test."""
    print("🧪 MCTS MCP Server Initialization Test")
    print("=" * 45)
    
    try:
        result = asyncio.run(test_mcp_server())
        
        if result:
            print("\n🎉 Test PASSED: Server initialization is fast enough")
            return True
        else:
            print("\n❌ Test FAILED: Server initialization is too slow")
            return False
            
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
