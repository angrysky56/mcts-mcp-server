# MCTS MCP Server - Timeout Fix Guide

## Issue: MCP Initialization Timeout

If you're seeing logs like:
```
Error: MCP error -32001: Request timed out
Server transport closed unexpectedly
```
This is not true Claude:
This means the MCTS server is taking too long to respond to Claude Desktop's initialization request.

## ✅ Solution 1: Use Fast Mode (Recommended)

The server now includes a fast startup mode that defers heavy operations:

**Update your Claude Desktop config to use fast mode:**

```json
{
  "mcpServers": {
    "mcts-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcts-mcp-server",
        "run",
        "mcts-mcp-server"
      ],
      "env": {
        "UV_PROJECT_ENVIRONMENT": "/path/to/mcts-mcp-server/.venv",
        "MCTS_FAST_MODE": "true"
      }
    }
  }
}
```

## ✅ Solution 2: Increase Claude Desktop Timeout

Add a longer timeout to your Claude Desktop config:

```json
{
  "mcpServers": {
    "mcts-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcts-mcp-server",
        "run",
        "mcts-mcp-server"
      ],
      "env": {
        "UV_PROJECT_ENVIRONMENT": "/path/to/mcts-mcp-server/.venv"
      },
      "timeout": 120
    }
  }
}
```

## ✅ Solution 3: Pre-warm Dependencies

If using Ollama, make sure it's running and responsive:

```bash
# Start Ollama server
ollama serve

# Check if it's responding
curl http://localhost:11434/

# Pre-pull a model if needed
ollama pull qwen3:latest
```

## ✅ Solution 4: Check System Performance

Slow startup can be caused by:

- **Low RAM**: MCTS requires sufficient memory
- **Slow disk**: State files and dependencies on slow storage
- **CPU load**: Other processes competing for resources

**Quick checks:**
```bash
# Check available RAM
free -h

# Check disk speed
df -h

# Check CPU load
top
```

## ✅ Solution 5: Use Alternative Server Script

Try the ultra-fast server version:

```bash
# In your Claude Desktop config, use:
"command": "uv",
"args": [
  "--directory",
  "/path/to/mcts-mcp-server",
  "run",
  "python",
  "src/mcts_mcp_server/server_fast.py"
]
```

## 🔧 Testing Your Fix

1. **Restart Claude Desktop** completely after config changes
2. **Check server logs** in Claude Desktop developer tools
3. **Test with simple command**: `get_config()` should respond quickly
4. **Monitor startup time**: Should respond within 10-30 seconds

## 📊 Fast Mode vs Standard Mode

| Feature | Fast Mode | Standard Mode |
|---------|-----------|---------------|
| Startup Time | < 10 seconds | 30-60+ seconds |
| Memory Usage | Lower initial | Higher initial |
| Ollama Check | Deferred | At startup |
| State Loading | Lazy | Immediate |
| Recommended | ✅ Yes | For debugging only |

## 🆘 Still Having Issues?

1. **Check Python version**: Ensure Python 3.10+
2. **Verify dependencies**: Run `python verify_installation.py`
3. **Test manually**: Run `uv run mcts-mcp-server` directly
4. **Check Claude Desktop logs**: Look for specific error messages
5. **Try different timeout values**: Start with 120, increase if needed

## 💡 Prevention Tips

- **Keep Ollama running** if using local models
- **Close unnecessary applications** to free resources
- **Use SSD storage** for better I/O performance
- **Monitor system resources** during startup

---

**The fast mode should resolve timeout issues for most users. If problems persist, the issue may be system-specific and require further investigation.**
