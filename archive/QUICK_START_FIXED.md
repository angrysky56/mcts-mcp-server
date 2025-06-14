# MCTS MCP Server - Quick Start Guide

## Fixed and Working ✅

This MCTS MCP server has been **fixed** to resolve timeout issues and now:
- Starts quickly (no 60-second hangs)
- Defaults to Gemini (better for low compute)
- Requires minimal setup
- No complex dependencies

## Quick Setup

### 1. Get Gemini API Key
Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 2. Set Environment Variable
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Add to Claude Desktop
Copy `example_mcp_config.json` content to your Claude Desktop config:

**Location**: `~/.config/claude-desktop/config.json` (Linux/Mac) or `%APPDATA%/Claude/config.json` (Windows)

```json
{
  "mcpServers": {
    "mcts-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/mcts-mcp-server",
        "run",
        "mcts-mcp-server"
      ],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

### 4. Restart Claude Desktop
The server will now be available in Claude.

## Using the Tools

### Check Status
```
Use the get_status tool to verify the server is working
```

### Initialize Analysis
```
Use initialize_mcts with your question:
- question: "How can we improve team productivity?"
```

### Get Analysis
```
Use simple_analysis to get insights on your question
```

## What's Fixed

- ❌ **Before**: Complex threading causing 60s timeouts
- ✅ **After**: Simple, fast startup in <2 seconds

- ❌ **Before**: Required Ollama + heavy dependencies  
- ✅ **After**: Just Gemini API key needed

- ❌ **Before**: Complex state management causing hangs
- ✅ **After**: Simple, reliable operation

## Support

If you have issues:
1. Check that GEMINI_API_KEY is set correctly
2. Verify the path in config.json matches your system
3. Restart Claude Desktop after config changes

The server now works reliably and focuses on core functionality over complex features that were causing problems.
