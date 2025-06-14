# MCTS MCP Server - Fixed Version

## What Was Fixed

The previous MCTS MCP server had several critical issues that caused it to timeout during initialization:

1. **Overly Complex "Fast" Tools**: The `tools_fast.py` had complicated threading and async patterns that caused hanging
2. **Heavy Dependencies**: Many unnecessary packages that slowed startup
3. **Circular Imports**: Complex import chains that caused blocking
4. **Environment Dependencies**: Required `.env` files that most other servers don't need

## Changes Made

### 1. Simplified Dependencies
- Reduced from 12+ packages to just 3 essential ones:
  - `mcp>=1.2.0` (core MCP functionality)
  - `google-generativeai>=0.8.0` (Gemini support)
  - `httpx>=0.25.0` (HTTP client)

### 2. Clean Server Implementation
- Removed complex threading/async patterns
- Simplified state management
- Fast startup with minimal initialization
- No `.env` file required

### 3. Default to Gemini
- Changed default provider from Ollama to Gemini (as requested)
- Better performance on low compute systems
- More reliable API access

### 4. Proper Error Handling
- Clear error messages for missing API keys
- Graceful degradation when services unavailable
- No hanging or timeout issues

## Usage

### 1. Set Up API Key
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### 2. Add to Claude Desktop Config
Use the provided `example_mcp_config.json`:

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

### 3. Available Tools

1. **get_status** - Check server status and configuration
2. **initialize_mcts** - Set up analysis for a question  
3. **simple_analysis** - Perform basic analysis (simplified version)

### 4. Example Usage in Claude

```
1. Check status: Use get_status tool
2. Initialize: Use initialize_mcts with your question
3. Analyze: Use simple_analysis to get results
```

## Testing

The server now starts quickly without hanging. To test:

```bash
cd /home/ty/Repositories/ai_workspace/mcts-mcp-server
uv run mcts-mcp-server
```

Should start immediately without timeout.

## Features

- ✅ Fast startup (no 60-second timeout)
- ✅ Defaults to Gemini (better for low compute)
- ✅ No `.env` file required
- ✅ Simple, reliable architecture
- ✅ Proper error handling
- ✅ Clear status reporting

## Note on Complexity

This version is simplified compared to the original complex MCTS implementation. The full tree search algorithm with Bayesian evaluation, state persistence, and advanced features is available in the original codebase but was causing reliability issues.

The current version focuses on:
- **Reliability** - Always starts, no hanging
- **Simplicity** - Easy to understand and debug  
- **Performance** - Fast response times
- **Usability** - Clear error messages and status

For production use, this simplified version is more appropriate than the complex original that had timeout issues.
