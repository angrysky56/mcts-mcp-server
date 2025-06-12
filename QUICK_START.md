# Quick Start Guide - MCTS MCP Server

Welcome! This guide will get you up and running with the MCTS MCP Server in just a few minutes.

## 🚀 One-Command Setup

**Step 1: Clone and Setup**
```bash
git clone https://github.com/angrysky56/mcts-mcp-server.git
cd mcts-mcp-server
python setup.py
```

**That's it!** The setup script handles everything automatically.

## 🔧 Platform-Specific Alternatives

If the Python setup doesn't work, try these platform-specific scripts:

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup_windows.bat
```

## ✅ Verify Installation

```bash
python verify_installation.py
```

This checks that everything is working correctly.

## 🔑 Configure API Keys

Edit the `.env` file and add your API keys:

```env
# Choose one or more providers
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here  
GEMINI_API_KEY=your-gemini-api-key-here

# For local models (no API key needed)
# Just make sure Ollama is running: ollama serve
```

## 🖥️ Add to Claude Desktop

1. Copy the contents of `claude_desktop_config.json`
2. Add to your Claude Desktop config file:
   - **Linux/macOS**: `~/.config/claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
3. **Update the paths** in the config to match your installation
4. Restart Claude Desktop

## 🧪 Test It Works

Open Claude Desktop and try:

```
Can you help me analyze the implications of artificial intelligence on human creativity using the MCTS system?
```

Claude should use the MCTS tools to perform deep analysis!

## 🎯 Quick Commands

Once working, you can use these in Claude:

```python
# Set up your preferred model
set_active_llm(provider_name="gemini", model_name="gemini-2.0-flash")

# Start analysis  
initialize_mcts(question="Your question", chat_id="analysis_001")

# Run the search
run_mcts(iterations=2, simulations_per_iteration=5)

# Get results
generate_synthesis()
```

## 🆘 Need Help?

**Common Issues:**

1. **Python not found**: Install Python 3.10+ from python.org
2. **Permission denied**: Run `chmod +x setup.sh` on Linux/macOS
3. **Claude not seeing tools**: Check config file paths and restart Claude Desktop
4. **Import errors**: Run the verification script: `python verify_installation.py`

**Still stuck?** Check the full README.md for detailed troubleshooting.

## 🎉 You're Ready!

The MCTS MCP Server is now installed and ready to help Claude perform sophisticated analysis using Monte Carlo Tree Search algorithms. Enjoy exploring complex topics with AI-powered reasoning!

---

**For detailed documentation, see:**
- `README.md` - Complete documentation
- `USAGE_GUIDE.md` - Detailed usage examples
- `ANALYSIS_TOOLS.md` - Analysis tools guide
