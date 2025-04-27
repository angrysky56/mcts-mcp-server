# MCTS MCP Server Usage Guide

This guide explains how to effectively use the MCTS MCP Server with Claude for deep, explorative analysis.

## Setup

1. Run the setup script to prepare the environment:
   ```bash
   ./setup.sh
   ```
   
   The setup script will:
   - Install UV (Astral UV) if not already installed
   - Create a virtual environment using UV
   - Install dependencies with UV
   - Create necessary state directory

2. Add the MCP server configuration to Claude Desktop:
   - Copy the content from `claude_desktop_config.json`
   - Add it to your Claude Desktop configuration file (typically at `~/.claude/claude_desktop_config.json`)
   - Update paths in the configuration if necessary
   - Restart Claude Desktop

## Using the MCTS Analysis with Claude

Once the MCP server is configured, Claude can leverage MCTS for deep analysis of topics. Here are some example conversation patterns:

### Starting a New Analysis

Simply provide a question, topic, or text that you want Claude to analyze deeply:

```
Analyze the ethical implications of artificial general intelligence.
```

Claude will:
1. Initialize the MCTS system
2. Generate an initial analysis
3. Run MCTS iterations to explore different perspectives
4. Find the best analysis and generate a synthesis
5. Present the results

### Continuing an Analysis

To build upon a previous analysis in the same chat session:

```
Continue exploring the technological feasibility aspects.
```

Claude will:
1. Load the state from the previous analysis
2. Start a new MCTS run that builds upon the previous knowledge
3. Leverage learned approach preferences and avoid unfit areas
4. Present an updated analysis

### Asking About the Last Run

To get information about the previous analysis:

```
What was the best score and key insights from your last analysis run?
```

Claude will summarize the results of the previous MCTS run, including the best score, approach preferences, and analysis tags.

### Asking About the Process

To learn more about how the MCTS analysis works:

```
How does your MCTS analysis process work?
```

Claude will explain the MCTS algorithm and how it's used for analysis.

### Viewing/Changing Configuration

To see or modify the MCTS configuration:

```
Show me the current MCTS configuration.
```

Or:

```
Can you update the MCTS configuration to use 3 iterations and 8 simulations per iteration?
```

## Understanding MCTS Analysis Output

The MCTS analysis output typically includes:

1. **Initial Analysis**: The starting point of the exploration
2. **Best Analysis Found**: The highest-scored analysis discovered through MCTS
3. **Analysis Tags**: Key concepts identified in the analysis
4. **Final Synthesis**: A conclusive statement that integrates the key insights

## Advanced Usage

### Adjusting Parameters

You can ask Claude to modify parameters for deeper or more focused analysis:

- Increase `max_iterations` for more thorough exploration
- Increase `simulations_per_iteration` for more simulations per iteration
- Adjust `exploration_weight` to balance exploration vs. exploitation
- Set `early_stopping` to false to ensure all iterations complete

### Using Different Approaches

You can guide Claude to explore specific philosophical approaches:

```
Continue the analysis using a more empirical approach.
```

Or:

```
Can you explore this topic from a more critical perspective?
```

## Development Notes

If you want to run or test the server directly during development:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the server directly 
uv run server.py

# Or use the MCP CLI tools
uv run -m mcp dev server.py
```

## Troubleshooting

- If Claude doesn't recognize the MCTS server, check that Claude Desktop is correctly configured and restarted
- If analysis seems shallow, ask for more iterations or simulations
- If Claude says it can't continue an analysis, it might mean no state was saved from a previous run
- If you encounter dependency issues, try `uv pip sync requirements.txt` to ensure exact package versions
