# Using MCTS for Complex Problem Solving

## When to Use MCTS
Use the MCTS server when you need:
- Deep analysis of complex questions
- Exploration of multiple solution approaches
- Systematic reasoning through difficult problems
- Optimal solutions requiring iterative refinement

## Basic Workflow

### 1. Initialize MCTS
```
Tool: initialize_mcts
Required: question, chat_id
Optional: provider (default: "gemini"), model

Example:
- question: "How can we reduce carbon emissions in urban transportation?"
- chat_id: "urban_transport_analysis_001"
- provider: "gemini" (recommended for performance)
```

### 2. Run Search Iterations
```
Tool: run_mcts_search
Parameters:
- iterations: 3-5 for most problems (more for complex issues)
- simulations_per_iteration: 5-10

Start with: iterations=3, simulations_per_iteration=5
Increase for more thorough analysis
```

### 3. Get Final Analysis
```
Tool: get_synthesis
No parameters needed - uses current MCTS state
Returns comprehensive analysis with best solutions
```

## Pro Tips

1. **Start Simple**: Begin with 3 iterations and 5 simulations
2. **Monitor Status**: Use get_status to check progress
3. **Provider Choice**: Gemini is default and recommended for balanced performance
4. **Unique Chat IDs**: Use descriptive IDs for state persistence
5. **Iterative Refinement**: Run additional searches if needed

## Example Complete Session

1. `initialize_mcts("How to improve team productivity?", "productivity_analysis_001")`
2. `run_mcts_search(iterations=3, simulations_per_iteration=5)`
3. `get_synthesis()` - Get the final recommendations

## Error Handling

- Check get_status if tools return errors
- Ensure provider API keys are set if using non-Gemini providers
- Reinitialize if needed with a new chat_id
