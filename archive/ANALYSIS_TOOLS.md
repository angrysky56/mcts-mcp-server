# MCTS Analysis Tools

This extension adds powerful analysis tools to the MCTS-MCP Server, making it easy to extract insights and understand results from your MCTS runs.

## Overview

The MCTS Analysis Tools provide a suite of integrated functions to:

1. List and browse MCTS runs
2. Extract key concepts, arguments, and conclusions
3. Generate comprehensive reports
4. Compare results across different runs
5. Suggest improvements for better performance

## Installation

The tools are now integrated directly into the MCTS-MCP Server. No additional setup is required.

## Available Tools

### Browsing and Basic Information

- `list_mcts_runs(count=10, model=None)`: List recent MCTS runs with key metadata
- `get_mcts_run_details(run_id)`: Get detailed information about a specific run
- `get_mcts_solution(run_id)`: Get the best solution from a run

### Analysis and Insights

- `analyze_mcts_run(run_id)`: Perform a comprehensive analysis of a run
- `get_mcts_insights(run_id, max_insights=5)`: Extract key insights from a run
- `extract_mcts_conclusions(run_id)`: Extract conclusions from a run
- `suggest_mcts_improvements(run_id)`: Get suggestions for improvement

### Reporting and Comparison

- `get_mcts_report(run_id, format='markdown')`: Generate a comprehensive report (formats: 'markdown', 'text', 'html')
- `get_best_mcts_runs(count=5, min_score=7.0)`: Get the best runs based on score
- `compare_mcts_runs(run_ids)`: Compare multiple runs to identify similarities and differences

## Usage Examples

### Getting Started

To list your recent MCTS runs:

```python
list_mcts_runs()
```

To get details about a specific run:

```python
get_mcts_run_details('cogito:latest_1745979984')
```

### Extracting Insights

To get key insights from a run:

```python
get_mcts_insights(run_id='cogito:latest_1745979984')
```

### Generating Reports

To generate a comprehensive markdown report:

```python
get_mcts_report(run_id='cogito:latest_1745979984', format='markdown')
```

### Improving Results

To get suggestions for improving a run:

```python
suggest_mcts_improvements(run_id='cogito:latest_1745979984')
```

### Comparing Runs

To compare multiple runs:

```python
compare_mcts_runs(['cogito:latest_1745979984', 'qwen3:0.6b_1745979584'])
```

## Understanding the Results

The analysis tools extract several key elements from MCTS runs:

1. **Key Concepts**: The core ideas and frameworks in the analysis
2. **Arguments For/Against**: The primary arguments on both sides of a question
3. **Conclusions**: The synthesized conclusions or insights from the analysis
4. **Tags**: Automatically generated topic tags from the content

## Troubleshooting

If you encounter any issues with the analysis tools:

1. Check that your MCTS run completed successfully (status: "completed")
2. Verify that the run ID you're using exists and is correct
3. Try listing all runs to see what's available: `list_mcts_runs()`
4. Make sure the `.best_solution.txt` file exists in the run's directory

## Advanced Usage

### Customizing Reports

You can generate reports in different formats:

```python
# Generate a markdown report
report = get_mcts_report(run_id='cogito:latest_1745979984', format='markdown')

# Generate a text report
report = get_mcts_report(run_id='cogito:latest_1745979984', format='text')

# Generate an HTML report
report = get_mcts_report(run_id='cogito:latest_1745979984', format='html')
```

### Finding the Best Runs

To find your best-performing runs:

```python
best_runs = get_best_mcts_runs(count=3, min_score=8.0)
```

This returns the top 3 runs with a score of at least 8.0.
