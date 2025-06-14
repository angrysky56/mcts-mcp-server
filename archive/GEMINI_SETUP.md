# Google Gemini Setup Guide

This guide will help you set up the Google Gemini adapter properly with the new `google-genai` library.

## Prerequisites

✅ **Already Done**: You have `google-genai>=1.20.0` installed via your `pyproject.toml`

## 1. Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

## 2. Set Up Environment Variable

Add your API key to your environment. You can use either name:

```bash
# Option 1: Using GEMINI_API_KEY
export GEMINI_API_KEY="your-api-key-here"

# Option 2: Using GOOGLE_API_KEY (also supported)
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file in your project root:

```env
GEMINI_API_KEY=your-api-key-here
```

## 3. Test Your Setup

Run the test script to verify everything is working:

```bash
uv run python test_gemini_setup.py
```

## 4. Usage Examples

### Basic Usage

```python
import asyncio
from mcts_mcp_server.gemini_adapter import GeminiAdapter

async def main():
    # Initialize the adapter
    adapter = GeminiAdapter()

    # Simple completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    response = await adapter.get_completion(model=None, messages=messages)
    print(response)

asyncio.run(main())
```

### With Rate Limiting

```python
# Rate limiting is enabled by default for free tier models
adapter = GeminiAdapter(enable_rate_limiting=True)

# Check rate limit status
status = adapter.get_rate_limit_status()
print(f"Requests remaining: {status['requests_remaining']}")
```

### Streaming Responses

```python
async def stream_example():
    adapter = GeminiAdapter()

    messages = [{"role": "user", "content": "Write a short story about a robot."}]

    async for chunk in adapter.get_streaming_completion(model=None, messages=messages):
        print(chunk, end='', flush=True)

asyncio.run(stream_example())
```

### Using Different Models

```python
# Use a specific model
response = await adapter.get_completion(
    model="gemini-1.5-pro",  # More capable but slower
    messages=messages
)

# Available models:
# - gemini-1.5-flash-latest (default, fast)
# - gemini-1.5-pro (more capable)
# - gemini-2.0-flash-exp (experimental)
# - gemini-2.5-flash-preview-05-20 (preview)
```

## 5. Key Changes from google-generativeai

The new `google-genai` library has a different API:

### Old (google-generativeai)
```python
import google.generativeai as genai
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(messages)
```

### New (google-genai)
```python
from google import genai
client = genai.Client(api_key=api_key)
response = await client.aio.models.generate_content(
    model="gemini-1.5-flash",
    contents=messages
)
```

## 6. Rate Limits

The adapter includes built-in rate limiting for free tier usage:

- **gemini-1.5-flash**: 15 requests/minute
- **gemini-1.5-pro**: 360 requests/minute
- **gemini-2.0-flash-exp**: 10 requests/minute
- **gemini-2.5-flash-preview**: 10 requests/minute

## 7. Troubleshooting

### Common Issues

1. **"API key not provided"**
   - Make sure `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set
   - Check the environment variable is exported correctly

2. **Rate limit errors**
   - Enable rate limiting: `GeminiAdapter(enable_rate_limiting=True)`
   - Check your quota at [Google AI Studio](https://aistudio.google.com/quota)

3. **Import errors**
   - Make sure you're using `google-genai` not `google-generativeai`
   - Check version: `uv run python -c "import google.genai; print(google.genai.__version__)"`

### Getting Help

- [Google AI Studio Documentation](https://ai.google.dev/gemini-api/docs)
- [google-genai GitHub](https://github.com/googleapis/python-aiplatform)
- Check the test script output for detailed error messages

## 8. Next Steps

Once your setup is working:

1. Test with your MCP server
2. Experiment with different models
3. Adjust rate limits if needed
4. Integrate with your MCTS system

Your Gemini adapter is now ready to use with the latest API! 🚀
