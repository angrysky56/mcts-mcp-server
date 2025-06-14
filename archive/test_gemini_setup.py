#!/usr/bin/env python3
"""
Test script for Gemini Adapter setup
====================================

Quick test to verify your Gemini setup is working correctly.
"""

import asyncio
import os
import sys

# Add src to path so we can import our modules
sys.path.insert(0, 'src')

from mcts_mcp_server.gemini_adapter import GeminiAdapter


async def test_gemini_setup():
    """Test basic Gemini functionality"""

    print("🧪 Testing Gemini Adapter Setup...")
    print("=" * 50)

    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API key found!")
        print("   Please set either GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        print("   You can get a free API key at: https://aistudio.google.com/app/apikey")
        return False

    print(f"✅ API key found: {api_key[:8]}...")

    try:
        # Initialize adapter
        adapter = GeminiAdapter(api_key=api_key, enable_rate_limiting=False)
        print(f"✅ Adapter initialized successfully!")
        print(f"   Default model: {adapter.model_name}")
        print(f"   Client type: {type(adapter.client).__name__}")

        # Test simple completion
        print("\n🤖 Testing simple completion...")
        messages = [
            {"role": "user", "content": "Say hello and confirm you're working. Keep it short."}
        ]

        response = await adapter.get_completion(model=None, messages=messages)
        print(f"✅ Completion successful!")
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        # Test streaming completion
        print("\n📡 Testing streaming completion...")
        stream_messages = [
            {"role": "user", "content": "Count to 3, one number per line."}
        ]

        chunks = []
        async for chunk in adapter.get_streaming_completion(model=None, messages=stream_messages):
            chunks.append(chunk)
            if len(chunks) >= 5:  # Limit chunks for testing
                break

        print(f"✅ Streaming successful!")
        print(f"   Received {len(chunks)} chunks")
        print(f"   Sample: {''.join(chunks)[:50]}...")

        print("\n🎉 All tests passed! Your Gemini setup is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gemini_setup())
    sys.exit(0 if success else 1)
