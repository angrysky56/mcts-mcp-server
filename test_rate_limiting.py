#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Gemini Rate Limiting
=========================

This script tests the rate limiting functionality for the Gemini adapter.
"""
import asyncio
import logging
import time
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcts_mcp_server.rate_limiter import RateLimitConfig, TokenBucketRateLimiter, ModelRateLimitManager

async def test_rate_limiter_basic():
    """Test basic rate limiter functionality."""
    print("=== Testing Basic Rate Limiter ===")
    
    # Create a fast rate limiter for testing (6 RPM = 1 request per 10 seconds)
    config = RateLimitConfig(requests_per_minute=6, burst_allowance=2)
    limiter = TokenBucketRateLimiter(config)
    
    print(f"Initial status: {limiter.get_status()}")
    
    # Make burst requests (should be fast)
    print("Making burst requests...")
    for i in range(2):
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        print(f"  Request {i+1}: {elapsed:.3f}s")
    
    # This should be rate limited
    print("Making rate-limited request...")
    start = time.time()
    await limiter.acquire()
    elapsed = time.time() - start
    print(f"  Rate limited request: {elapsed:.3f}s (should be ~10s)")
    
    print(f"Final status: {limiter.get_status()}")
    print()

async def test_gemini_rate_limits():
    """Test Gemini-specific rate limits."""
    print("=== Testing Gemini Rate Limits ===")
    
    manager = ModelRateLimitManager()
    
    # Test the specific models
    test_models = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-1.5-flash-latest", 
        "gemini-1.5-pro",
        "unknown-model"
    ]
    
    for model in test_models:
        limiter = manager.get_limiter(model)
        status = limiter.get_status()
        print(f"{model}:")
        print(f"  Rate: {status['rate_per_minute']:.0f} RPM")
        print(f"  Burst: {status['max_tokens']:.0f}")
        print(f"  Available: {status['available_tokens']:.2f}")
        print()

async def test_concurrent_requests():
    """Test how rate limiting handles concurrent requests."""
    print("=== Testing Concurrent Requests ===")
    
    # Create a restrictive rate limiter (3 RPM = 1 request per 20 seconds)
    config = RateLimitConfig(requests_per_minute=3, burst_allowance=1)
    limiter = TokenBucketRateLimiter(config)
    
    async def make_request(request_id):
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        print(f"Request {request_id}: waited {elapsed:.3f}s")
        return elapsed
    
    # Launch multiple concurrent requests
    print("Launching 3 concurrent requests...")
    start_time = time.time()
    
    tasks = [make_request(i) for i in range(3)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    print(f"Total time for 3 requests: {total_time:.3f}s")
    print(f"Average wait per request: {sum(results)/len(results):.3f}s")
    print()

async def test_model_pattern_matching():
    """Test model pattern matching for rate limits."""
    print("=== Testing Model Pattern Matching ===")
    
    manager = ModelRateLimitManager()
    
    # Test various model names and see what rate limits they get
    test_models = [
        "gemini-2.5-flash-preview-05-20",  # Should match "gemini-2.5-flash-preview"
        "gemini-2.5-flash-preview-06-01",  # Should also match pattern
        "gemini-1.5-flash-8b-001",         # Should match "gemini-1.5-flash-8b"
        "gemini-1.5-flash-latest",         # Should match "gemini-1.5-flash"
        "gemini-1.5-pro-latest",           # Should match "gemini-1.5-pro"
        "gpt-4",                           # Should get default
        "claude-3-opus",                   # Should get default
    ]
    
    for model in test_models:
        limiter = manager.get_limiter(model)
        status = limiter.get_status()
        print(f"{model}: {status['rate_per_minute']:.0f} RPM, {status['max_tokens']:.0f} burst")

async def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Gemini Rate Limiting System")
    print("=" * 50)
    print()
    
    await test_rate_limiter_basic()
    await test_gemini_rate_limits()
    await test_model_pattern_matching()
    await test_concurrent_requests()
    
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
