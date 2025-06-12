# -*- coding: utf-8 -*-
"""
Rate Limiter Utility
====================

This module provides rate limiting functionality for API calls, specifically
designed for LLM providers with rate limits like Gemini's free tier.
"""
import asyncio
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int
    burst_allowance: int = 1  # How many requests can be made immediately
    
    @property
    def requests_per_second(self) -> float:
        """Convert RPM to RPS for easier calculations."""
        return self.requests_per_minute / 60.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter implementation.
    
    This allows for burst requests up to the bucket capacity, then refills
    tokens at a steady rate based on the configured rate limit.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.burst_allowance)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized rate limiter: {config.requests_per_minute} RPM, "
                   f"burst: {config.burst_allowance}")
    
    async def acquire(self, tokens_needed: int = 1) -> None:
        """
        Acquire tokens for making requests. Will wait if necessary.
        
        Args:
            tokens_needed: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            await self._wait_for_tokens(tokens_needed)
            self.tokens -= tokens_needed
            
            logger.debug(f"Acquired {tokens_needed} tokens, {self.tokens:.2f} remaining")
    
    async def _wait_for_tokens(self, tokens_needed: int) -> None:
        """Wait until enough tokens are available."""
        while True:
            self._refill_tokens()
            
            if self.tokens >= tokens_needed:
                break
                
            # Calculate how long to wait for enough tokens
            tokens_deficit = tokens_needed - self.tokens
            wait_time = tokens_deficit / self.config.requests_per_second
            
            logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s for {tokens_deficit:.2f} tokens")
            await asyncio.sleep(wait_time)
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.config.requests_per_second
        self.tokens = min(self.config.burst_allowance, self.tokens + tokens_to_add)
        
        self.last_refill = now
    
    def get_status(self) -> Dict[str, float]:
        """Get current rate limiter status."""
        self._refill_tokens()
        return {
            "available_tokens": self.tokens,
            "max_tokens": self.config.burst_allowance,
            "rate_per_minute": self.config.requests_per_minute,
            "rate_per_second": self.config.requests_per_second
        }


class ModelRateLimitManager:
    """
    Manages rate limiters for different models.
    
    Allows different rate limits for different models, with sensible defaults
    for known model tiers.
    """
    
    # Default rate limits for known model patterns
    DEFAULT_RATE_LIMITS = {
        # Gemini free tier models
        "gemini-1.5-flash": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
        "gemini-1.5-flash-8b": RateLimitConfig(requests_per_minute=15, burst_allowance=2),
        "gemini-2.0-flash-exp": RateLimitConfig(requests_per_minute=10, burst_allowance=1),
        "gemini-2.5-flash-preview": RateLimitConfig(requests_per_minute=10, burst_allowance=1),
        
        # Gemini paid tier models (higher limits)
        "gemini-1.5-pro": RateLimitConfig(requests_per_minute=360, burst_allowance=5),
        "gemini-2.0-flash-thinking-exp": RateLimitConfig(requests_per_minute=60, burst_allowance=3),
        
        # Default fallback
        "default": RateLimitConfig(requests_per_minute=10, burst_allowance=1)
    }
    
    def __init__(self, custom_limits: Optional[Dict[str, RateLimitConfig]] = None):
        self.rate_limits = self.DEFAULT_RATE_LIMITS.copy()
        if custom_limits:
            self.rate_limits.update(custom_limits)
        
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        
        logger.info(f"Initialized ModelRateLimitManager with {len(self.rate_limits)} rate limit configs")
    
    def _get_rate_limit_config(self, model_name: str) -> RateLimitConfig:
        """Get rate limit config for a model, using pattern matching."""
        # Direct match first
        if model_name in self.rate_limits:
            return self.rate_limits[model_name]
        
        # Pattern matching for model families
        for pattern, config in self.rate_limits.items():
            if pattern != "default" and pattern in model_name:
                logger.debug(f"Matched model '{model_name}' to pattern '{pattern}'")
                return config
        
        # Fallback to default
        logger.debug(f"Using default rate limit for model '{model_name}'")
        return self.rate_limits["default"]
    
    def get_limiter(self, model_name: str) -> TokenBucketRateLimiter:
        """Get or create a rate limiter for a specific model."""
        if model_name not in self.limiters:
            config = self._get_rate_limit_config(model_name)
            self.limiters[model_name] = TokenBucketRateLimiter(config)
            logger.info(f"Created rate limiter for model '{model_name}': {config}")
        
        return self.limiters[model_name]
    
    async def acquire_for_model(self, model_name: str, tokens_needed: int = 1) -> None:
        """Acquire tokens for a specific model."""
        limiter = self.get_limiter(model_name)
        await limiter.acquire(tokens_needed)
    
    def get_all_status(self) -> Dict[str, Dict[str, float]]:
        """Get status for all active rate limiters."""
        return {
            model: limiter.get_status()
            for model, limiter in self.limiters.items()
        }
    
    def add_custom_limit(self, model_name: str, config: RateLimitConfig) -> None:
        """Add a custom rate limit for a specific model."""
        self.rate_limits[model_name] = config
        # Remove existing limiter so it gets recreated with new config
        if model_name in self.limiters:
            del self.limiters[model_name]
        logger.info(f"Added custom rate limit for '{model_name}': {config}")


# Global rate limit manager instance (can be imported and used across modules)
global_rate_limit_manager = ModelRateLimitManager()


async def test_rate_limiter():
    """Test the rate limiter functionality."""
    print("Testing rate limiter...")
    
    # Test high-frequency model
    config = RateLimitConfig(requests_per_minute=60, burst_allowance=3)
    limiter = TokenBucketRateLimiter(config)
    
    print(f"Initial status: {limiter.get_status()}")
    
    # Make some requests quickly (should work due to burst)
    for i in range(3):
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        print(f"Request {i+1}: {elapsed:.3f}s")
    
    # This should be rate limited
    start = time.time()
    await limiter.acquire()
    elapsed = time.time() - start
    print(f"Rate limited request: {elapsed:.3f}s")
    
    print(f"Final status: {limiter.get_status()}")
    
    # Test model manager
    print("\nTesting model manager...")
    manager = ModelRateLimitManager()
    
    # Test different models
    test_models = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-1.5-pro",
        "unknown-model"
    ]
    
    for model in test_models:
        limiter = manager.get_limiter(model)
        status = limiter.get_status()
        print(f"{model}: {status['rate_per_minute']} RPM, {status['max_tokens']} burst")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rate_limiter())
