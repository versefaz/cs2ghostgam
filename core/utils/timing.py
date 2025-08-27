#!/usr/bin/env python3
"""
Scraper Timing Utilities - Human-like patterns to avoid detection
Implements random jitter, exponential backoff, and respectful delays
"""

import random
import asyncio
import math
import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RateLimitError(Exception):
    """Raised when rate limiting is detected"""
    status_code: int
    retry_after: Optional[int] = None
    message: str = "Rate limit exceeded"


@dataclass
class TimingConfig:
    """Configuration for scraper timing"""
    base_interval_sec: int
    jitter_pct: float
    max_backoff: int = 4
    min_delay: float = 2.0


class HumanLikeTiming:
    """Manages human-like timing patterns for scrapers"""
    
    def __init__(self, config: TimingConfig):
        self.config = config
        self.backoff_level = 0
        self.last_request_time = None
        self.consecutive_errors = 0
        
    def next_delay(self) -> float:
        """Calculate next delay with jitter and backoff"""
        base = self.config.base_interval_sec
        jitter = base * self.config.jitter_pct
        
        # Add random jitter
        jittered_delay = random.uniform(base - jitter, base + jitter)
        
        # Apply exponential backoff if needed
        if self.backoff_level > 0:
            backoff_multiplier = 2 ** min(self.backoff_level, self.config.max_backoff)
            jittered_delay *= backoff_multiplier
            
        # Ensure minimum delay
        return max(jittered_delay, self.config.min_delay)
    
    def on_success(self):
        """Reset backoff on successful request"""
        self.backoff_level = 0
        self.consecutive_errors = 0
        self.last_request_time = datetime.now()
        
    def on_rate_limit(self, retry_after: Optional[int] = None):
        """Increase backoff on rate limit"""
        self.backoff_level = min(self.backoff_level + 1, self.config.max_backoff)
        self.consecutive_errors += 1
        
        if retry_after:
            # Respect server's retry-after header
            additional_delay = retry_after + random.uniform(5, 15)  # Add 5-15 sec buffer
            logger.warning(f"Rate limited, backing off for {additional_delay:.1f} seconds")
            return additional_delay
            
        delay = self.next_delay()
        logger.warning(f"Rate limited, backing off for {delay:.1f} seconds (level {self.backoff_level})")
        return delay
        
    def on_error(self, error_type: str = "general"):
        """Handle general errors with moderate backoff"""
        if error_type in ["connection", "timeout"]:
            self.backoff_level = min(self.backoff_level + 1, 2)  # Lighter backoff for network issues
        else:
            self.consecutive_errors += 1
            
        if self.consecutive_errors > 5:
            self.backoff_level = min(self.backoff_level + 1, self.config.max_backoff)


async def respectful_sleep(base_sec: int, jitter_pct: float, backoff_factor: int = 0):
    """
    Sleep with jitter and exponential backoff
    
    Args:
        base_sec: Base delay in seconds
        jitter_pct: Jitter percentage (0.0 to 1.0)
        backoff_factor: Exponential backoff factor (0 = no backoff)
    """
    jitter = base_sec * jitter_pct
    delay = random.uniform(base_sec - jitter, base_sec + jitter)
    
    if backoff_factor > 0:
        delay *= (2 ** backoff_factor)
        
    # Ensure minimum delay
    delay = max(delay, 2.0)
    
    logger.debug(f"Sleeping for {delay:.1f} seconds (base={base_sec}, jitter={jitter_pct:.1%}, backoff={backoff_factor})")
    await asyncio.sleep(delay)


def get_random_user_agent(user_agents: List[str]) -> str:
    """Get a random user agent from the pool"""
    return random.choice(user_agents)


def should_respect_delay(last_request: Optional[datetime], min_delay: float) -> bool:
    """Check if we should wait before making another request"""
    if not last_request:
        return False
        
    elapsed = (datetime.now() - last_request).total_seconds()
    return elapsed < min_delay


async def wait_for_min_delay(last_request: Optional[datetime], min_delay: float):
    """Wait if minimum delay hasn't passed"""
    if not last_request:
        return
        
    elapsed = (datetime.now() - last_request).total_seconds()
    if elapsed < min_delay:
        wait_time = min_delay - elapsed + random.uniform(0.5, 1.5)  # Add small random buffer
        logger.debug(f"Waiting {wait_time:.1f}s to respect minimum delay")
        await asyncio.sleep(wait_time)


class RequestThrottler:
    """Throttles requests to appear more human-like"""
    
    def __init__(self, max_concurrent: int = 3, min_delay: float = 2.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_delay = min_delay
        self.last_request_times = {}
        
    async def throttled_request(self, domain: str, request_func, *args, **kwargs):
        """Execute a request with throttling"""
        async with self.semaphore:
            # Wait for minimum delay per domain
            if domain in self.last_request_times:
                await wait_for_min_delay(self.last_request_times[domain], self.min_delay)
            
            try:
                result = await request_func(*args, **kwargs)
                self.last_request_times[domain] = datetime.now()
                return result
            except Exception as e:
                self.last_request_times[domain] = datetime.now()
                raise


def calculate_adaptive_delay(success_rate: float, base_delay: float, max_delay: float = None) -> float:
    """
    Calculate adaptive delay based on success rate
    Lower success rate = longer delays
    """
    if max_delay is None:
        max_delay = base_delay * 5
        
    if success_rate >= 0.9:
        return base_delay
    elif success_rate >= 0.7:
        return base_delay * 1.5
    elif success_rate >= 0.5:
        return base_delay * 2.5
    else:
        return min(base_delay * 4, max_delay)


class AdaptiveTimer:
    """Adaptive timing based on success rates"""
    
    def __init__(self, base_config: TimingConfig, window_size: int = 10):
        self.base_config = base_config
        self.window_size = window_size
        self.recent_results = []  # True for success, False for failure
        
    def record_result(self, success: bool):
        """Record the result of a request"""
        self.recent_results.append(success)
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)
            
    def get_success_rate(self) -> float:
        """Get current success rate"""
        if not self.recent_results:
            return 1.0
        return sum(self.recent_results) / len(self.recent_results)
    
    def get_adaptive_delay(self) -> float:
        """Get delay adapted to current success rate"""
        success_rate = self.get_success_rate()
        base_delay = self.base_config.base_interval_sec
        
        adaptive_delay = calculate_adaptive_delay(success_rate, base_delay)
        
        # Apply jitter
        jitter = adaptive_delay * self.base_config.jitter_pct
        final_delay = random.uniform(adaptive_delay - jitter, adaptive_delay + jitter)
        
        return max(final_delay, self.base_config.min_delay)


# Utility functions for common patterns
async def random_startup_delay(max_delay: float = 30.0):
    """Add random delay at startup to spread out scraper starts"""
    delay = random.uniform(0, max_delay)
    logger.info(f"Startup delay: {delay:.1f} seconds")
    await asyncio.sleep(delay)


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Get exponential backoff delay for retries"""
    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
    return min(delay, max_delay)
