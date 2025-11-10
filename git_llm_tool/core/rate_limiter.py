"""Rate limiting and retry mechanisms for API calls."""

import time
import random
from typing import Any, Callable, Optional
from dataclasses import dataclass
from functools import wraps
import logging

from git_llm_tool.core.exceptions import ApiError


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    rate_limit_delay: float = 0.5  # Minimum delay between requests


class RateLimiter:
    """Rate limiter with exponential backoff and jitter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.last_request_time = 0.0
        self.logger = logging.getLogger(__name__)

    def wait_if_needed(self):
        """Ensure minimum delay between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def exponential_backoff(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = min(
            self.config.initial_delay * (self.config.backoff_multiplier ** attempt),
            self.config.max_delay
        )

        if self.config.jitter:
            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)

        return delay

    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                # Wait before making request (rate limiting)
                self.wait_if_needed()

                # Execute the function
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if it's a rate limit error
                is_rate_limit_error = self._is_rate_limit_error(e)

                if not is_rate_limit_error and attempt == 0:
                    # If it's not a rate limit error, don't retry on first attempt
                    # unless it's a network error
                    if not self._is_retryable_error(e):
                        raise e

                if attempt == self.config.max_retries - 1:
                    # Last attempt, re-raise the exception
                    break

                delay = self.exponential_backoff(attempt)
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        # All retries exhausted
        raise ApiError(f"API call failed after {self.config.max_retries} attempts: {last_exception}")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is related to rate limiting."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "429",
            "throttled",
            "rate_limit_exceeded"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_str = str(error).lower()
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "502",
            "503",
            "504",
            "internal server error",
            "service unavailable",
            "gateway timeout"
        ]
        return any(indicator in error_str for indicator in retryable_indicators)


def rate_limited(config: Optional[RateLimitConfig] = None):
    """Decorator for rate limiting API calls."""
    if config is None:
        config = RateLimitConfig()

    rate_limiter = RateLimiter(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return rate_limiter.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator