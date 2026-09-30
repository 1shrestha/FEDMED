"""
retry.py

Generic async retry decorator with exponential backoff + full jitter,
scoped to transient gRPC failures. This is the ONE place retry policy
lives — round_manager, rpc_client, and anything else that calls a node
wraps its call with @grpc_retry instead of hand-rolling a try/except
loop. Keeping it centralized means "how many times do we retry a
deadline-exceeded" is a single knob, not a decision made differently in
five call sites.

Backoff formula (full jitter, per the AWS backoff-jitter writeup):
    delay = random.uniform(0, min(cap, base * 2**attempt))
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Callable, TypeVar

import grpc

logger = logging.getLogger("fedmed.retry")

T = TypeVar("T")

# gRPC status codes worth retrying — anything else (e.g. INVALID_ARGUMENT,
# PERMISSION_DENIED) is a real error and retrying it would just waste time
# and hide the actual bug.
RETRYABLE_GRPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,        # connection reset / node unreachable
    grpc.StatusCode.DEADLINE_EXCEEDED,  # transient slowness
    grpc.StatusCode.RESOURCE_EXHAUSTED, # node momentarily overloaded
    grpc.StatusCode.ABORTED,
}


class RetryExhausted(Exception):
    """Raised when every attempt failed. Wraps the last underlying error
    so callers can still inspect what actually went wrong."""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"gave up after {attempts} attempts: {last_error!r}")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, grpc.aio.AioRpcError):
        return exc.code() in RETRYABLE_GRPC_CODES
    if isinstance(exc, asyncio.TimeoutError):
        return True
    return False


def grpc_retry(
    max_attempts: int = 4,
    base_delay_s: float = 0.25,
    max_delay_s: float = 5.0,
    retryable: Callable[[Exception], bool] = _is_retryable,
):
    """Decorator for async functions that make a single gRPC call.
    Retries only on `retryable` exceptions; anything else propagates
    immediately on the first attempt."""

    def decorator(fn: Callable[..., "asyncio.Future[T]"]):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 — deliberately broad, filtered below
                    last_error = exc
                    if not retryable(exc) or attempt == max_attempts:
                        break
                    delay = random.uniform(0, min(max_delay_s, base_delay_s * (2 ** (attempt - 1))))
                    logger.warning(
                        "%s: attempt %d/%d failed (%r), retrying in %.2fs",
                        fn.__qualname__, attempt, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)

            raise RetryExhausted(max_attempts, last_error)

        return wrapper

    return decorator
