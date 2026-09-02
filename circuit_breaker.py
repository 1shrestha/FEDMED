"""
circuit_breaker.py

One circuit breaker per node_id. This sits ABOVE the retry decorator:
retry.py handles "this one call flaked, try it a couple more times";
this module handles "this node has flaked on the last N calls in a row,
stop even trying it for a while so a dead/flaky hospital can't drag out
every round's fan-out timeout."

States (standard three-state breaker):
    CLOSED     — normal operation, calls go through.
    OPEN       — too many recent failures; calls fail fast (no RPC attempt)
                 until reset_timeout_s elapses.
    HALF_OPEN  — reset timeout elapsed; the next call is allowed through
                 as a probe. Success -> CLOSED. Failure -> OPEN again.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("fedmed.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    """Raised instead of attempting the call when the breaker is OPEN."""


@dataclass
class _NodeBreaker:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3       # consecutive failures before opening
    reset_timeout_s: float = 15.0    # how long to stay OPEN before probing


class NodeCircuitBreakers:
    """Keyed registry of per-node breakers. One instance shared across
    the process; round_manager/rpc_client call before_call()/record_*()
    around each node RPC."""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._breakers: dict[str, _NodeBreaker] = {}

    def _get(self, node_id: str) -> _NodeBreaker:
        return self._breakers.setdefault(node_id, _NodeBreaker())

    def before_call(self, node_id: str) -> None:
        """Call before attempting an RPC. Raises CircuitOpenError if this
        node's breaker is OPEN and hasn't hit its reset timeout yet —
        caller should treat that exactly like a failed call (record it
        as a non-accept) without spending a network round trip on it."""
        b = self._get(node_id)
        if b.state == CircuitState.OPEN:
            if time.time() - b.opened_at >= self._config.reset_timeout_s:
                b.state = CircuitState.HALF_OPEN
                logger.info("node %s: OPEN -> HALF_OPEN (probing)", node_id)
            else:
                raise CircuitOpenError(f"circuit open for node {node_id}")

    def record_success(self, node_id: str) -> None:
        b = self._get(node_id)
        if b.state != CircuitState.CLOSED:
            logger.info("node %s: %s -> CLOSED (recovered)", node_id, b.state)
        b.state = CircuitState.CLOSED
        b.consecutive_failures = 0

    def record_failure(self, node_id: str) -> None:
        b = self._get(node_id)
        b.consecutive_failures += 1

        if b.state == CircuitState.HALF_OPEN:
            # probe failed — back to OPEN immediately, don't wait for threshold
            b.state = CircuitState.OPEN
            b.opened_at = time.time()
            logger.warning("node %s: HALF_OPEN probe failed -> OPEN", node_id)
            return

        if b.consecutive_failures >= self._config.failure_threshold and b.state == CircuitState.CLOSED:
            b.state = CircuitState.OPEN
            b.opened_at = time.time()
            logger.warning(
                "node %s: %d consecutive failures -> OPEN (fast-failing for %.0fs)",
                node_id, b.consecutive_failures, self._config.reset_timeout_s,
            )

    def state_of(self, node_id: str) -> CircuitState:
        return self._get(node_id).state

    def snapshot(self) -> dict[str, str]:
        return {node_id: b.state.value for node_id, b in self._breakers.items()}
