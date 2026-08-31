"""
round_manager.py

Round lifecycle state machine + node selection + fan-out orchestration.

    IDLE -> ROUND_STARTING -> WAITING_FOR_UPDATES -> AGGREGATING -> ROUND_COMPLETE
                                                            |
                                                        (error/timeout)
                                                            v
                                                        ROUND_FAILED

This module is the control plane only. It decides WHO participates and
WHEN, and it fans out the "round is starting" signal over gRPC
(NotifyRoundStart, see proto/coordination.proto). It does NOT touch model
weights — once a node acks, weight exchange happens over Flower's own
server<->client channel (flwr.server), independent of this code path.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from registry import NodeRecord, NodeRegistry

logger = logging.getLogger("fedmed.round_manager")


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #

class RoundState(str, Enum):
    IDLE = "IDLE"
    ROUND_STARTING = "ROUND_STARTING"
    WAITING_FOR_UPDATES = "WAITING_FOR_UPDATES"
    AGGREGATING = "AGGREGATING"
    ROUND_COMPLETE = "ROUND_COMPLETE"
    ROUND_FAILED = "ROUND_FAILED"


# Explicit transition table. Anything not listed here is illegal and raises —
# this is deliberate: a stray state.foo = X somewhere in the codebase should
# never be able to silently skip a step in a training round.
_ALLOWED_TRANSITIONS: dict[RoundState, set[RoundState]] = {
    RoundState.IDLE: {RoundState.ROUND_STARTING},
    RoundState.ROUND_STARTING: {RoundState.WAITING_FOR_UPDATES, RoundState.ROUND_FAILED},
    RoundState.WAITING_FOR_UPDATES: {RoundState.AGGREGATING, RoundState.ROUND_FAILED},
    RoundState.AGGREGATING: {RoundState.ROUND_COMPLETE, RoundState.ROUND_FAILED},
    RoundState.ROUND_COMPLETE: {RoundState.IDLE},
    RoundState.ROUND_FAILED: {RoundState.IDLE},
}


class IllegalTransition(Exception):
    pass


@dataclass
class NodeResponse:
    node_id: str
    accepted: bool
    reason: str = ""
    rtt_ms: Optional[float] = None


@dataclass
class RoundRecord:
    round_id: str
    round_number: int
    state: RoundState = RoundState.IDLE
    selected_nodes: list[NodeRecord] = field(default_factory=list)
    responses: dict[str, NodeResponse] = field(default_factory=dict)
    created_unix: float = field(default_factory=time.time)
    state_history: list[tuple[RoundState, float]] = field(default_factory=list)

    def transition(self, new_state: RoundState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS.get(self.state, set()):
            raise IllegalTransition(f"{self.state} -> {new_state} is not a legal transition")
        logger.info("round %s: %s -> %s", self.round_id, self.state, new_state)
        self.state_history.append((self.state, time.time()))
        self.state = new_state

    @property
    def accepted_count(self) -> int:
        return sum(1 for r in self.responses.values() if r.accepted)


# --------------------------------------------------------------------------- #
# Node selection
# --------------------------------------------------------------------------- #

class SelectionStrategy(str, Enum):
    ALL_ACTIVE = "all_active"     # every live node participates
    FRACTION = "fraction"         # Flower-style fraction_fit sampling


def select_nodes(
    live_nodes: list[NodeRecord],
    strategy: SelectionStrategy,
    min_available_clients: int,
    fraction_fit: float = 1.0,
    rng: Optional[random.Random] = None,
) -> list[NodeRecord]:
    """
    Mirrors the two Flower knobs directly so this maps 1:1 onto
    flwr.server.strategy.FedAvg(min_available_clients=..., fraction_fit=...)
    when Shrestha/Chevvakaula wire up the actual strategy object:

      - min_available_clients: the round can't start at all unless at least
        this many live nodes exist (a hard gate, not a sampling target).
      - fraction_fit: of the available pool, what fraction gets sampled in
        for this round. 1.0 == ALL_ACTIVE. <1.0 == FRACTION.

    Raises ValueError if min_available_clients isn't met — caller should
    transition the round to ROUND_FAILED rather than start with too few
    nodes to be statistically meaningful for FL aggregation.
    """
    rng = rng or random.Random()

    if len(live_nodes) < min_available_clients:
        raise ValueError(
            f"only {len(live_nodes)} live nodes, need >= {min_available_clients} "
            f"(min_available_clients) to start a round"
        )

    if strategy == SelectionStrategy.ALL_ACTIVE or fraction_fit >= 1.0:
        return list(live_nodes)

    if strategy == SelectionStrategy.FRACTION:
        sample_size = max(min_available_clients, math.ceil(len(live_nodes) * fraction_fit))
        sample_size = min(sample_size, len(live_nodes))
        return rng.sample(live_nodes, sample_size)

    raise ValueError(f"unknown selection strategy: {strategy}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

# Injected so this module has no hard gRPC/Flower import at module load time —
# swap this for the real grpc stub call (rpc_client.notify_round_start) or a
# mock in tests. Signature matches rpc_client.notify_round_start exactly.
NotifyFn = Callable[[NodeRecord, "RoundRecord", str, dict, float], "asyncio.Future"]


class RoundManager:
    def __init__(
        self,
        registry: NodeRegistry,
        notify_fn: NotifyFn,
        flower_server_address: str,
        per_node_timeout_s: float = 5.0,
    ):
        self._registry = registry
        self._notify_fn = notify_fn
        self._flower_server_address = flower_server_address
        self._per_node_timeout_s = per_node_timeout_s
        self._current: Optional[RoundRecord] = None
        self._round_counter = 0
        self._lock = asyncio.Lock()

    @property
    def current_round(self) -> Optional[RoundRecord]:
        return self._current

    async def start_round(
        self,
        strategy: SelectionStrategy = SelectionStrategy.ALL_ACTIVE,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fit_config: Optional[dict] = None,
        deadline_s: float = 60.0,
    ) -> RoundRecord:
        """Entry point for the API layer. Runs the whole
        ROUND_STARTING -> WAITING_FOR_UPDATES -> AGGREGATING -> ROUND_COMPLETE
        arc for a single round and returns the final record.

        Aggregation itself is stubbed here (that's Flower's/ the FL
        engineer's job once weights land) — this owns getting nodes
        signaled and their round-participation acks collected."""
        async with self._lock:
            if self._current is not None and self._current.state not in (
                RoundState.IDLE, RoundState.ROUND_COMPLETE, RoundState.ROUND_FAILED,
            ):
                raise RuntimeError(f"round {self._current.round_id} still in progress")

            self._round_counter += 1
            record = RoundRecord(round_id=str(uuid.uuid4()), round_number=self._round_counter)
            self._current = record

        try:
            record.transition(RoundState.ROUND_STARTING)

            live_nodes = await self._registry.get_live_nodes()
            selected = select_nodes(
                live_nodes, strategy, min_available_clients, fraction_fit
            )
            record.selected_nodes = selected
            logger.info(
                "round %s: selected %d/%d live nodes (strategy=%s, fraction_fit=%s)",
                record.round_id, len(selected), len(live_nodes), strategy, fraction_fit,
            )

            record.transition(RoundState.WAITING_FOR_UPDATES)
            await self._fan_out(record, fit_config or {}, deadline_s)

            if record.accepted_count < min_available_clients:
                record.transition(RoundState.ROUND_FAILED)
                logger.warning(
                    "round %s failed: only %d/%d nodes accepted (need %d)",
                    record.round_id, record.accepted_count, len(selected), min_available_clients,
                )
                return record

            record.transition(RoundState.AGGREGATING)
            # NOTE: actual weight aggregation happens inside Flower's
            # strategy.aggregate_fit(), triggered by Flower's own server
            # once enough clients report in over the payload channel.
            # This control plane just marks the round complete once that
            # signal comes back (wired up by whoever owns the Flower
            # server loop — left as a hook here).
            record.transition(RoundState.ROUND_COMPLETE)
            return record

        except (ValueError, IllegalTransition) as e:
            logger.error("round %s aborted before fan-out: %s", record.round_id, e)
            if record.state != RoundState.IDLE:
                try:
                    record.transition(RoundState.ROUND_FAILED)
                except IllegalTransition:
                    pass
            raise

    async def _fan_out(
        self, record: RoundRecord, fit_config: dict, deadline_s: float
    ) -> None:
        """Fires NotifyRoundStart at every selected node concurrently and
        logs each response as it comes in. A node that times out or errors
        is recorded as a non-accept rather than raising — one bad node
        should never take down the round."""
        deadline_unix = int(time.time() + deadline_s)

        async def _notify_one(node: NodeRecord) -> None:
            start = time.perf_counter()
            try:
                ack = await asyncio.wait_for(
                    self._notify_fn(
                        node, record, self._flower_server_address, fit_config, deadline_s
                    ),
                    timeout=self._per_node_timeout_s,
                )
                rtt_ms = (time.perf_counter() - start) * 1000
                response = NodeResponse(
                    node_id=node.node_id,
                    accepted=bool(getattr(ack, "accepted", False)),
                    reason=getattr(ack, "reason", ""),
                    rtt_ms=rtt_ms,
                )
            except asyncio.TimeoutError:
                response = NodeResponse(node.node_id, accepted=False, reason="timeout")
            except Exception as e:  # noqa: BLE001 — one node's gRPC error must not sink the round
                response = NodeResponse(node.node_id, accepted=False, reason=f"error: {e}")

            record.responses[node.node_id] = response
            logger.info(
                "round %s: node %s (%s) -> accepted=%s reason=%r rtt_ms=%s",
                record.round_id, node.node_id, node.hospital_label,
                response.accepted, response.reason,
                f"{response.rtt_ms:.1f}" if response.rtt_ms else None,
            )

        await asyncio.gather(*(_notify_one(n) for n in record.selected_nodes))
