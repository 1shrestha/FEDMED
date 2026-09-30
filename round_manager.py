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
    min_available_clients: int = 1
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

    @property
    def has_quorum(self) -> bool:
        return self.accepted_count >= self.min_available_clients


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

    async def _persist(self, record: RoundRecord) -> None:
        """Write the round's full current state to Postgres. Called after
        every transition so a server restart mid-round has something to
        recover from (see recover_on_startup()) instead of the round
        simply vanishing with the process."""
        try:
            await self._registry.save_round(
                round_id=record.round_id,
                round_number=record.round_number,
                state=record.state.value,
                min_available_clients=record.min_available_clients,
                selected_nodes=[n.node_id for n in record.selected_nodes],
                responses={
                    node_id: {"accepted": r.accepted, "reason": r.reason, "rtt_ms": r.rtt_ms}
                    for node_id, r in record.responses.items()
                },
            )
        except Exception as e:  # noqa: BLE001 — persistence failing must not crash the round
            logger.error("round %s: failed to persist state to Postgres: %s", record.round_id, e)

    async def recover_on_startup(self) -> None:
        """Call once, right after registry.connect(). If the previous
        process died mid-round, the row in Postgres will still show a
        non-terminal state (WAITING_FOR_UPDATES/AGGREGATING) — we can't
        trust that in-flight fan-out actually completed, so we mark it
        ROUND_FAILED and log it rather than silently resuming into
        unknown node state. Nodes themselves don't need any special
        recovery — they just keep heartbeating against the registry,
        which is already durable in Postgres."""
        latest = await self._registry.load_latest_round()
        if latest is None:
            logger.info("no prior round found in Postgres — starting cold, IDLE")
            return

        if latest.state in (RoundState.ROUND_COMPLETE.value, RoundState.ROUND_FAILED.value):
            logger.info(
                "recovered prior round %s (#%d) — already terminal (%s), nothing to do",
                latest.round_id, latest.round_number, latest.state,
            )
            return

        logger.warning(
            "recovered round %s (#%d) was mid-flight (%s) when the server last stopped — "
            "marking ROUND_FAILED, a fresh round can be triggered normally",
            latest.round_id, latest.round_number, latest.state,
        )
        await self._registry.save_round(
            round_id=latest.round_id,
            round_number=latest.round_number,
            state=RoundState.ROUND_FAILED.value,
            min_available_clients=latest.min_available_clients,
            selected_nodes=latest.selected_nodes,
            responses=latest.responses,
        )
        self._round_counter = max(self._round_counter, latest.round_number)

    async def start_round(
        self,
        strategy: SelectionStrategy = SelectionStrategy.ALL_ACTIVE,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fit_config: Optional[dict] = None,
        deadline_s: float = 60.0,
        simulated_training_s: float = 0.0,
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
            record = RoundRecord(
                round_id=str(uuid.uuid4()),
                round_number=self._round_counter,
                min_available_clients=min_available_clients,
            )
            self._current = record

        try:
            record.transition(RoundState.ROUND_STARTING)
            await self._persist(record)

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
            await self._persist(record)
            await self._fan_out(record, fit_config or {}, deadline_s)
            await self._persist(record)

            if simulated_training_s > 0 and record.has_quorum:
                # Real Flower fit() takes real wall-clock time on the
                # nodes' own channel — the round legitimately sits in
                # WAITING_FOR_UPDATES while that happens, not just for
                # the instant it takes to send the "start" signal. That
                # window is exactly when a node can die mid-round: the
                # lock isn't held here, so health_monitor's background
                # task can call handle_node_failure() concurrently and
                # exclude a node before we re-check quorum below.
                logger.info(
                    "round %s: waiting %.1fs for simulated training before aggregating",
                    record.round_id, simulated_training_s,
                )
                await asyncio.sleep(simulated_training_s)

            # Tolerance policy: EXCLUDE AND PROCEED IF QUORUM MET.
            # Nodes that declined, timed out, errored, or (via
            # handle_node_failure, called by health_monitor concurrently)
            # went DEAD mid-round are simply absent from accepted_count.
            # We only abort the round if too few remain — we never retry
            # individual nodes or abort just because *some* node dropped.
            if not record.has_quorum:
                record.transition(RoundState.ROUND_FAILED)
                await self._persist(record)
                logger.warning(
                    "round %s failed: only %d/%d selected nodes accepted (need %d for quorum)",
                    record.round_id, record.accepted_count, len(selected), min_available_clients,
                )
                return record

            record.transition(RoundState.AGGREGATING)
            await self._persist(record)
            # NOTE: actual weight aggregation happens inside Flower's
            # strategy.aggregate_fit(), triggered by Flower's own server
            # once enough clients report in over the payload channel.
            # This control plane just marks the round complete once that
            # signal comes back (wired up by whoever owns the Flower
            # server loop — left as a hook here). A node dying during
            # AGGREGATING itself (after fan-out, before this line) is
            # still caught by handle_node_failure() re-checking quorum
            # and can flip this round to ROUND_FAILED before we get here.
            if not record.has_quorum:
                record.transition(RoundState.ROUND_FAILED)
                await self._persist(record)
                logger.warning(
                    "round %s failed during aggregation: quorum lost (%d/%d, need %d)",
                    record.round_id, record.accepted_count, len(selected), min_available_clients,
                )
                return record

            record.transition(RoundState.ROUND_COMPLETE)
            await self._persist(record)
            return record

        except (ValueError, IllegalTransition) as e:
            logger.error("round %s aborted before fan-out: %s", record.round_id, e)
            if record.state != RoundState.IDLE:
                try:
                    record.transition(RoundState.ROUND_FAILED)
                    await self._persist(record)
                except IllegalTransition:
                    pass
            raise

    async def handle_node_failure(self, node_id: str) -> None:
        """Called by health_monitor.py the moment it marks a node DEAD.
        If that node is part of the round currently in flight, exclude it
        from the accepted set and re-check quorum right away rather than
        waiting for the round's own timeout — this is what makes "kill a
        node mid-round" actually get detected promptly instead of the
        round just hanging until deadline_s expires.

        Implements the "exclude and proceed if quorum met" tolerance
        policy: the round keeps going with whatever nodes remain if
        min_available_clients is still satisfied; only flips to
        ROUND_FAILED if excluding this node drops it below quorum."""
        record = self._current
        if record is None or record.state not in (
            RoundState.WAITING_FOR_UPDATES, RoundState.AGGREGATING
        ):
            return  # no round in flight, or this node isn't relevant right now

        if not any(n.node_id == node_id for n in record.selected_nodes):
            return  # node wasn't part of this round anyway

        previous = record.responses.get(node_id)
        if previous is not None and not previous.accepted:
            return  # already excluded, nothing new to do

        record.responses[node_id] = NodeResponse(
            node_id=node_id, accepted=False, reason="node marked DEAD mid-round"
        )
        logger.warning(
            "round %s: node %s went DEAD mid-round, excluding it (%d/%d accepted remain, need %d)",
            record.round_id, node_id, record.accepted_count,
            len(record.selected_nodes), record.min_available_clients,
        )
        await self._persist(record)

        if not record.has_quorum:
            try:
                record.transition(RoundState.ROUND_FAILED)
                await self._persist(record)
                logger.warning(
                    "round %s: quorum lost after node %s dropped — round failed",
                    record.round_id, node_id,
                )
            except IllegalTransition:
                pass  # round already moved past a state where this applies

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
