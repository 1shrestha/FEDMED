"""
health_monitor.py

Background task that periodically walks the node registry and applies
heartbeat-based failure detection:

    ACTIVE  --(elapsed > ttl_seconds * suspect_multiplier)-->  SUSPECT
    SUSPECT --(elapsed > ttl_seconds * dead_multiplier)-->     DEAD

Recovery (DEAD/SUSPECT -> ACTIVE) happens the moment a real heartbeat
arrives — see NodeRegistry.heartbeat() — this monitor only ever moves
things toward DEAD, never back.

When a node newly becomes DEAD, the monitor calls
round_manager.handle_node_failure(node_id) so an in-flight round finds
out immediately instead of waiting for its own per-node RPC timeout —
that's what makes "kill a node mid-round" get detected promptly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from registry import NodeRegistry
from round_manager import RoundManager

logger = logging.getLogger("fedmed.health_monitor")


class HealthMonitor:
    def __init__(
        self,
        registry: NodeRegistry,
        round_manager: Optional[RoundManager] = None,
        check_interval_s: float = 3.0,
        suspect_multiplier: float = 1.0,   # SUSPECT once elapsed > ttl_seconds
        dead_multiplier: float = 3.0,      # DEAD once elapsed > ttl_seconds * 3
    ):
        self._registry = registry
        self._round_manager = round_manager
        self._check_interval_s = check_interval_s
        self._suspect_multiplier = suspect_multiplier
        self._dead_multiplier = dead_multiplier
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="health-monitor")
        logger.info(
            "health monitor started (interval=%.1fs, suspect>%sx ttl, dead>%sx ttl)",
            self._check_interval_s, self._suspect_multiplier, self._dead_multiplier,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self._check_once()
            except Exception as e:  # noqa: BLE001 — one bad tick must not kill the monitor loop
                logger.error("health monitor tick failed: %s", e)
            await asyncio.sleep(self._check_interval_s)

    async def _check_once(self) -> None:
        await self._registry.reap_stale()  # housekeeping: long-DEAD rows -> RETIRED
        nodes = await self._registry.get_all_nodes()
        for node in nodes:
            elapsed = node.seconds_since_heartbeat()

            if elapsed > node.ttl_seconds * self._dead_multiplier:
                if node.status != "DEAD":
                    logger.warning(
                        "node %s (%s) DEAD — no heartbeat for %.1fs (threshold %.1fs)",
                        node.node_id, node.hospital_label, elapsed,
                        node.ttl_seconds * self._dead_multiplier,
                    )
                    await self._registry.set_status(node.node_id, "DEAD")
                    if self._round_manager is not None:
                        await self._round_manager.handle_node_failure(node.node_id)

            elif elapsed > node.ttl_seconds * self._suspect_multiplier:
                if node.status == "ACTIVE":
                    logger.info(
                        "node %s (%s) SUSPECT — no heartbeat for %.1fs (threshold %.1fs)",
                        node.node_id, node.hospital_label, elapsed,
                        node.ttl_seconds * self._suspect_multiplier,
                    )
                    await self._registry.set_status(node.node_id, "SUSPECT")
