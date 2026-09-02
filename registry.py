"""
registry.py

Simple Postgres-backed service registry for FedMed hospital nodes.

Design goals (per Week 2 spec):
  - No hardcoded IPs: nodes self-register with an address + TTL.
  - "Live" == heartbeat received within TTL window. A background reaper
    (or a plain WHERE clause, see get_live_nodes) treats stale rows as gone
    without needing a separate service like Consul/etcd for a project this size.
  - This is intentionally boring/relational — one table, upserts on heartbeat.

Swap-out path: if this ever needs multi-DC discovery, cross-cluster health
checks, or leader election, promote to Consul/etcd. Not needed yet.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import asyncpg


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS node_registry (
    node_id             TEXT PRIMARY KEY,
    address             TEXT NOT NULL,          -- host:port nodes are reachable at
    hospital_label      TEXT NOT NULL,          -- human-readable silo name, no PHI
    registered_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_heartbeat_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ttl_seconds          INTEGER NOT NULL DEFAULT 30,
    capabilities         JSONB NOT NULL DEFAULT '{}'::jsonb,  -- e.g. {"gpu": true, "dataset_size": 4200}
    status                TEXT NOT NULL DEFAULT 'ACTIVE'       -- ACTIVE | DRAINING | RETIRED
);

CREATE INDEX IF NOT EXISTS idx_node_registry_heartbeat
    ON node_registry (last_heartbeat_at);
"""


@dataclass
class NodeRecord:
    node_id: str
    address: str
    hospital_label: str
    last_heartbeat_at: float  # unix seconds, populated on read
    ttl_seconds: int
    capabilities: dict
    status: str

    def is_live(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.last_heartbeat_at) <= self.ttl_seconds and self.status == "ACTIVE"


class NodeRegistry:
    """Thin async wrapper around the node_registry table."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def register(
        self,
        address: str,
        hospital_label: str,
        ttl_seconds: int = 30,
        capabilities: Optional[dict] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Register a node (or re-register with a fresh id if none supplied).
        Called once on node startup; heartbeat() is used after that."""
        node_id = node_id or str(uuid.uuid4())
        import json

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO node_registry (node_id, address, hospital_label, ttl_seconds, capabilities, status)
                VALUES ($1, $2, $3, $4, $5::jsonb, 'ACTIVE')
                ON CONFLICT (node_id) DO UPDATE
                    SET address = EXCLUDED.address,
                        hospital_label = EXCLUDED.hospital_label,
                        ttl_seconds = EXCLUDED.ttl_seconds,
                        capabilities = EXCLUDED.capabilities,
                        last_heartbeat_at = now(),
                        status = 'ACTIVE'
                """,
                node_id, address, hospital_label, ttl_seconds, json.dumps(capabilities or {}),
            )
        return node_id

    async def heartbeat(self, node_id: str) -> bool:
        """Bump last_heartbeat_at. Returns False if the node was never registered."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE node_registry SET last_heartbeat_at = now(), status = 'ACTIVE' WHERE node_id = $1",
                node_id,
            )
        return result.endswith("1")  # 'UPDATE 1' vs 'UPDATE 0'

    async def deregister(self, node_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE node_registry SET status = 'RETIRED' WHERE node_id = $1", node_id
            )

    async def get_live_nodes(self) -> list[NodeRecord]:
        """Live = ACTIVE status AND heartbeat within its own TTL window.
        This is the query the round manager calls for node selection —
        staleness is computed in SQL so we never orchestrate around a
        node that's actually dead but whose row hasn't been reaped yet."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT node_id, address, hospital_label, capabilities, status,
                       EXTRACT(EPOCH FROM last_heartbeat_at) AS last_heartbeat_unix,
                       ttl_seconds
                FROM node_registry
                WHERE status = 'ACTIVE'
                  AND last_heartbeat_at > now() - (ttl_seconds || ' seconds')::interval
                """
            )
        return [
            NodeRecord(
                node_id=r["node_id"],
                address=r["address"],
                hospital_label=r["hospital_label"],
                last_heartbeat_at=r["last_heartbeat_unix"],
                ttl_seconds=r["ttl_seconds"],
                capabilities=r["capabilities"],
                status=r["status"],
            )
            for r in rows
        ]

    async def reap_stale(self) -> int:
        """Optional janitor pass: mark long-dead rows RETIRED so the table
        doesn't grow forever with zombie entries. Not required for
        get_live_nodes() correctness (that's TTL-filtered already) — this
        is just housekeeping, safe to run on a cron/background task."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE node_registry
                SET status = 'RETIRED'
                WHERE status = 'ACTIVE'
                  AND last_heartbeat_at < now() - (ttl_seconds * 5 || ' seconds')::interval
                """
            )
        return int(result.split(" ")[-1])
