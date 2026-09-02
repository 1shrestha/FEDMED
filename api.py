"""
api.py

Control-plane HTTP API. This is what the dashboard (or a curl/Postman call)
hits to kick off a round. It never touches model weights — it only drives
the state machine in round_manager.py and reports back what happened.

Run:
    uvicorn api:app --reload --port 8001

Env:
    DATABASE_URL   postgres DSN for the node registry
    FLOWER_SERVER_ADDRESS   host:port of the Flower server nodes should
                            connect to for the actual fit() round
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from health_monitor import HealthMonitor
from registry import NodeRegistry
from round_manager import RoundManager, RoundState, SelectionStrategy
from rpc_client import circuit_breakers, notify_round_start

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fedmed.api")

app = FastAPI(title="FedMed Coordination Control Plane")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://fedmed:fedmed@localhost:5432/fedmed")
FLOWER_SERVER_ADDRESS = os.environ.get("FLOWER_SERVER_ADDRESS", "0.0.0.0:8080")

registry = NodeRegistry(DATABASE_URL)
round_manager = RoundManager(
    registry=registry,
    notify_fn=notify_round_start,
    flower_server_address=FLOWER_SERVER_ADDRESS,
)
health_monitor = HealthMonitor(registry=registry, round_manager=round_manager)


@app.on_event("startup")
async def startup() -> None:
    await registry.connect()
    # Week 3 recovery/resume: if the server crashed mid-round last time,
    # this reads that state back from Postgres and resolves it (marks it
    # FAILED) instead of the process just starting cold with no memory
    # that anything was ever in flight. Node liveness needs no equivalent
    # step — nodes just keep heartbeating and get picked up naturally.
    await round_manager.recover_on_startup()
    health_monitor.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await health_monitor.stop()
    await registry.close()


# --------------------------------------------------------------------------- #
# Node registry endpoints (nodes call these; Week 1 territory, kept here so
# the whole control plane is runnable from one file set)
# --------------------------------------------------------------------------- #

class RegisterRequest(BaseModel):
    address: str
    hospital_label: str
    ttl_seconds: int = 30
    capabilities: dict = Field(default_factory=dict)


@app.post("/nodes/register")
async def register_node(req: RegisterRequest):
    node_id = await registry.register(
        address=req.address,
        hospital_label=req.hospital_label,
        ttl_seconds=req.ttl_seconds,
        capabilities=req.capabilities,
    )
    return {"node_id": node_id}


@app.post("/nodes/{node_id}/heartbeat")
async def heartbeat(node_id: str):
    ok = await registry.heartbeat(node_id)
    if not ok:
        raise HTTPException(404, "unknown node_id — call /nodes/register first")
    return {"ok": True}


@app.get("/nodes/live")
async def list_live_nodes():
    """ACTIVE nodes only — this is the pool round selection draws from."""
    nodes = await registry.get_live_nodes()
    return [
        {"node_id": n.node_id, "address": n.address, "hospital_label": n.hospital_label}
        for n in nodes
    ]


@app.get("/nodes/health")
async def node_health():
    """Full picture including SUSPECT/DEAD nodes and their circuit
    breaker state — this is what you'd watch live during the
    kill-a-container demo to see detection happen."""
    nodes = await registry.get_all_nodes()
    return [
        {
            "node_id": n.node_id,
            "hospital_label": n.hospital_label,
            "status": n.status,
            "seconds_since_heartbeat": round(n.seconds_since_heartbeat(), 1),
            "circuit_breaker": circuit_breakers.state_of(n.node_id).value,
        }
        for n in nodes
    ]


# --------------------------------------------------------------------------- #
# Round orchestration — the Week 2 deliverable
# --------------------------------------------------------------------------- #

class StartRoundRequest(BaseModel):
    strategy: SelectionStrategy = SelectionStrategy.ALL_ACTIVE
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fit_config: dict = Field(default_factory=dict)
    deadline_s: float = 60.0
    simulated_training_s: float = 0.0
    """How long the round holds in WAITING_FOR_UPDATES after nodes ack,
    standing in for real Flower fit() time. Set this >0 (e.g. 15-20s) for
    the kill-a-node demo so there's an actual window to `docker kill` a
    node container and watch it get excluded before aggregation."""


@app.post("/rounds/start")
async def start_round(req: StartRoundRequest):
    """Trigger a round: select nodes, fan out NotifyRoundStart, collect acks.
    Returns the final RoundRecord state — ROUND_COMPLETE if enough nodes
    accepted, ROUND_FAILED otherwise (see round_manager.start_round)."""
    try:
        record = await round_manager.start_round(
            strategy=req.strategy,
            min_available_clients=req.min_available_clients,
            fraction_fit=req.fraction_fit,
            fit_config=req.fit_config,
            deadline_s=req.deadline_s,
            simulated_training_s=req.simulated_training_s,
        )
    except ValueError as e:
        # e.g. not enough live nodes to even attempt the round
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        # a round is already in flight
        raise HTTPException(409, str(e))

    return {
        "round_id": record.round_id,
        "round_number": record.round_number,
        "state": record.state,
        "selected_nodes": [n.node_id for n in record.selected_nodes],
        "responses": {
            node_id: {"accepted": r.accepted, "reason": r.reason, "rtt_ms": r.rtt_ms}
            for node_id, r in record.responses.items()
        },
        "accepted_count": record.accepted_count,
    }


@app.get("/rounds/current")
async def current_round():
    record = round_manager.current_round
    if record is None:
        return {"state": RoundState.IDLE}
    return {
        "round_id": record.round_id,
        "round_number": record.round_number,
        "state": record.state,
        "accepted_count": record.accepted_count,
        "selected_count": len(record.selected_nodes),
    }
