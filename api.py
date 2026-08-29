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

from registry import NodeRegistry
from round_manager import RoundManager, RoundState, SelectionStrategy
from rpc_client import notify_round_start

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


@app.on_event("startup")
async def startup() -> None:
    await registry.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
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
    nodes = await registry.get_live_nodes()
    return [
        {"node_id": n.node_id, "address": n.address, "hospital_label": n.hospital_label}
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
