"""
Small FastAPI/WebSocket service that the training strategy pushes round
metrics into, and that the React dashboard subscribes to for live updates.
Runs as its own process/container so a dashboard reconnect never touches
the Flower gRPC training loop.
"""
import asyncio
import json
import logging
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger("fedmed.server.metrics_hub")

app = FastAPI(title="FedMed Metrics Hub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the dashboard's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_clients: Set[WebSocket] = set()
_history: list = []


@app.websocket("/ws/metrics")
async def metrics_ws(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        for entry in _history:
            await websocket.send_text(json.dumps(entry))
        while True:
            await websocket.receive_text()  # keepalive / ignored pings
    except WebSocketDisconnect:
        _clients.discard(websocket)


@app.post("/push")
async def push(request: Request):
    """Called by the Flower server process (a separate container) to push
    a new round's metrics for fan-out to connected dashboards."""
    entry = await request.json()
    await broadcast(entry)
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok", "connected_dashboards": len(_clients)}


@app.get("/history")
async def history():
    return _history


async def broadcast(entry: dict):
    _history.append(entry)
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(entry))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def push_metrics_sync(entry: dict):
    """Called from the (synchronous) Flower strategy callback."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        asyncio.ensure_future(broadcast(entry))
    else:
        loop.run_until_complete(broadcast(entry))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
