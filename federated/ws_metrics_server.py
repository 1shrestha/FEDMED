"""
ws_metrics_server.py  (WEEK 3 DELIVERABLE — "Live Metrics")
--------------------------------------------------------------
Tails federated/metrics.jsonl (written by server.py after every FedAvg
round) and streams new rounds to any connected WebSocket client — i.e.
the React dashboard. Keeping this as a separate tiny process (rather than
bolting websockets into the Flower server) keeps the FL logic and the
demo/UI plumbing decoupled — you can restart the dashboard without
touching a live training run.

Run (while training is running or after):
    pip install websockets
    python federated/ws_metrics_server.py

Dashboard connects to ws://localhost:8765
"""

import asyncio
import json
import os

import websockets

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.jsonl")


async def stream_metrics(websocket):
    print("[ws] dashboard connected")
    last_pos = 0
    # replay everything logged so far, then keep tailing new lines
    try:
        while True:
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                for line in new_lines:
                    line = line.strip()
                    if line:
                        await websocket.send(line)
            await asyncio.sleep(1.0)
    except websockets.exceptions.ConnectionClosed:
        print("[ws] dashboard disconnected")


async def main():
    async with websockets.serve(stream_metrics, "localhost", 8765):
        print("[ws] metrics stream ready at ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
