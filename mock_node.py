"""
mock_node.py

Stand-in for a hospital node's control-plane listener. Implements just
NodeCoordination.NotifyRoundStart, so you can spin up N of these on
different ports and use them as the "live nodes" for a real end-to-end
fan-out test of the Week 2 deliverable, without needing actual Flower
clients running yet.

Usage:
    python mock_node.py --port 9001 --hospital "General Hospital" \
        --registry-url http://localhost:8001

This both serves gRPC on --port AND self-registers + heartbeats against
the FastAPI registry endpoints, so `POST /rounds/start` will actually see
and fan out to it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import grpc
import httpx

import coordination_pb2 as pb
import coordination_pb2_grpc as pb_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s node: %(message)s")
logger = logging.getLogger("fedmed.mock_node")


class NodeCoordinationServicer(pb_grpc.NodeCoordinationServicer):
    def __init__(self, node_id: str, accept: bool = True):
        self.node_id = node_id
        self.accept = accept

    async def NotifyRoundStart(self, request: pb.RoundStartRequest, context):
        logger.info(
            "received NotifyRoundStart round=%s round_number=%d flower_addr=%s deadline=%d",
            request.round_id, request.round_number,
            request.flower_server_address, request.deadline_unix,
        )
        # Real node would kick off flwr.client here, pointed at
        # request.flower_server_address, and return accepted=True
        # immediately (the actual training happens async over Flower's
        # own channel, not inside this RPC).
        return pb.RoundStartAck(
            node_id=self.node_id,
            round_id=request.round_id,
            accepted=self.accept,
            reason="" if self.accept else "simulated decline",
            received_unix=int(time.time()),
        )

    async def NotifyRoundAbort(self, request, context):
        logger.info("round %s aborted: %s", request.round_id, request.reason)
        return pb.Ack(ok=True)

    async def Heartbeat(self, request, context):
        return pb.HeartbeatAck(ok=True, server_unix=int(time.time()))


async def register_and_heartbeat_loop(registry_url: str, address: str, hospital: str, ttl: int):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{registry_url}/nodes/register",
            json={"address": address, "hospital_label": hospital, "ttl_seconds": ttl},
        )
        resp.raise_for_status()
        node_id = resp.json()["node_id"]
        logger.info("registered as node_id=%s address=%s", node_id, address)

        while True:
            await asyncio.sleep(ttl / 3)
            r = await client.post(f"{registry_url}/nodes/{node_id}/heartbeat")
            if r.status_code != 200:
                logger.warning("heartbeat failed: %s", r.text)


async def serve(port: int, hospital: str, registry_url: str, decline: bool, advertise_host: str, ttl_seconds: int):
    # The address registered with the control plane must be reachable
    # FROM the server's container/host — "localhost" only works when
    # everything's on one machine. In Docker Compose, pass this node's
    # own service name (e.g. --advertise-host node1) so the API
    # container can actually dial back into it for NotifyRoundStart.
    address = f"{advertise_host}:{port}"
    servicer = NodeCoordinationServicer(node_id=address, accept=not decline)

    server = grpc.aio.server()
    pb_grpc.add_NodeCoordinationServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    logger.info("mock node listening on %s (%s)", address, hospital)

    heartbeat_task = asyncio.create_task(
        register_and_heartbeat_loop(registry_url, address, hospital, ttl=ttl_seconds)
    )
    try:
        await server.wait_for_termination()
    finally:
        heartbeat_task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hospital", default="Unnamed Hospital")
    parser.add_argument("--registry-url", default="http://localhost:8001")
    parser.add_argument("--decline", action="store_true", help="simulate this node declining rounds")
    parser.add_argument(
        "--advertise-host", default="localhost",
        help="hostname this node registers itself as (use the Docker Compose service name in compose)",
    )
    parser.add_argument(
        "--ttl", type=int, default=6, dest="ttl_seconds",
        help="registry TTL in seconds — kept short by default so the kill-node demo doesn't need a long wait",
    )
    args = parser.parse_args()

    asyncio.run(serve(
        args.port, args.hospital, args.registry_url, args.decline,
        args.advertise_host, args.ttl_seconds,
    ))
