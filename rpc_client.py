"""
rpc_client.py

Wraps the generated gRPC stub for NodeCoordination so round_manager.py
never has to import grpc directly. This is the ONLY place that talks to a
node's control-plane port — it sends the "round is starting" signal and
nothing else. Weight exchange is a completely separate connection owned
by Flower's own server (flwr.server.start_server), typically on a
different port on the same node.

Generate the stubs once with:
    python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. \
        proto/coordination.proto
which produces coordination_pb2.py / coordination_pb2_grpc.py referenced below.
"""

from __future__ import annotations

import time

import grpc

import coordination_pb2 as pb
import coordination_pb2_grpc as pb_grpc
from registry import NodeRecord
from round_manager import RoundRecord


async def notify_round_start(
    node: NodeRecord,
    record: RoundRecord,
    flower_server_address: str,
    fit_config: dict | None = None,
    deadline_s: float = 60.0,
) -> pb.RoundStartAck:
    """Async unary call to a single node's NotifyRoundStart RPC.
    Raises on transport failure — RoundManager._fan_out() catches that
    and records it as a non-accept, so failures here are expected and
    handled, not exceptional at the system level."""
    async with grpc.aio.insecure_channel(node.address) as channel:
        stub = pb_grpc.NodeCoordinationStub(channel)
        request = pb.RoundStartRequest(
            round_id=record.round_id,
            round_number=record.round_number,
            flower_server_address=flower_server_address,
            fit_config={k: str(v) for k, v in (fit_config or {}).items()},
            deadline_unix=int(time.time() + deadline_s),
            min_available_clients=len(record.selected_nodes),
        )
        return await stub.NotifyRoundStart(request)
