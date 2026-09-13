"""
rpc_client.py

Wraps the generated gRPC stub for NodeCoordination so round_manager.py
never has to import grpc directly. This is the ONLY place that talks to a
node's control-plane port — it sends the "round is starting" signal and
nothing else. Weight exchange is a completely separate connection owned
by Flower's own server (flwr.server.start_server), typically on a
different port on the same node.

Week 3: every call here goes through, in order:
    1. circuit breaker check (before_call) — fail fast, no network hop,
       if this node has been flaky recently.
    2. @grpc_retry — exponential backoff + jitter for transient errors
       (UNAVAILABLE, DEADLINE_EXCEEDED, ...) on the underlying RPC itself.
    3. circuit breaker record_success/record_failure — feeds back into
       whether future calls short-circuit.

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
from circuit_breaker import CircuitOpenError, NodeCircuitBreakers
from registry import NodeRecord
from retry import RetryExhausted, grpc_retry
from round_manager import RoundRecord

# One breaker registry shared by every call this process makes. Imported
# by api.py too, so /nodes/health can report live breaker state.
circuit_breakers = NodeCircuitBreakers()


@grpc_retry(max_attempts=4, base_delay_s=0.25, max_delay_s=5.0)
async def _call_notify_round_start(
    node: NodeRecord, request: pb.RoundStartRequest, timeout_s: float
) -> pb.RoundStartAck:
    async with grpc.aio.insecure_channel(node.address) as channel:
        stub = pb_grpc.NodeCoordinationStub(channel)
        return await stub.NotifyRoundStart(request, timeout=timeout_s)


async def notify_round_start(
    node: NodeRecord,
    record: RoundRecord,
    flower_server_address: str,
    fit_config: dict | None = None,
    deadline_s: float = 60.0,
) -> pb.RoundStartAck:
    """Async call to a single node's NotifyRoundStart RPC, guarded by a
    circuit breaker and retried with backoff+jitter on transient errors.

    Raises CircuitOpenError or RetryExhausted on failure — round_manager's
    fan-out catches both and records the node as a non-accept, so a flaky
    or dead node never crashes the round."""
    circuit_breakers.before_call(node.node_id)  # raises CircuitOpenError if OPEN

    request = pb.RoundStartRequest(
        round_id=record.round_id,
        round_number=record.round_number,
        flower_server_address=flower_server_address,
        fit_config={k: str(v) for k, v in (fit_config or {}).items()},
        deadline_unix=int(time.time() + deadline_s),
        min_available_clients=len(record.selected_nodes),
    )

    try:
        ack = await _call_notify_round_start(node, request, timeout_s=min(deadline_s, 10.0))
    except RetryExhausted:
        circuit_breakers.record_failure(node.node_id)
        raise
    except Exception:
        circuit_breakers.record_failure(node.node_id)
        raise

    circuit_breakers.record_success(node.node_id)
    return ack
