"""Simulated hospital node — Week 1.

Registers itself with the FedMed control-plane server, then heartbeats
on a loop. Run several of these (different NODE_ID/HOSPITAL_NAME) to
simulate multiple hospitals — that's Week 1 deliverable.
"""

import os
import time

import grpc

import node_service_pb2 as node_pb2
import node_service_pb2_grpc as node_pb2_grpc

SERVER_ADDR = os.environ.get("SERVER_ADDR", "localhost:50051")
NODE_ID = os.environ.get("NODE_ID", "hospital-a")
HOSPITAL_NAME = os.environ.get("HOSPITAL_NAME", "Hospital A")
DATASET_SIZE = int(os.environ.get("DATASET_SIZE", "1000"))
HEARTBEAT_INTERVAL_SEC = float(os.environ.get("HEARTBEAT_INTERVAL_SEC", "5"))

# Week 3 will replace this with a real retry/backoff interceptor.
MAX_CONNECT_RETRIES = 5
RETRY_BACKOFF_SEC = 2


def connect_with_retry():
    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        try:
            channel = grpc.insecure_channel(SERVER_ADDR)
            grpc.channel_ready_future(channel).result(timeout=5)
            return channel
        except grpc.FutureTimeoutError:
            print(f"[{NODE_ID}] server not ready, retry {attempt}/"
                  f"{MAX_CONNECT_RETRIES} in {RETRY_BACKOFF_SEC}s")
            time.sleep(RETRY_BACKOFF_SEC)
    raise RuntimeError(f"[{NODE_ID}] could not reach server at {SERVER_ADDR}")


def run():
    channel = connect_with_retry()
    stub = node_pb2_grpc.NodeServiceStub(channel)

    reg_response = stub.RegisterNode(
        node_pb2.RegisterNodeRequest(
            node_id=NODE_ID,
            hospital_name=HOSPITAL_NAME,
            address=f"{NODE_ID}:0",  # placeholder, not used for callbacks yet
            dataset_size=DATASET_SIZE,
        )
    )
    print(f"[{NODE_ID}] register response: success={reg_response.success} "
          f"message='{reg_response.message}'")

    while True:
        try:
            hb_response = stub.Heartbeat(
                node_pb2.HeartbeatRequest(
                    node_id=NODE_ID, client_time_unix=int(time.time())
                )
            )
            print(f"[{NODE_ID}] heartbeat ack={hb_response.acknowledged}")
        except grpc.RpcError as e:
            print(f"[{NODE_ID}] heartbeat failed: {e.code()} — will retry "
                  f"next interval")
        time.sleep(HEARTBEAT_INTERVAL_SEC)


if __name__ == "__main__":
    run()
