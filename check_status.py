"""Utility: queries GetNodeStatus and prints registry — handy for
manually verifying the Week 1 deliverable ("3 dummy nodes appear in
GetNodeStatus")."""

import os

import grpc

import node_service_pb2 as node_pb2
import node_service_pb2_grpc as node_pb2_grpc

SERVER_ADDR = os.environ.get("SERVER_ADDR", "localhost:50051")

STATE_NAMES = {0: "UNKNOWN", 1: "ACTIVE", 2: "SUSPECT", 3: "DEAD"}

if __name__ == "__main__":
    channel = grpc.insecure_channel(SERVER_ADDR)
    stub = node_pb2_grpc.NodeServiceStub(channel)
    status = stub.GetNodeStatus(node_pb2.GetNodeStatusRequest())
    print(f"{'node_id':<14}{'hospital':<14}{'state':<10}{'dataset_size':<14}")
    for n in status.nodes:
        print(f"{n.node_id:<14}{n.hospital_name:<14}"
              f"{STATE_NAMES.get(n.state, n.state):<10}{n.dataset_size:<14}")
