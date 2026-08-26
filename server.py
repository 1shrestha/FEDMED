"""FedMed control-plane gRPC server — Week 1.

Runs NodeService (registration/heartbeat/status) and a stub
TrainingControlService so round-start signal path exists, even
though the round orchestration logic itself lands in Week 2.
"""

import time
from concurrent import futures

import grpc

import node_service_pb2 as node_pb2
import node_service_pb2_grpc as node_pb2_grpc
import training_control_pb2 as control_pb2
import training_control_pb2_grpc as control_pb2_grpc

from registry import NodeRegistry

registry = NodeRegistry()


class NodeServiceServicer(node_pb2_grpc.NodeServiceServicer):
    def RegisterNode(self, request, context):
        registry.register(
            node_id=request.node_id,
            hospital_name=request.hospital_name,
            address=request.address,
            dataset_size=request.dataset_size,
        )
        print(f"[server] registered node '{request.node_id}' "
              f"({request.hospital_name}, {request.dataset_size} samples)")
        return node_pb2.RegisterNodeResponse(
            success=True,
            message=f"welcome, {request.node_id}",
            server_time_unix=int(time.time()),
        )

    def Heartbeat(self, request, context):
        ok = registry.heartbeat(request.node_id)
        if not ok:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"unknown node_id '{request.node_id}'")
            return node_pb2.HeartbeatResponse(acknowledged=False)
        return node_pb2.HeartbeatResponse(
            acknowledged=True, server_time_unix=int(time.time())
        )

    def GetNodeStatus(self, request, context):
        entries = []
        for rec in registry.snapshot():
            entries.append(
                node_pb2.NodeStatusEntry(
                    node_id=rec.node_id,
                    hospital_name=rec.hospital_name,
                    state=rec.current_state(),
                    last_heartbeat_unix=int(rec.last_heartbeat),
                    dataset_size=rec.dataset_size,
                )
            )
        return node_pb2.NodeStatusList(nodes=entries)


class TrainingControlServicer(control_pb2_grpc.TrainingControlServiceServicer):
    def NotifyRoundStart(self, request, context):
        # Week 2 will actually call this out to nodes; for now the
        # server can receive/ack it so the contract is exercised.
        print(f"[server] round {request.round_number} start notice for "
              f"nodes {list(request.selected_node_ids)}")
        return control_pb2.Ack(received=True)

    def SubmitUpdateAck(self, request, context):
        print(f"[server] node '{request.node_id}' ack'd update for round "
              f"{request.round_number} (success={request.success})")
        return control_pb2.Ack(received=True)


def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    node_pb2_grpc.add_NodeServiceServicer_to_server(
        NodeServiceServicer(), server
    )
    control_pb2_grpc.add_TrainingControlServiceServicer_to_server(
        TrainingControlServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[server] FedMed control-plane listening on :{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
