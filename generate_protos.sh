#!/usr/bin/env bash
# Regenerates Python gRPC stubs from .proto contracts.
# Run this from the repo root: ./scripts/generate_protos.sh
# Every teammate runs this after pulling changes to the proto/*.proto so
# everyone's generated code stays in sync.

set -e

PROTO_DIR="proto"
OUT_DIRS=("server" "client_node")

for OUT in "${OUT_DIRS[@]}"; do
  python3 -m grpc_tools.protoc \
    -I"${PROTO_DIR}" \
    --python_out="${OUT}" \
    --grpc_python_out="${OUT}" \
    "${PROTO_DIR}"/node_service.proto \
    "${PROTO_DIR}"/training_control.proto
  echo "Generated stubs into ${OUT}/"
done

echo "Done. Commit the regenerated *_pb2.py and *_pb2_grpc.py files."
