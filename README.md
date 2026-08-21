# FedMed — Week 1: Node Communication Skeleton

Distributed-systems control-plane for FedMed. This week proves out node
registration, discovery, and heartbeat over gRPC between the central
server and simulated hospital nodes — the foundation the rest of the
resilience/orchestration work (Weeks 2–4) builds on.

## What's here

```
proto/                    # shared .proto contracts (source of truth)
  node_service.proto       #   RegisterNode, Heartbeat, GetNodeStatus
  training_control.proto   #   NotifyRoundStart, SubmitUpdateAck (control signals only —
                            #   actual model weights flow through Flower, not this service)
server/                   # central control-plane gRPC server
  server.py
  registry.py              # in-memory node registry (Postgres-backed from Week 3)
client_node/               # simulated hospital node
  client.py                 # registers + heartbeats on a loop
  check_status.py           # manual utility: prints GetNodeStatus table
scripts/
  generate_protos.sh        # regenerate *_pb2.py / *_pb2_grpc.py after editing protos
docker-compose.yml          # server + 3 simulated hospitals
```

## Run it locally (no Docker)

```bash
# one-time, or after editing any .proto file
./scripts/generate_protos.sh

# terminal 1
cd server && pip install -r requirements.txt && python3 server.py

# terminal 2, 3, 4 — one simulated hospital each
cd client_node && pip install -r requirements.txt
NODE_ID=hospital-a HOSPITAL_NAME="Hospital A" DATASET_SIZE=1000 python3 client.py
NODE_ID=hospital-b HOSPITAL_NAME="Hospital B" DATASET_SIZE=800  python3 client.py
NODE_ID=hospital-c HOSPITAL_NAME="Hospital C" DATASET_SIZE=1200 python3 client.py

# terminal 5 — check the registry
cd client_node && python3 check_status.py
```

## Run it with Docker Compose (the actual Week 1 deliverable)

```bash
docker compose up --build
```

This brings up the server plus 3 simulated hospital containers (`hospital-a/b/c`),
each registering and heartbeating automatically. Confirm the deliverable with:

```bash
docker compose exec hospital-a python3 check_status.py
```

Expected output — all three nodes listed as `ACTIVE`:

```
node_id       hospital      state     dataset_size
hospital-a    Hospital A    ACTIVE    1000
hospital-b    Hospital B    ACTIVE    800
hospital-c    Hospital C    ACTIVE    1200
```

## Design notes for the team

- **Why a separate control-plane service from Flower:** Flower already runs its
  own gRPC channel for the FL training loop (weight exchange). This service is a
  thin layer *around* that — node lifecycle, discovery, and round-start signaling —
  so orchestration/resilience logic stays decoupled from the ML framework internals.
- **NodeState (ACTIVE/SUSPECT/DEAD):** currently computed from heartbeat recency
  in `registry.py`. Week 3 wires this into failure detection and round-exclusion logic.
- **Retry stub in `client.py`:** `connect_with_retry()` is a placeholder for the
  proper exponential-backoff + circuit-breaker interceptor coming in Week 3 — it's
  intentionally simple for now so Week 1 stays focused on the happy path.
- **In-memory registry:** fine for Week 1 demo; becomes a bug the moment the server
  restarts mid-round. Flagged for the Postgres-backed rewrite in Week 3.

## Next (Week 2 preview)

Wire `NotifyRoundStart` into an actual round-lifecycle state machine on the
server (`IDLE → ROUND_STARTING → WAITING_FOR_UPDATES → AGGREGATING → ROUND_COMPLETE`),
and have nodes react to it instead of the server just logging the call.
