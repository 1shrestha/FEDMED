# Week 2 — Distributed Coordination Layer

Control-plane service that decides *who* participates in a training round
and tells them to start. It is explicitly the **signal layer**, not the
payload layer — actual model weights move over Flower's own
server↔client channel, started separately once a node accepts.

## Files

| File | What it does |
|---|---|
| `proto/coordination.proto` | `NodeCoordination` service — `NotifyRoundStart`, `NotifyRoundAbort`, `Heartbeat`. Control signals only, no weight tensors. |
| `registry.py` | Postgres-backed node registry. Nodes `register()`/`heartbeat()`; `get_live_nodes()` filters by TTL in SQL — no cron/reaper required for correctness, `reap_stale()` is optional housekeeping. |
| `round_manager.py` | The state machine (`IDLE → ROUND_STARTING → WAITING_FOR_UPDATES → AGGREGATING → ROUND_COMPLETE`, plus `ROUND_FAILED`), node selection (`ALL_ACTIVE` / `FRACTION`), and the concurrent fan-out that calls `NotifyRoundStart` on every selected node. |
| `rpc_client.py` | The actual gRPC call `round_manager` invokes per node — swap-in target for `notify_fn`. |
| `api.py` | FastAPI app: `POST /rounds/start` triggers a round end-to-end; `GET /rounds/current` polls status; `POST /nodes/register` + `POST /nodes/{id}/heartbeat` for node lifecycle. |
| `mock_node.py` | Standalone gRPC node you can run N copies of to demo the fan-out without real hospital clients. |
| `test_round_manager_smoke.py` | No-infra tests for the state machine + selection logic (fake registry, fake RPC). |

## How the pieces map to the spec

- **State machine** — `round_manager.RoundState` + `_ALLOWED_TRANSITIONS`. Illegal jumps (e.g. `IDLE → AGGREGATING`) raise `IllegalTransition` rather than silently succeeding — this is deliberate so a bug can't skip a step.
- **Node selection** — `select_nodes()` takes `min_available_clients` (hard gate — refuses to start below it) and `fraction_fit` (sampling fraction), the same two knobs Flower's `FedAvg` strategy takes. `ALL_ACTIVE` == `fraction_fit=1.0`; `FRACTION` samples `ceil(N * fraction_fit)` nodes. **Coordinate with Shrestha/Chevvakaula**: when they wire the real `flwr.server.strategy.FedAvg(...)`, these two values should come from the same config so the control plane's selection and Flower's own internal sampling agree on the same node set.
- **Coordinate handoff** — `RoundStartRequest` carries `flower_server_address` + `fit_config`, nothing else. A node's ack just means "I'll dial into Flower for this round," not that training happened.
- **Service registry** — `registry.py`, Postgres + TTL, no hardcoded IPs. Swap-out path to Consul/etcd is noted in the file if this ever needs multi-cluster discovery.
- **Deliverable (trigger a round, watch fan-out, log responses)** — see below.

## Running the live demo

```bash
pip install -r requirements.txt
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/coordination.proto

# 1. Postgres reachable at DATABASE_URL (default: postgresql://fedmed:fedmed@localhost:5432/fedmed)
export DATABASE_URL=postgresql://fedmed:fedmed@localhost:5432/fedmed

# 2. start the control plane
uvicorn api:app --port 8001 &

# 3. start a few mock nodes (each self-registers + heartbeats)
python mock_node.py --port 9001 --hospital "General Hospital" &
python mock_node.py --port 9002 --hospital "St. Mary's" &
python mock_node.py --port 9003 --hospital "City Clinic" --decline &   # simulates a decline

# 4. confirm they're visible
curl localhost:8001/nodes/live

# 5. trigger a round
curl -X POST localhost:8001/rounds/start \
  -H 'Content-Type: application/json' \
  -d '{"strategy": "all_active", "min_available_clients": 2, "fraction_fit": 1.0}'
```

You'll see the control plane log each `NotifyRoundStart` response as it
comes back (accept/decline/timeout, with RTT), and the API response shows
the final round state — `ROUND_COMPLETE` if enough nodes accepted,
`ROUND_FAILED` otherwise.

## No-infra sanity check

If Postgres/gRPC isn't set up yet, `python test_round_manager_smoke.py`
exercises the state machine and selection logic against in-memory fakes.

## Notes for next week (failure handling / resilience)

- `_fan_out` already isolates per-node failures (timeout/exception →
  recorded as a non-accept, doesn't crash the round) — Week 3 probably
  wants retry-with-backoff here rather than a single attempt.
- `reap_stale()` exists but isn't scheduled anywhere yet — needs a
  background task or cron.
- No round timeout/cancellation path from `WAITING_FOR_UPDATES` yet
  beyond the per-node RPC timeout — worth adding a round-level deadline
  that transitions to `ROUND_FAILED` if too much wall-clock time passes.
