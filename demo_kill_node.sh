#!/usr/bin/env bash
# demo_kill_node.sh
#
# Week 3 deliverable: trigger a round, kill a node container while the
# round is mid-flight (WAITING_FOR_UPDATES), and show the control plane
# detects the death, excludes the node, and completes the round with
# the remaining nodes.
#
# Requires: docker compose stack already up (docker compose up -d --build)
# and node1/node2/node3 showing under `curl localhost:8001/nodes/live`.

set -euo pipefail

API=http://localhost:8001

echo "== live nodes before the round =="
curl -s "$API/nodes/live" | python3 -m json.tool

echo
echo "== triggering a round (min_available_clients=2, 40s simulated training window) =="
echo "   (nodes register with ttl=6s -> SUSPECT ~6s after last heartbeat, DEAD ~18s after,"
echo "    plus up to 3s health-monitor polling lag — well inside the 40s window)"
curl -s -X POST "$API/rounds/start" \
  -H 'Content-Type: application/json' \
  -d '{"strategy": "all_active", "min_available_clients": 2, "simulated_training_s": 40}' \
  > /tmp/round_result.json &
ROUND_PID=$!

echo "round triggered in the background (pid $ROUND_PID) — give it a moment to reach WAITING_FOR_UPDATES..."
sleep 3

echo
echo "== current round state (should be WAITING_FOR_UPDATES) =="
curl -s "$API/rounds/current" | python3 -m json.tool

echo
echo "== killing node2's container NOW (simulating a hospital dropping mid-round) =="
docker kill week3_coordination-node2-1 || docker kill node2 || {
  echo "container name didn't match — run 'docker ps' and kill the node2 container manually"
}

echo
echo "waiting for the health monitor to notice (dead_multiplier * ttl_seconds, ~90s worst case;"
echo "usually much faster since node2 also stops heartbeating immediately)..."
wait $ROUND_PID

echo
echo "== final round result =="
python3 -m json.tool < /tmp/round_result.json

echo
echo "== node health after the round =="
curl -s "$API/nodes/health" | python3 -m json.tool
