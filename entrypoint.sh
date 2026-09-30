#!/usr/bin/env bash
set -e
# Runs both the Flower aggregation server and the metrics WebSocket relay
# in one container. They're independent processes reading/writing the same
# federated/metrics.jsonl file, so both need to run for the dashboard to
# receive live updates during a real training run.
python federated/ws_metrics_server.py &
WS_PID=$!
python federated/server.py --rounds "${FL_ROUNDS:-10}" --n-clients "${FL_N_CLIENTS:-3}" --min-clients "${FL_MIN_CLIENTS:-2}"
kill $WS_PID
