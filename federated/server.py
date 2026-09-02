"""
server.py  (WEEK 1-4 DELIVERABLE)
-----------------------------------
Central aggregation server. Orchestrates FedAvg across the hospital nodes.

WEEK 2 - "Node Resilience" requirement is handled via the strategy config
below: min_fit_clients / min_available_clients are set LOWER than
n_clients, so a training round proceeds even if one of the 3 nodes drops
mid-round. To demo this for your review: start 3 clients, then Ctrl+C one
of them mid-training and show the server still completes the round with
the remaining 2.

Metrics are appended to metrics.jsonl after every round — this is what
ws_server.py streams to the React dashboard (Week 3/4 "Live Metrics").

Run:
    python federated/server.py --rounds 5 --min-clients 2
"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Dict, Optional

import flwr as fl
from flwr.common import Metrics

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.jsonl")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates per-client eval metrics (weighted by each hospital's #samples)."""
    total_examples = sum(n for n, _ in metrics)
    dice = sum(m["dice"] * n for n, m in metrics) / total_examples
    return {"dice": dice}


def log_round(server_round: int, loss: Optional[float], metrics: Metrics):
    entry = {"round": server_round, "loss": loss, **metrics}
    with open(METRICS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[server] round {server_round} -> loss={loss} metrics={metrics}")


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with round-by-round metric logging for the dashboard."""

    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            print(f"[server] round {server_round}: {len(failures)} client(s) failed/dropped — "
                  f"continuing with {len(results)} available result(s)")
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            log_round(server_round, loss, metrics)
        return aggregated


def main(rounds=5, n_clients=3, min_clients=2, server_address="0.0.0.0:8080"):
    # reset metrics log for a fresh run
    open(METRICS_PATH, "w").close()

    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,          # <= n_clients: tolerates dropped nodes
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,    # server won't even start a round below this
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"[server] starting on {server_address}  rounds={rounds}  "
          f"min_clients={min_clients}/{n_clients} (tolerates {n_clients - min_clients} node dropout)")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2, help="min nodes required per round (resilience)")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    main(rounds=args.rounds, n_clients=args.n_clients,
         min_clients=args.min_clients, server_address=args.server_address)
