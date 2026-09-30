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
import base64
import io
import json
import os
import sys
from typing import List, Tuple, Dict, Optional

import flwr as fl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.common import Metrics, parameters_to_ndarrays

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet3d import build_model
from data.synthetic_data import SyntheticBraTSDataset

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.jsonl")

# one fixed held-out sample the server uses to render a segmentation preview
# each round — swap SyntheticBraTSDataset for real_brats_loader in production
_PREVIEW_SAMPLE = SyntheticBraTSDataset(n_samples=1, volume_size=48, seed=12345)[0]


def render_segmentation_png(model) -> str:
    """
    Runs the CURRENT global model on one fixed preview volume and renders the
    middle axial slice's predicted tumor mask as a base64 PNG. This replaces
    the static placeholder SVG in the dashboard with an actual model output —
    it updates every round as the global model improves.
    """
    model.eval()
    with torch.no_grad():
        image = _PREVIEW_SAMPLE["image"].unsqueeze(0)  # [1,4,D,H,W]
        pred = torch.sigmoid(model(image))[0]           # [3,D,H,W]
    mid = pred.shape[1] // 2
    wt, tc, et = pred[0, mid].numpy(), pred[1, mid].numpy(), pred[2, mid].numpy()
    base = _PREVIEW_SAMPLE["image"][0, mid].numpy()  # T1 slice as backdrop

    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=90)
    ax.imshow(base, cmap="gray")
    ax.imshow(np.ma.masked_where(wt < 0.5, wt), cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(tc < 0.5, tc), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(et < 0.5, et), cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates per-client eval metrics (weighted by each hospital's #samples)."""
    total_examples = sum(n for n, _ in metrics)
    dice = sum(m["dice"] * n for n, m in metrics) / total_examples
    return {"dice": dice}


def log_round(server_round: int, loss: Optional[float], metrics: Metrics, seg_png_b64: Optional[str] = None):
    entry = {"round": server_round, "loss": loss, **metrics}
    if seg_png_b64:
        entry["segmentation_png"] = seg_png_b64
    with open(METRICS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[server] round {server_round} -> loss={loss} metrics={ {k: v for k, v in metrics.items()} }")


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with round-by-round metric logging + live segmentation preview for the dashboard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preview_model = build_model()
        self._latest_ndarrays = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            parameters, _ = aggregated
            self._latest_ndarrays = parameters_to_ndarrays(parameters)
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            print(f"[server] round {server_round}: {len(failures)} client(s) failed/dropped — "
                  f"continuing with {len(results)} available result(s)")
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            seg_png = None
            if self._latest_ndarrays is not None:
                keys = list(self._preview_model.state_dict().keys())
                state = {k: torch.tensor(v) for k, v in zip(keys, self._latest_ndarrays)}
                self._preview_model.load_state_dict(state, strict=True)
                seg_png = render_segmentation_png(self._preview_model)
            log_round(server_round, loss, metrics, seg_png)
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
    parser.add_argument("--server-address", type=str,
                         default=f"0.0.0.0:{os.environ.get('FL_PORT', 8080)}",
                         help="Bind address:port for hospital nodes to connect to. "
                              "On a cloud VM, use the machine's public IP/hostname when "
                              "telling clients where to connect (this flag stays 0.0.0.0 to bind all interfaces).")
    args = parser.parse_args()

    main(rounds=args.rounds, n_clients=args.n_clients,
         min_clients=args.min_clients, server_address=args.server_address)
