"""
main_client.py
==============
FedMed FL Client entry point.

Launches a Flower client (one hospital node) that:
- Trains locally on synthetic (or real) hospital data
- Signs all model updates with ECDSA
- Optionally masks updates via SecAgg before transmission
- Uses mTLS for the gRPC channel to the FL server

Usage
-----
Basic (local dev, no mTLS):

    python main_client.py --client-id 0

With mTLS:

    python main_client.py \\
        --client-id 0 \\
        --mtls \\
        --cert-dir certs \\
        --server localhost:8080

Full configuration:

    python main_client.py \\
        --client-id 1 \\
        --server fl-server:8080 \\
        --mtls \\
        --cert-dir certs \\
        --model cnn \\
        --secagg \\
        --total-clients 5 \\
        --local-epochs 2 \\
        --data-samples 512
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedmed.client")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedMed Hospital FL Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--client-id", type=int, required=True, help="Hospital node ID (0-indexed)")
    p.add_argument("--server", default="localhost:8080", help="FL server address (default: localhost:8080)")
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp", help="Model type (default: mlp)")
    p.add_argument("--total-clients", type=int, default=3, help="Total number of FL clients (for SecAgg, default: 3)")
    p.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per round (default: 1)")
    p.add_argument("--data-samples", type=int, default=256, help="Synthetic dataset size (default: 256)")
    p.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    p.add_argument("--mtls", action="store_true", help="Enable mutual TLS for gRPC channel")
    p.add_argument("--cert-dir", default="certs", help="Certificate directory (default: certs/)")
    p.add_argument("--secagg", action="store_true", default=True, help="Enable SecAgg masking (default: on)")
    p.add_argument("--no-secagg", dest="secagg", action="store_false", help="Disable SecAgg masking")
    return p.parse_args()


def build_model(model_type: str, device: str):
    """Instantiate the correct model architecture."""
    import torch
    from model.simple_model import MedCNN, MedMLP

    if model_type == "cnn":
        model = MedCNN(in_channels=1, num_classes=2)
    else:
        model = MedMLP(input_dim=20, hidden_dim=64, output_dim=2)

    return model.to(torch.device(device))


def build_data_loaders(
    model_type: str,
    n_samples: int,
    client_id: int,
):
    """Build train and validation DataLoaders from synthetic hospital data."""
    from client.fl_client import make_synthetic_dataset
    from torch.utils.data import random_split, DataLoader

    # Each client gets different data (different seed = non-IID simulation)
    full_loader = make_synthetic_dataset(
        n_samples=n_samples,
        n_features=20,
        n_classes=2,
        model_type=model_type,
        seed=client_id * 42,
    )
    dataset = full_loader.dataset
    n_val = max(1, n_samples // 5)
    n_train = n_samples - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FedMed FL Client starting")
    logger.info("  Client ID   : %d", args.client_id)
    logger.info("  Server      : %s", args.server)
    logger.info("  Model       : %s", args.model)
    logger.info("  SecAgg      : %s", args.secagg)
    logger.info("  mTLS        : %s", args.mtls)
    logger.info("  Device      : %s", args.device)
    logger.info("=" * 60)

    # ── Build model and data ──────────────────────────────────────────────────
    model = build_model(args.model, args.device)
    train_loader, val_loader = build_data_loaders(
        args.model, args.data_samples, args.client_id
    )

    # ── Build Flower client ───────────────────────────────────────────────────
    from client.fl_client import FedMedClient

    fl_client = FedMedClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        use_secagg=args.secagg,
        n_clients=args.total_clients,
    )

    logger.info(
        "[main_client] ECDSA public key (PEM):\n%s",
        fl_client.public_key_pem.decode(),
    )

    # ── mTLS channel credentials ─────────────────────────────────────────────
    if args.mtls:
        from security.grpc_tls import load_channel_credentials, is_cert_expired

        cert_dir = Path(args.cert_dir)
        client_cert_path = cert_dir / f"client_{args.client_id}.pem"

        if not client_cert_path.exists():
            logger.error(
                "Client cert not found: %s\n"
                "Run: python scripts/gen_certs.py --clients %d",
                client_cert_path, args.total_clients,
            )
            sys.exit(1)

        if is_cert_expired(client_cert_path.read_bytes()):
            logger.error(
                "Client certificate for ID %d has EXPIRED. Regenerate certs.",
                args.client_id,
            )
            sys.exit(1)

        channel_creds = load_channel_credentials(cert_dir, client_id=args.client_id)
        logger.info("[main_client] mTLS credentials loaded for client %d", args.client_id)
    else:
        channel_creds = None
        logger.warning(
            "[main_client] mTLS DISABLED — running in plaintext mode. "
            "Do NOT use in production!"
        )

    # ── Start Flower client ───────────────────────────────────────────────────
    import flwr as fl

    if channel_creds:
        fl.client.start_numpy_client(
            server_address=args.server,
            client=fl_client,
            # Flower >= 1.4 supports root_certificates for TLS:
            root_certificates=(Path(args.cert_dir) / "ca.pem").read_bytes(),
        )
    else:
        fl.client.start_numpy_client(
            server_address=args.server,
            client=fl_client,
        )

    logger.info("[main_client] Client %d finished.", args.client_id)


if __name__ == "__main__":
    main()
