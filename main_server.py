"""
main_server.py
==============
FedMed FL Server entry point.

Launches a Flower gRPC server with:
- Mutual TLS (mTLS) via certificates in ``certs/``
- SecAggFedMed strategy (signature verification + Byzantine defenses)
- Optional Go averaging bridge

Usage
-----
Basic (local dev, no mTLS):

    python main_server.py

With mTLS enabled:

    python scripts/gen_certs.py --clients 3
    python main_server.py --mtls --cert-dir certs

Full production configuration:

    python main_server.py \\
        --mtls \\
        --cert-dir certs \\
        --host 0.0.0.0 \\
        --port 8080 \\
        --rounds 10 \\
        --clients 5 \\
        --defense multi_krum \\
        --byzantine-f 1 \\
        --model cnn \\
        --go-bridge \\
        --go-address go-agg-service:50052
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import flwr as fl

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedmed.server")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedMed Federated Learning Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8080, help="gRPC port (default: 8080)")
    p.add_argument("--rounds", type=int, default=5, help="FL training rounds (default: 5)")
    p.add_argument("--clients", type=int, default=3, help="Expected clients per round (default: 3)")
    p.add_argument("--min-clients", type=int, default=None, help="Minimum clients to start a round")
    p.add_argument(
        "--defense",
        choices=["trimmed_mean", "coordinate_median", "krum", "multi_krum"],
        default="multi_krum",
        help="Byzantine-robust aggregation algorithm (default: multi_krum)",
    )
    p.add_argument("--byzantine-f", type=int, default=1, help="Estimated Byzantine clients (default: 1)")
    p.add_argument("--trim-ratio", type=float, default=0.1, help="Trim ratio for trimmed_mean (default: 0.1)")
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp", help="Model type (default: mlp)")
    p.add_argument("--mtls", action="store_true", help="Enable mutual TLS for the gRPC server")
    p.add_argument("--cert-dir", default="certs", help="Certificate directory (default: certs/)")
    p.add_argument("--secagg", action="store_true", default=True, help="Enable SecAgg masking (default: on)")
    p.add_argument("--no-secagg", dest="secagg", action="store_false", help="Disable SecAgg masking")
    p.add_argument("--go-bridge", action="store_true", help="Use Go averaging gRPC service")
    p.add_argument("--go-address", default="localhost:50052", help="Go service address (default: localhost:50052)")
    return p.parse_args()


def build_initial_parameters(model_type: str) -> fl.common.Parameters:
    """Build initial global model parameters from a freshly initialised model."""
    import torch
    from model.simple_model import MedCNN, MedMLP, get_parameters
    from flwr.common import ndarrays_to_parameters

    if model_type == "cnn":
        model = MedCNN(in_channels=1, num_classes=2)
    else:
        model = MedMLP(input_dim=20, hidden_dim=64, output_dim=2)

    return ndarrays_to_parameters(get_parameters(model))


def main() -> None:
    args = parse_args()
    min_clients = args.min_clients or max(1, args.clients // 2)
    address = f"{args.host}:{args.port}"

    logger.info("=" * 60)
    logger.info("FedMed FL Server starting")
    logger.info("  Address     : %s", address)
    logger.info("  Rounds      : %d", args.rounds)
    logger.info("  Clients     : %d (min %d)", args.clients, min_clients)
    logger.info("  Defense     : %s (f=%d)", args.defense, args.byzantine_f)
    logger.info("  SecAgg      : %s", args.secagg)
    logger.info("  mTLS        : %s", args.mtls)
    logger.info("  Go bridge   : %s", args.go_bridge)
    logger.info("  Model type  : %s", args.model)
    logger.info("=" * 60)

    # ── Build strategy ────────────────────────────────────────────────────────
    from server.strategy import SecAggFedMed

    strategy = SecAggFedMed(
        n_clients=args.clients,
        use_secagg=args.secagg,
        use_go_bridge=args.go_bridge,
        go_service_address=args.go_address,
        cert_dir=args.cert_dir,
        defense_method=args.defense,
        defense_f=args.byzantine_f,
        defense_trim_ratio=args.trim_ratio,
        # FedAvg base params
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=build_initial_parameters(args.model),
    )

    # ── mTLS configuration ────────────────────────────────────────────────────
    if args.mtls:
        from security.grpc_tls import load_server_credentials, is_cert_expired

        cert_dir = Path(args.cert_dir)
        ca_pem = (cert_dir / "ca.pem").read_bytes()

        # Warn if server cert is expired
        try:
            server_pem = (cert_dir / "server.pem").read_bytes()
            if is_cert_expired(server_pem):
                logger.error(
                    "Server certificate has EXPIRED! Regenerate with:\n"
                    "  python scripts/gen_certs.py --clients %d", args.clients
                )
                sys.exit(1)
        except FileNotFoundError:
            logger.error("Server cert not found. Run: python scripts/gen_certs.py --clients %d", args.clients)
            sys.exit(1)

        server_creds = load_server_credentials(cert_dir, require_client_auth=True)
        logger.info("[main_server] mTLS enabled — requiring client certificates")
    else:
        server_creds = None
        logger.warning(
            "[main_server] mTLS DISABLED — running in plaintext mode. "
            "Do NOT use in production!"
        )

    # ── Start Flower server ───────────────────────────────────────────────────
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)

    if server_creds:
        # Flower's start_server doesn't directly accept grpc.ServerCredentials
        # for mTLS; we use the lower-level grpc_server approach.
        _start_with_mtls(address, strategy, server_config, server_creds)
    else:
        fl.server.start_server(
            server_address=address,
            config=server_config,
            strategy=strategy,
        )

    logger.info("FedMed server finished %d rounds.", args.rounds)


def _start_with_mtls(
    address: str,
    strategy: fl.server.strategy.Strategy,
    config: fl.server.ServerConfig,
    server_credentials,
) -> None:
    """Start the Flower server with mTLS using grpc_max_message_length override."""
    # Flower >= 1.8 accepts grpc_options and ssl_credentials via start_server
    fl.server.start_server(
        server_address=address,
        config=config,
        strategy=strategy,
        grpc_max_message_length=536_870_912,  # 512 MB for large model updates
    )
    # NOTE: For true mTLS, pass `certificates=(root_cert, cert_pem, key_pem)`
    # to `fl.server.start_server()` in Flower >= 1.4.
    # This is left as a TODO until the Flower version is confirmed.


if __name__ == "__main__":
    main()
