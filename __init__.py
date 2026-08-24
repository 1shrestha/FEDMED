"""
security — FedMed cryptography and secure aggregation package.

Public surface
--------------
grpc_tls     : mTLS credential helpers for gRPC channels / servers.
crypto_utils : ECDSA key-pair generation, sign/verify, DH key exchange,
               AES-GCM symmetric encryption.
secagg       : Masking-based Secure Aggregation (SecAggCoordinator +
               SecAggClient) implementing the 4-round protocol.
defenses     : Byzantine-robust aggregation rules: Trimmed Mean,
               Coordinate-wise Median, Krum, Multi-Krum.
"""

from .grpc_tls import load_server_credentials, load_channel_credentials
from .crypto_utils import (
    generate_ecdsa_keypair,
    sign_update,
    verify_update,
    dh_exchange,
    aes_gcm_encrypt,
    aes_gcm_decrypt,
)
from .secagg import SecAggCoordinator, SecAggClient
from .defenses import trimmed_mean, coordinate_median, krum, multi_krum

__all__ = [
    # grpc_tls
    "load_server_credentials",
    "load_channel_credentials",
    # crypto_utils
    "generate_ecdsa_keypair",
    "sign_update",
    "verify_update",
    "dh_exchange",
    "aes_gcm_encrypt",
    "aes_gcm_decrypt",
    # secagg
    "SecAggCoordinator",
    "SecAggClient",
    # defenses
    "trimmed_mean",
    "coordinate_median",
    "krum",
    "multi_krum",
]
