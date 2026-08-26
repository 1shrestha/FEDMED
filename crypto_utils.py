"""
security.crypto_utils
=====================
Essential cryptographic primitives for FedMed.

Algorithms
----------
* **ECDSA (NIST P-256)** — digital signatures for model updates.
* **ECDH (NIST P-256)** — ephemeral Diffie-Hellman key exchange used by
  the Secure Aggregation protocol to derive pairwise mask seeds.
* **AES-256-GCM** — authenticated symmetric encryption for metadata /
  audit payloads.

All key objects are from the ``cryptography`` library (>=42).  Private keys
are **never** serialised by default; callers that need the persistence should
use the provided ``serialise_*`` helpers and store keys securely (HSM / Vault
in production; file system for local dev only).
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ECPrivateKey = ec.EllipticCurvePrivateKey
ECPublicKey = ec.EllipticCurvePublicKey

AES_GCM_KEY_BYTES = 32   # AES-256
AES_GCM_NONCE_BYTES = 12  # 96-bit nonce (GCM standard)

# ---------------------------------------------------------------------------
# ECDSA — Key generation
# ---------------------------------------------------------------------------


def generate_ecdsa_keypair() -> Tuple[ECPrivateKey, ECPublicKey]:
    """Generate an ECDSA key pair on NIST P-256.

    Returns
    -------
    (private_key, public_key)
        Both are ``cryptography`` key objects.  Public keys are safe to
        transmit; private keys must be kept secret.
    """
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    return private_key, private_key.public_key()


# ---------------------------------------------------------------------------
# ECDSA — Sign / Verify
# ---------------------------------------------------------------------------


def sign_update(private_key: ECPrivateKey, data: bytes) -> bytes:
    """Sign *data* with *private_key* using ECDSA-SHA256.

    Parameters
    ----------
    private_key:
        ECDSA private key (NIST P-256).
    data:
        Arbitrary bytes — typically the serialised model update tensor.

    Returns
    -------
    bytes
        Raw DER-encoded signature (typically 70–72 bytes).
    """
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return signature


def verify_update(public_key: ECPublicKey, data: bytes, signature: bytes) -> bool:
    """Verify an ECDSA-SHA256 *signature* over *data* using *public_key*.

    Returns
    -------
    bool
        *True* if the signature is valid, *False* on any verification error
        (wrong key, tampered data, malformed signature, etc.).
    """
    from cryptography.exceptions import InvalidSignature

    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# ECDH — Key exchange
# ---------------------------------------------------------------------------


def dh_exchange(
    private_key: ECPrivateKey,
    peer_public_key: ECPublicKey,
    *,
    info: bytes = b"fedmed-secagg-v1",
) -> bytes:
    """Derive a 32-byte shared secret via ECDH + HKDF-SHA256.

    The raw ECDH shared secret is passed through HKDF to produce a
    uniformly distributed key suitable for seeding a PRG or deriving AES
    keys.  The *info* parameter domain-separates usages; override it if you
    need separate keys for different sub-protocols.

    Parameters
    ----------
    private_key:
        Own ECDH private key (P-256).
    peer_public_key:
        Peer's ECDH public key (P-256).
    info:
        HKDF context / domain separator.

    Returns
    -------
    bytes
        32-byte shared secret.
    """
    shared_secret_raw = private_key.exchange(ec.ECDH(), peer_public_key)
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_GCM_KEY_BYTES,
        salt=None,
        info=info,
        backend=default_backend(),
    ).derive(shared_secret_raw)
    return derived


# ---------------------------------------------------------------------------
# AES-256-GCM — Authenticated Symmetric Encryption
# ---------------------------------------------------------------------------


def aes_gcm_encrypt(key: bytes, plaintext: bytes, *, aad: bytes = b"") -> bytes:
    """Encrypt *plaintext* with AES-256-GCM.

    Parameters
    ----------
    key:
        32-byte AES key.
    plaintext:
        Data to encrypt.
    aad:
        Additional authenticated data (not encrypted, but authenticated).

    Returns
    -------
    bytes
        ``nonce (12 bytes) || ciphertext+tag`` — the nonce is prepended so
        the ciphertext is self-contained.

    Raises
    ------
    ValueError
        If *key* is not exactly 32 bytes.
    """
    if len(key) != AES_GCM_KEY_BYTES:
        raise ValueError(f"AES-GCM key must be {AES_GCM_KEY_BYTES} bytes, got {len(key)}")
    nonce = os.urandom(AES_GCM_NONCE_BYTES)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad or None)
    return nonce + ciphertext


def aes_gcm_decrypt(key: bytes, ciphertext: bytes, *, aad: bytes = b"") -> bytes:
    """Decrypt and authenticate a payload produced by :func:`aes_gcm_encrypt`.

    Parameters
    ----------
    key:
        32-byte AES key.
    ciphertext:
        ``nonce (12 bytes) || ciphertext+tag`` as returned by encrypt.
    aad:
        Must match the *aad* used during encryption.

    Returns
    -------
    bytes
        Decrypted plaintext.

    Raises
    ------
    ValueError
        On key length mismatch or truncated payload.
    cryptography.exceptions.InvalidTag
        If authentication fails (wrong key, tampered data, wrong aad).
    """
    if len(key) != AES_GCM_KEY_BYTES:
        raise ValueError(f"AES-GCM key must be {AES_GCM_KEY_BYTES} bytes, got {len(key)}")
    if len(ciphertext) < AES_GCM_NONCE_BYTES + 16:  # 16 = GCM tag length
        raise ValueError("Ciphertext too short — likely truncated or corrupted.")
    nonce = ciphertext[:AES_GCM_NONCE_BYTES]
    body = ciphertext[AES_GCM_NONCE_BYTES:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, body, aad or None)


# ---------------------------------------------------------------------------
# Serialisation helpers (PEM)
# ---------------------------------------------------------------------------


def serialise_private_key(private_key: ECPrivateKey, password: bytes | None = None) -> bytes:
    """Serialise *private_key* to PEM (optionally password-protected)."""
    enc = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=enc,
    )


def serialise_public_key(public_key: ECPublicKey) -> bytes:
    """Serialise *public_key* to uncompressed PEM (SubjectPublicKeyInfo)."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def load_private_key(pem: bytes, password: bytes | None = None) -> ECPrivateKey:
    """Load an ECDSA/ECDH private key from PEM bytes."""
    return serialization.load_pem_private_key(pem, password=password, backend=default_backend())  # type: ignore[return-value]


def load_public_key(pem: bytes) -> ECPublicKey:
    """Load an ECDSA/ECDH public key from PEM bytes."""
    return serialization.load_pem_public_key(pem, backend=default_backend())  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PRG helper — deterministic pseudo-random mask from a seed
# ---------------------------------------------------------------------------


def prg_mask(seed: bytes, shape: tuple[int, ...], dtype_itemsize: int = 4) -> bytes:
    """Generate a deterministic pseudo-random byte string from *seed*.

    Uses repeated SHA-256 in counter mode.  Output length is
    ``product(shape) * dtype_itemsize`` bytes so it can be interpreted as a
    NumPy / PyTorch tensor of the given *shape* and data type.

    Parameters
    ----------
    seed:
        32-byte PRG seed (e.g. derived via :func:`dh_exchange`).
    shape:
        Tuple of dimension sizes of the target tensor.
    dtype_itemsize:
        Bytes per element (4 for float32, 8 for float64).

    Returns
    -------
    bytes
        Pseudo-random bytes of length ``product(shape) * dtype_itemsize``.
    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    total_bytes = total_elements * dtype_itemsize

    output = bytearray()
    counter = 0
    while len(output) < total_bytes:
        h = hashlib.sha256(seed + struct.pack(">Q", counter)).digest()
        output.extend(h)
        counter += 1
    return bytes(output[:total_bytes])
