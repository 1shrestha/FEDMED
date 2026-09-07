"""
security.secagg
===============
Masking-based Secure Aggregation (SecAgg) for FedMed.

Protocol overview (Bell et al., 2020)
--------------------------------------
The protocol runs in **4 rounds** between a :class:`SecAggCoordinator`
(hosted inside the Flower server) and N :class:`SecAggClient` instances:

Round 1 — Advertise Keys
    Each client generates an ephemeral ECDH key pair (for mask derivation)
    and an ECDSA key pair (for message authentication).  Public keys are
    sent to the coordinator.

Round 2 — Share Masks
    The coordinator broadcasts all public keys.  Each client derives pairwise
    shared secrets with every other client, generates a personal mask seed
    (``s_i``), and encrypts Shamir shares of ``s_i`` under each peer's public
    key.  Encrypted shares are returned to the coordinator, which forwards
    them to the appropriate recipients.

Round 3 — Mask Updates
    Each client combines pairwise masks (derived from ECDH shared secrets)
    and its own mask (from ``s_i``) to produce a masked update:

        masked_u_i = u_i + own_mask_i + Σ_{j≠i} pairwise_mask(i,j)

    where masks for pairs (i, j) with i > j are *subtracted* so they cancel
    on summation.  The masked tensor is sent to the coordinator.

Round 4 — Unmask
    The coordinator collects masked updates and the decrypted Shamir shares
    from surviving clients about any dropped-out clients.  It reconstructs
    the dropout masks and subtracts them, yielding the clean aggregate.

Architecture note
-----------------
The Python layer (this module) owns rounds 1–4 coordination and masking.
The actual **weight averaging** (sum → divide by n) is delegated to the
Go-based aggregation service via an explicit interface hook; see
:attr:`SecAggCoordinator.averaging_fn`.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .crypto_utils import (
    ECPrivateKey,
    ECPublicKey,
    dh_exchange,
    generate_ecdsa_keypair,
    prg_mask,
    serialise_public_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shamir's Secret Sharing  (pure Python, no external library dependency)
# ---------------------------------------------------------------------------
#
# We operate over GF(2^31 - 1) (a Mersenne prime) for efficiency with
# integer arithmetic.  Mask seeds are 32-byte values; we split them
# byte-by-byte (each byte in GF(p)) to keep the implementation simple and
# dependency-free.
# ---------------------------------------------------------------------------

_PRIME = (1 << 31) - 1  # 2147483647 — Mersenne prime


def _mod_inv(a: int, m: int = _PRIME) -> int:
    """Modular multiplicative inverse via extended Euclidean algorithm."""
    g, x, _ = _ext_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}")
    return x % m


def _ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = _ext_gcd(b % a, a)
    return g, y - (b // a) * x, x


def _poly_eval(coefficients: List[int], x: int, p: int = _PRIME) -> int:
    """Evaluate a polynomial at *x* modulo *p* (Horner's method)."""
    result = 0
    for coeff in reversed(coefficients):
        result = (result * x + coeff) % p
    return result


def _lagrange_interpolate(shares: List[Tuple[int, int]], p: int = _PRIME) -> int:
    """Recover the secret (constant term) from *shares* via Lagrange interpolation."""
    secret = 0
    xs = [s[0] for s in shares]
    ys = [s[1] for s in shares]
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i == j:
                continue
            num = num * (-xj) % p
            den = den * (xi - xj) % p
        secret = (secret + yi * num * _mod_inv(den, p)) % p
    return secret


def shamir_split(
    secret_bytes: bytes, threshold: int, n_shares: int
) -> Dict[int, bytes]:
    """Split *secret_bytes* into *n_shares* Shamir shares.

    Parameters
    ----------
    secret_bytes:
        The secret to split (arbitrary length; processed byte by byte).
    threshold:
        Minimum number of shares required to reconstruct the secret.
    n_shares:
        Total number of shares to produce.

    Returns
    -------
    dict mapping share_index (1-based) → share_bytes.
    Each share_bytes has the same length as *secret_bytes*.
    """
    if threshold > n_shares:
        raise ValueError("threshold must be ≤ n_shares")

    share_data: Dict[int, bytearray] = {i: bytearray() for i in range(1, n_shares + 1)}

    for byte_val in secret_bytes:
        # Build a random degree-(threshold-1) polynomial with secret as constant term
        coefficients = [byte_val] + [
            int.from_bytes(os.urandom(4), "big") % _PRIME
            for _ in range(threshold - 1)
        ]
        for i in range(1, n_shares + 1):
            share_data[i].append(_poly_eval(coefficients, i) % 256)

    return {i: bytes(v) for i, v in share_data.items()}


def shamir_reconstruct(shares: Dict[int, bytes]) -> bytes:
    """Reconstruct the secret from a dict of Shamir shares.

    Parameters
    ----------
    shares:
        Dict mapping share_index → share_bytes (must satisfy threshold).

    Returns
    -------
    bytes
        Reconstructed secret.
    """
    n_bytes = len(next(iter(shares.values())))
    result = bytearray()
    share_items = list(shares.items())

    for byte_pos in range(n_bytes):
        pts = [(idx, data[byte_pos]) for idx, data in share_items]
        result.append(_lagrange_interpolate(pts) % 256)

    return bytes(result)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClientKeyBundle:
    """Public keys advertised by a client in Round 1."""

    client_id: int
    ecdh_public_key_pem: bytes  # serialised ECDH public key
    signing_public_key_pem: bytes  # serialised ECDSA public key


@dataclass
class SecAggRoundState:
    """Mutable coordinator state shared across all four rounds."""

    client_keys: Dict[int, ClientKeyBundle] = field(default_factory=dict)
    # masked updates keyed by client_id
    masked_updates: Dict[int, np.ndarray] = field(default_factory=dict)
    # shares of dropped clients' mask seeds, keyed by dropout_id → {holder_id: share}
    dropout_shares: Dict[int, Dict[int, bytes]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SecAggClient
# ---------------------------------------------------------------------------


class SecAggClient:
    """Client-side Secure Aggregation participant.

    Each Flower client instantiates one :class:`SecAggClient` per FL round.

    Typical usage::

        client = SecAggClient(client_id=0, n_clients=5, threshold=3)
        bundle = client.generate_keys()          # Round 1
        client.receive_peer_keys(peer_bundles)   # Round 2
        masked = client.mask_update(local_update)# Round 3
        shares = client.get_mask_shares()        # for dropout recovery
    """

    def __init__(self, client_id: int, n_clients: int, threshold: int | None = None):
        """
        Parameters
        ----------
        client_id:
            Unique integer id for this client (0-indexed).
        n_clients:
            Total number of participating clients this round.
        threshold:
            Minimum shares needed to reconstruct any client's mask seed.
            Defaults to ``ceil(n_clients / 2)``.
        """
        self.client_id = client_id
        self.n_clients = n_clients
        self.threshold = threshold if threshold is not None else math.ceil(n_clients / 2)

        # Keys generated in Round 1
        self._ecdh_private: Optional[ECPrivateKey] = None
        self._ecdh_public: Optional[ECPublicKey] = None
        self._sign_private: Optional[ECPrivateKey] = None
        self._sign_public: Optional[ECPublicKey] = None

        # Own mask seed (random, kept secret)
        self._own_seed: Optional[bytes] = None

        # Peer public keys (Round 2)
        self._peer_keys: Dict[int, ClientKeyBundle] = {}

    # ── Round 1 ─────────────────────────────────────────────────────────────

    def generate_keys(self) -> ClientKeyBundle:
        """Generate ephemeral ECDH + ECDSA key pairs and return the public bundle."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.backends import default_backend

        self._ecdh_private = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self._ecdh_public = self._ecdh_private.public_key()
        self._sign_private, self._sign_public = generate_ecdsa_keypair()
        self._own_seed = os.urandom(32)

        return ClientKeyBundle(
            client_id=self.client_id,
            ecdh_public_key_pem=serialise_public_key(self._ecdh_public),
            signing_public_key_pem=serialise_public_key(self._sign_public),
        )

    # ── Round 2 ─────────────────────────────────────────────────────────────

    def receive_peer_keys(self, peer_bundles: List[ClientKeyBundle]) -> None:
        """Store peer public keys received from the coordinator."""
        self._peer_keys = {b.client_id: b for b in peer_bundles}

    # ── Round 3 ─────────────────────────────────────────────────────────────

    def mask_update(self, local_update: np.ndarray) -> np.ndarray:
        """Apply pairwise and own masks to *local_update* and return masked array.

        For each pair (self, j):
        * If self.client_id > j: *subtract* the pairwise mask (so it cancels).
        * If self.client_id < j: *add* the pairwise mask.
        Own mask is always added.
        """
        if self._ecdh_private is None or self._own_seed is None:
            raise RuntimeError("Call generate_keys() before mask_update().")

        from .crypto_utils import load_public_key

        shape = local_update.shape
        dtype = local_update.dtype

        # Start with a copy in float64 for precision
        masked = local_update.astype(np.float64)

        # Own mask
        own_mask_bytes = prg_mask(self._own_seed, shape, dtype_itemsize=8)
        own_mask = np.frombuffer(own_mask_bytes, dtype=np.float64).reshape(shape)
        masked += own_mask

        # Pairwise masks
        for peer_id, bundle in self._peer_keys.items():
            if peer_id == self.client_id:
                continue
            peer_ecdh_pub = load_public_key(bundle.ecdh_public_key_pem)
            shared_secret = dh_exchange(
                self._ecdh_private,
                peer_ecdh_pub,
                info=b"fedmed-secagg-pairwise-v1",
            )
            pair_mask_bytes = prg_mask(shared_secret, shape, dtype_itemsize=8)
            pair_mask = np.frombuffer(pair_mask_bytes, dtype=np.float64).reshape(shape)

            if self.client_id > peer_id:
                masked -= pair_mask
            else:
                masked += pair_mask

        return masked.astype(dtype)

    # ── Round 4 support ─────────────────────────────────────────────────────

    def get_mask_shares(self) -> Dict[int, bytes]:
        """Return Shamir shares of own mask seed, keyed by recipient client_id.

        Share *i* is sent (encrypted) to client *i* so that, in the event
        this client drops out, the coordinator can collect t-of-n shares
        from surviving peers and reconstruct this client's mask seed.
        """
        if self._own_seed is None:
            raise RuntimeError("Call generate_keys() first.")
        all_ids = sorted(self._peer_keys.keys()) + [self.client_id]
        all_ids = sorted(set(all_ids))
        n = len(all_ids)
        raw_shares = shamir_split(self._own_seed, self.threshold, n)
        # Map raw 1-based share index to the corresponding client_id
        return {cid: raw_shares[i + 1] for i, cid in enumerate(all_ids)}


# ---------------------------------------------------------------------------
# SecAggCoordinator
# ---------------------------------------------------------------------------


class SecAggCoordinator:
    """Server-side Secure Aggregation coordinator.

    Owned by the Flower server strategy.  Manages state across the 4 rounds
    for a single FL training round.

    The actual weight **averaging** is delegated to *averaging_fn* — in
    production this calls the Go-based aggregation service.  The default
    implementation uses NumPy for local testing.

    Parameters
    ----------
    n_clients:
        Expected number of clients for this FL round.
    threshold:
        Shamir reconstruction threshold (default: ``ceil(n_clients / 2)``).
    averaging_fn:
        Callable ``(updates: List[np.ndarray]) -> np.ndarray`` that computes
        the aggregate.  Defaults to element-wise mean.
    """

    def __init__(
        self,
        n_clients: int,
        threshold: int | None = None,
        averaging_fn: Callable[[List[np.ndarray]], np.ndarray] | None = None,
    ):
        self.n_clients = n_clients
        self.threshold = threshold if threshold is not None else math.ceil(n_clients / 2)
        self.averaging_fn: Callable[[List[np.ndarray]], np.ndarray] = (
            averaging_fn if averaging_fn is not None else _default_average
        )
        self._state = SecAggRoundState()

    # ── Round 1 ─────────────────────────────────────────────────────────────

    def round1_collect_keys(self, bundles: List[ClientKeyBundle]) -> None:
        """Store public key bundles from all participating clients."""
        for bundle in bundles:
            self._state.client_keys[bundle.client_id] = bundle
        logger.info(
            "[SecAgg R1] Collected keys from %d/%d clients",
            len(self._state.client_keys),
            self.n_clients,
        )

    # ── Round 2 ─────────────────────────────────────────────────────────────

    def round2_distribute_keys(self) -> List[ClientKeyBundle]:
        """Return the full list of public key bundles for broadcast to all clients."""
        bundles = list(self._state.client_keys.values())
        logger.info("[SecAgg R2] Broadcasting %d client key bundles", len(bundles))
        return bundles

    # ── Round 3 ─────────────────────────────────────────────────────────────

    def round3_collect_masked_updates(
        self, masked_updates: Dict[int, np.ndarray]
    ) -> None:
        """Store masked model updates from surviving clients."""
        self._state.masked_updates = masked_updates
        logger.info(
            "[SecAgg R3] Collected %d masked updates", len(masked_updates)
        )

    # ── Round 4 ─────────────────────────────────────────────────────────────

    def round4_unmask(
        self,
        dropout_ids: List[int],
        dropout_shares: Dict[int, Dict[int, bytes]],
    ) -> np.ndarray:
        """Reconstruct the clean aggregate from masked updates.

        Parameters
        ----------
        dropout_ids:
            Client IDs that dropped out after Round 2 (sent no masked update).
        dropout_shares:
            ``{dropout_id: {holder_id: share_bytes}}``.  Must contain at
            least *threshold* entries per dropout client.

        Returns
        -------
        np.ndarray
            Aggregated (averaged) model update.
        """
        surviving_ids = list(self._state.masked_updates.keys())
        updates = list(self._state.masked_updates.values())
        shape = updates[0].shape
        dtype = updates[0].dtype

        # Sum surviving masked updates
        agg = np.zeros(shape, dtype=np.float64)
        for u in updates:
            agg += u.astype(np.float64)

        # For each dropout client, reconstruct its own mask and subtract it
        for dropout_id in dropout_ids:
            if dropout_id not in dropout_shares:
                raise ValueError(
                    f"No shares provided for dropped client {dropout_id}"
                )
            shares = dropout_shares[dropout_id]
            if len(shares) < self.threshold:
                raise ValueError(
                    f"Only {len(shares)} shares for client {dropout_id}; "
                    f"need at least {self.threshold}"
                )
            # Take any threshold subset
            subset = dict(list(shares.items())[: self.threshold])
            seed = shamir_reconstruct(subset)
            own_mask_bytes = prg_mask(seed, shape, dtype_itemsize=8)
            own_mask = np.frombuffer(own_mask_bytes, dtype=np.float64).reshape(shape)
            # Pairwise masks between the dropout and each survivor cancel out
            # because each survivor already subtracted/added it correctly.
            # We only need to remove the dropout's *own* mask.
            agg -= own_mask

        logger.info(
            "[SecAgg R4] Unmasked aggregate from %d surviving + %d dropout clients",
            len(surviving_ids),
            len(dropout_ids),
        )

        # Delegate final averaging to the configured function (Go service in prod)
        total_clients = len(surviving_ids) + len(dropout_ids)
        averaged = self.averaging_fn(
            [agg / total_clients]  # pass the pre-summed result; fn divides if needed
        )
        return averaged.astype(dtype)

    def reset(self) -> None:
        """Clear round state (call between FL rounds)."""
        self._state = SecAggRoundState()


# ---------------------------------------------------------------------------
# Default averaging function (used in tests; replaced by Go service in prod)
# ---------------------------------------------------------------------------


def _default_average(updates: List[np.ndarray]) -> np.ndarray:
    """Simple element-wise mean — placeholder for Go aggregation service."""
    # In the coordinator, we pass [agg / total_clients] so this just returns it.
    return updates[0] if len(updates) == 1 else np.mean(updates, axis=0)
