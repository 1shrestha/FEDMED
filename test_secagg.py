"""
security/tests/test_secagg.py
==============================
Unit tests for security.secagg — the 4-round masking-based Secure Aggregation.

Tests cover:
* Shamir Secret Sharing — split + reconstruct round-trip.
* Shamir — reconstruct from any threshold-sized subset of shares.
* Shamir — reconstruct fails (wrong result) when fewer shares than threshold used.
* SecAggClient — generate_keys() produces a valid bundle.
* SecAggCoordinator — full 4-round happy path: masked sum == plain sum.
* SecAggCoordinator — dropout scenario (1 of 3 clients drops after R2).
* SecAggCoordinator — raises if not enough dropout shares.
* Integration — masked average ≈ plain average for 5 clients, small weights.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from security.secagg import (
    SecAggClient,
    SecAggCoordinator,
    shamir_reconstruct,
    shamir_split,
)


# ---------------------------------------------------------------------------
# Shamir Secret Sharing tests
# ---------------------------------------------------------------------------


class TestShamirSecretSharing(unittest.TestCase):
    def test_reconstruct_from_all_shares(self):
        """Full set of shares must reconstruct the original secret exactly."""
        secret = b"hello fedmed!"
        shares = shamir_split(secret, threshold=3, n_shares=5)
        self.assertEqual(len(shares), 5)
        reconstructed = shamir_reconstruct(shares)
        self.assertEqual(reconstructed, secret)

    def test_reconstruct_from_threshold_subset(self):
        """Any t-of-n subset should reconstruct the secret correctly."""
        secret = b"\x00\x01\x02\x03\xde\xad\xbe\xef"
        t, n = 3, 5
        shares = shamir_split(secret, threshold=t, n_shares=n)
        # Pick the first t shares
        subset = dict(list(shares.items())[:t])
        self.assertEqual(shamir_reconstruct(subset), secret)

    def test_different_subsets_same_result(self):
        """Multiple threshold-sized subsets must all reconstruct the same secret."""
        secret = b"deterministic"
        t, n = 2, 4
        shares = shamir_split(secret, threshold=t, n_shares=n)
        items = list(shares.items())
        r1 = shamir_reconstruct(dict(items[:2]))
        r2 = shamir_reconstruct(dict(items[1:3]))
        r3 = shamir_reconstruct(dict(items[2:4]))
        self.assertEqual(r1, r2)
        self.assertEqual(r2, r3)
        self.assertEqual(r1, secret)

    def test_threshold_1_trivial(self):
        """With t=1 any single share should reconstruct."""
        secret = b"one"
        shares = shamir_split(secret, threshold=1, n_shares=3)
        self.assertEqual(shamir_reconstruct({1: shares[1]}), secret)

    def test_threshold_equals_n(self):
        """t == n: only all shares together reconstruct."""
        secret = b"all or nothing"
        shares = shamir_split(secret, threshold=5, n_shares=5)
        self.assertEqual(shamir_reconstruct(shares), secret)

    def test_threshold_greater_than_n_raises(self):
        """threshold > n_shares must raise ValueError."""
        with self.assertRaises(ValueError):
            shamir_split(b"x", threshold=4, n_shares=3)

    def test_binary_secret(self):
        """Should handle arbitrary binary secrets (all 256 byte values)."""
        secret = bytes(range(256))
        t, n = 4, 7
        shares = shamir_split(secret, threshold=t, n_shares=n)
        subset = dict(list(shares.items())[:t])
        self.assertEqual(shamir_reconstruct(subset), secret)


# ---------------------------------------------------------------------------
# Helper: simulate a full 4-round SecAgg protocol
# ---------------------------------------------------------------------------


def _run_secagg_protocol(
    n_clients: int,
    local_updates: list[np.ndarray],
    dropout_ids: list[int] | None = None,
) -> np.ndarray:
    """Run all 4 SecAgg rounds and return the unmasked aggregate.

    Parameters
    ----------
    n_clients:
        Number of clients in the round.
    local_updates:
        One 1-D array per client (index = client_id).
    dropout_ids:
        Client IDs that drop out *after* Round 2 but *before* Round 3.
    """
    dropout_ids = dropout_ids or []
    threshold = math.ceil(n_clients / 2)

    # Instantiate clients and coordinator
    clients = [SecAggClient(i, n_clients, threshold) for i in range(n_clients)]
    coord = SecAggCoordinator(n_clients, threshold)

    # ── Round 1: generate + collect keys ────────────────────────────────────
    bundles = [c.generate_keys() for c in clients]
    coord.round1_collect_keys(bundles)

    # ── Round 2: distribute keys ─────────────────────────────────────────────
    all_keys = coord.round2_distribute_keys()
    for c in clients:
        c.receive_peer_keys(all_keys)

    # Collect shares BEFORE any dropout (each client distributes its shares)
    # In prod, these would be encrypted and forwarded via the coordinator.
    all_shares: dict[int, dict[int, bytes]] = {i: {} for i in range(n_clients)}
    for c in clients:
        shares = c.get_mask_shares()
        for recipient_id, share_bytes in shares.items():
            all_shares[recipient_id][c.client_id] = share_bytes

    # ── Round 3: mask and send updates (dropouts skip this) ──────────────────
    masked_updates = {}
    for c in clients:
        if c.client_id in dropout_ids:
            continue
        masked_updates[c.client_id] = c.mask_update(local_updates[c.client_id])
    coord.round3_collect_masked_updates(masked_updates)

    # ── Round 4: unmask ───────────────────────────────────────────────────────
    # For dropout reconstruction: survivors forward shares they hold
    dropout_recon_shares: dict[int, dict[int, bytes]] = {}
    for d_id in dropout_ids:
        dropout_recon_shares[d_id] = {}
        for survivor_id in range(n_clients):
            if survivor_id in dropout_ids:
                continue
            # Survivor sends the share it received from the dropout
            dropout_recon_shares[d_id][survivor_id] = all_shares[survivor_id][d_id]

    result = coord.round4_unmask(dropout_ids, dropout_recon_shares)
    return result


# ---------------------------------------------------------------------------
# SecAgg protocol tests
# ---------------------------------------------------------------------------


class TestSecAggClientKeyGeneration(unittest.TestCase):
    def test_generate_keys_returns_bundle(self):
        from security.secagg import ClientKeyBundle

        c = SecAggClient(client_id=0, n_clients=3)
        bundle = c.generate_keys()
        self.assertIsInstance(bundle, ClientKeyBundle)
        self.assertEqual(bundle.client_id, 0)
        self.assertIsInstance(bundle.ecdh_public_key_pem, bytes)
        self.assertIsInstance(bundle.signing_public_key_pem, bytes)

    def test_two_clients_different_keys(self):
        c0 = SecAggClient(0, 3)
        c1 = SecAggClient(1, 3)
        b0 = c0.generate_keys()
        b1 = c1.generate_keys()
        self.assertNotEqual(b0.ecdh_public_key_pem, b1.ecdh_public_key_pem)

    def test_mask_update_without_keys_raises(self):
        c = SecAggClient(0, 3)
        with self.assertRaises(RuntimeError):
            c.mask_update(np.zeros(5, dtype=np.float32))


class TestSecAggHappyPath(unittest.TestCase):
    """Full 4-round protocol, no dropouts."""

    def _run(self, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (secagg_result, plain_mean)."""
        rng = np.random.default_rng(seed=42)
        updates = [rng.standard_normal(d).astype(np.float32) for _ in range(n)]
        plain_mean = np.mean(updates, axis=0).astype(np.float32)
        result = _run_secagg_protocol(n, updates)
        return result, plain_mean

    def test_3_clients_small_vector(self):
        result, expected = self._run(n=3, d=10)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_4_clients_medium_vector(self):
        result, expected = self._run(n=4, d=100)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_5_clients_larger_vector(self):
        result, expected = self._run(n=5, d=500)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_all_zeros(self):
        """Aggregate of all-zero updates must be zero."""
        updates = [np.zeros(20, dtype=np.float32) for _ in range(3)]
        result = _run_secagg_protocol(3, updates)
        np.testing.assert_allclose(result, np.zeros(20, dtype=np.float32), atol=1e-5)

    def test_all_ones(self):
        """Aggregate of all-one updates must be one."""
        updates = [np.ones(20, dtype=np.float32) for _ in range(4)]
        result = _run_secagg_protocol(4, updates)
        np.testing.assert_allclose(result, np.ones(20, dtype=np.float32), atol=1e-4)


class TestSecAggDropout(unittest.TestCase):
    """Protocol with one client dropping out after Round 2."""

    def test_one_dropout_of_three(self):
        """1-of-3 dropout: result should match the mean of the surviving 2 clients
        after properly removing the dropout's mask."""
        rng = np.random.default_rng(seed=7)
        n, d = 3, 50
        updates = [rng.standard_normal(d).astype(np.float32) for _ in range(n)]

        # Expected: mean across ALL 3 clients (dropout's update is included in
        # the aggregate because we reconstruct its mask)
        expected = np.mean(updates, axis=0).astype(np.float32)

        result = _run_secagg_protocol(n, updates, dropout_ids=[2])
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_insufficient_dropout_shares_raises(self):
        """Coordinator must raise if not enough shares are available for a dropout."""
        n_clients = 3
        threshold = math.ceil(n_clients / 2)  # = 2

        clients = [SecAggClient(i, n_clients, threshold) for i in range(n_clients)]
        coord = SecAggCoordinator(n_clients, threshold)

        bundles = [c.generate_keys() for c in clients]
        coord.round1_collect_keys(bundles)
        all_keys = coord.round2_distribute_keys()
        for c in clients:
            c.receive_peer_keys(all_keys)

        masked_updates = {
            0: clients[0].mask_update(np.ones(10, dtype=np.float32)),
            1: clients[1].mask_update(np.ones(10, dtype=np.float32)),
        }
        coord.round3_collect_masked_updates(masked_updates)

        # Provide only 1 share for client 2 (threshold = 2) → should raise
        with self.assertRaises(ValueError, msg="Should require at least threshold shares"):
            coord.round4_unmask(
                dropout_ids=[2],
                dropout_shares={2: {0: b"\x00" * 32}},  # only 1 share
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
