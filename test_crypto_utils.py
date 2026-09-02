"""
security/tests/test_crypto_utils.py
=====================================
Unit tests for security.crypto_utils.

Tests cover:
* generate_ecdsa_keypair() — produces valid P-256 key pair.
* sign_update() + verify_update() — round-trip succeeds.
* verify_update() — tampered data returns False.
* verify_update() — wrong public key returns False.
* dh_exchange() — two parties derive the same shared secret.
* dh_exchange() — different key pairs produce different secrets.
* aes_gcm_encrypt() + aes_gcm_decrypt() — round-trip succeeds.
* aes_gcm_decrypt() — tampered ciphertext raises InvalidTag.
* aes_gcm_encrypt/decrypt() — wrong key raises InvalidTag.
* aes_gcm_encrypt() — invalid key length raises ValueError.
* prg_mask() — deterministic output for same seed/shape.
* prg_mask() — different seeds produce different output.
* Serialisation helpers — PEM round-trip for private + public keys.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from security.crypto_utils import (
    AES_GCM_KEY_BYTES,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    dh_exchange,
    generate_ecdsa_keypair,
    load_private_key,
    load_public_key,
    prg_mask,
    serialise_private_key,
    serialise_public_key,
    sign_update,
    verify_update,
)


class TestECDSAKeyGeneration(unittest.TestCase):
    def test_generates_valid_key_pair(self):
        """generate_ecdsa_keypair() should return a usable P-256 key pair."""
        from cryptography.hazmat.primitives.asymmetric import ec

        priv, pub = generate_ecdsa_keypair()
        self.assertIsInstance(priv, ec.EllipticCurvePrivateKey)
        self.assertIsInstance(pub, ec.EllipticCurvePublicKey)

    def test_each_call_generates_unique_key(self):
        """Two consecutive calls must produce different keys."""
        priv1, pub1 = generate_ecdsa_keypair()
        priv2, pub2 = generate_ecdsa_keypair()

        pub1_pem = serialise_public_key(pub1)
        pub2_pem = serialise_public_key(pub2)
        self.assertNotEqual(pub1_pem, pub2_pem)


class TestSignVerify(unittest.TestCase):
    def setUp(self):
        self.priv, self.pub = generate_ecdsa_keypair()
        self.data = b"hospital-A-gradient-update-round-7"
        self.sig = sign_update(self.priv, self.data)

    def test_valid_signature_verifies(self):
        """Correct key + correct data must verify."""
        self.assertTrue(verify_update(self.pub, self.data, self.sig))

    def test_tampered_data_fails(self):
        """Flipping a byte in data should break verification."""
        bad_data = self.data[:-1] + bytes([self.data[-1] ^ 0xFF])
        self.assertFalse(verify_update(self.pub, bad_data, self.sig))

    def test_tampered_signature_fails(self):
        """Flipping a byte in the signature should fail."""
        bad_sig = bytearray(self.sig)
        bad_sig[len(bad_sig) // 2] ^= 0xFF
        self.assertFalse(verify_update(self.pub, self.data, bytes(bad_sig)))

    def test_wrong_public_key_fails(self):
        """Signature verified with a different public key must return False."""
        _, other_pub = generate_ecdsa_keypair()
        self.assertFalse(verify_update(other_pub, self.data, self.sig))

    def test_empty_data_signs_and_verifies(self):
        """Edge case: signing empty bytes should work."""
        sig = sign_update(self.priv, b"")
        self.assertTrue(verify_update(self.pub, b"", sig))

    def test_large_payload(self):
        """Should handle large updates (1 MB)."""
        large = os.urandom(1024 * 1024)
        sig = sign_update(self.priv, large)
        self.assertTrue(verify_update(self.pub, large, sig))


class TestDHExchange(unittest.TestCase):
    def test_shared_secret_matches(self):
        """Both parties must derive the same shared secret."""
        priv_a, pub_a = generate_ecdsa_keypair()
        priv_b, pub_b = generate_ecdsa_keypair()

        secret_a = dh_exchange(priv_a, pub_b)
        secret_b = dh_exchange(priv_b, pub_a)
        self.assertEqual(secret_a, secret_b)

    def test_secret_is_32_bytes(self):
        """Derived secret must be exactly 32 bytes (AES-256 key)."""
        priv_a, pub_a = generate_ecdsa_keypair()
        priv_b, pub_b = generate_ecdsa_keypair()
        secret = dh_exchange(priv_a, pub_b)
        self.assertEqual(len(secret), 32)

    def test_different_pairs_give_different_secrets(self):
        """Independent key pairs should not collide."""
        priv_a, pub_a = generate_ecdsa_keypair()
        priv_b, pub_b = generate_ecdsa_keypair()
        priv_c, pub_c = generate_ecdsa_keypair()

        s1 = dh_exchange(priv_a, pub_b)
        s2 = dh_exchange(priv_a, pub_c)
        self.assertNotEqual(s1, s2)

    def test_custom_info_domain_separates(self):
        """Different *info* labels should produce different keys."""
        priv_a, pub_a = generate_ecdsa_keypair()
        priv_b, pub_b = generate_ecdsa_keypair()

        s1 = dh_exchange(priv_a, pub_b, info=b"context-1")
        s2 = dh_exchange(priv_a, pub_b, info=b"context-2")
        self.assertNotEqual(s1, s2)


class TestAESGCM(unittest.TestCase):
    def setUp(self):
        self.key = os.urandom(AES_GCM_KEY_BYTES)
        self.plaintext = b"sensitive audit metadata"

    def test_encrypt_decrypt_roundtrip(self):
        ct = aes_gcm_encrypt(self.key, self.plaintext)
        pt = aes_gcm_decrypt(self.key, ct)
        self.assertEqual(pt, self.plaintext)

    def test_each_encryption_unique_ciphertext(self):
        """Two encryptions of the same plaintext must produce different ciphertexts
        (different random nonces)."""
        ct1 = aes_gcm_encrypt(self.key, self.plaintext)
        ct2 = aes_gcm_encrypt(self.key, self.plaintext)
        self.assertNotEqual(ct1, ct2)

    def test_with_aad(self):
        """AAD is authenticated but not encrypted."""
        aad = b"round=7,hospital=A"
        ct = aes_gcm_encrypt(self.key, self.plaintext, aad=aad)
        pt = aes_gcm_decrypt(self.key, ct, aad=aad)
        self.assertEqual(pt, self.plaintext)

    def test_wrong_aad_raises(self):
        """Using wrong AAD during decryption must raise."""
        from cryptography.exceptions import InvalidTag

        aad = b"correct-aad"
        ct = aes_gcm_encrypt(self.key, self.plaintext, aad=aad)
        with self.assertRaises(InvalidTag):
            aes_gcm_decrypt(self.key, ct, aad=b"wrong-aad")

    def test_tampered_ciphertext_raises(self):
        """Flipping a byte in the ciphertext must raise InvalidTag."""
        from cryptography.exceptions import InvalidTag

        ct = bytearray(aes_gcm_encrypt(self.key, self.plaintext))
        ct[-1] ^= 0xFF
        with self.assertRaises(InvalidTag):
            aes_gcm_decrypt(self.key, bytes(ct))

    def test_wrong_key_raises(self):
        """Decrypting with a different key must raise InvalidTag."""
        from cryptography.exceptions import InvalidTag

        ct = aes_gcm_encrypt(self.key, self.plaintext)
        other_key = os.urandom(AES_GCM_KEY_BYTES)
        with self.assertRaises(InvalidTag):
            aes_gcm_decrypt(other_key, ct)

    def test_invalid_key_length_raises(self):
        """Passing a 16-byte key (AES-128) should raise ValueError."""
        bad_key = os.urandom(16)
        with self.assertRaises(ValueError):
            aes_gcm_encrypt(bad_key, self.plaintext)

    def test_truncated_ciphertext_raises(self):
        """A ciphertext shorter than nonce + tag should raise ValueError."""
        with self.assertRaises(ValueError):
            aes_gcm_decrypt(self.key, b"tooshort")

    def test_empty_plaintext(self):
        """AES-GCM must handle empty plaintext (only the GCM tag is stored)."""
        ct = aes_gcm_encrypt(self.key, b"")
        pt = aes_gcm_decrypt(self.key, ct)
        self.assertEqual(pt, b"")


class TestPRGMask(unittest.TestCase):
    def test_deterministic(self):
        """Same seed + shape must always produce the same output."""
        seed = os.urandom(32)
        shape = (100,)
        m1 = prg_mask(seed, shape)
        m2 = prg_mask(seed, shape)
        self.assertEqual(m1, m2)

    def test_correct_length(self):
        """Output length must match product(shape) * dtype_itemsize."""
        seed = os.urandom(32)
        shape = (10, 20)  # 200 elements
        mask = prg_mask(seed, shape, dtype_itemsize=4)
        self.assertEqual(len(mask), 200 * 4)

    def test_different_seeds_different_output(self):
        """Different seeds should produce different masks."""
        seed_a = b"\x00" * 32
        seed_b = b"\xFF" * 32
        shape = (256,)
        self.assertNotEqual(prg_mask(seed_a, shape), prg_mask(seed_b, shape))


class TestSerialisationHelpers(unittest.TestCase):
    def test_private_key_pem_roundtrip(self):
        """serialise_private_key() + load_private_key() must preserve the key."""
        priv, _ = generate_ecdsa_keypair()
        pem = serialise_private_key(priv)
        loaded = load_private_key(pem)
        # Compare by re-serialising
        self.assertEqual(pem, serialise_private_key(loaded))

    def test_private_key_password_protected(self):
        """Password-protected PEM must fail to load without the password."""
        priv, _ = generate_ecdsa_keypair()
        pwd = b"super-secret"
        pem = serialise_private_key(priv, password=pwd)
        # Should succeed with correct password
        load_private_key(pem, password=pwd)
        # Should fail without password
        with self.assertRaises(Exception):
            load_private_key(pem, password=None)

    def test_public_key_pem_roundtrip(self):
        """serialise_public_key() + load_public_key() must preserve the key."""
        _, pub = generate_ecdsa_keypair()
        pem = serialise_public_key(pub)
        loaded = load_public_key(pem)
        self.assertEqual(pem, serialise_public_key(loaded))


if __name__ == "__main__":
    unittest.main(verbosity=2)
