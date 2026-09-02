import unittest
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec
from security.crypto_utils import (
    generate_signing_key_pair,
    serialize_public_key,
    deserialize_public_key,
    sign_data,
    verify_signature,
    generate_dh_key_pair,
    compute_shared_secret,
    encrypt_message,
    decrypt_message,
    split_secret,
    reconstruct_secret
)
from security.grpc_tls import generate_self_signed_certificates
from security.defenses import coordinate_wise_median, trimmed_mean, krum, multi_krum


class TestCryptoUtils(unittest.TestCase):
    
    def test_signatures(self):
        # 1. Key generation
        priv, pub = generate_signing_key_pair()
        self.assertIsInstance(priv, ec.EllipticCurvePrivateKey)
        
        # Serialization & Deserialization
        serialized = serialize_public_key(pub)
        deserialized = deserialize_public_key(serialized)
        
        # 2. Signing
        data = b"Patient scan model weights data parameters"
        sig = sign_data(priv, data)
        
        # 3. Verification
        self.assertTrue(verify_signature(pub, data, sig))
        self.assertTrue(verify_signature(deserialized, data, sig))
        
        # 4. Tampering detection
        self.assertFalse(verify_signature(pub, data + b"extra_byte", sig))

    def test_key_exchange(self):
        priv_a, pub_a = generate_dh_key_pair()
        priv_b, pub_b = generate_dh_key_pair()
        
        secret_ab = compute_shared_secret(priv_a, pub_b)
        secret_ba = compute_shared_secret(priv_b, pub_a)
        
        self.assertEqual(secret_ab, secret_ba)
        self.assertEqual(len(secret_ab), 32)

    def test_encryption(self):
        key = np.random.bytes(32)
        plaintext = b"Highly sensitive medical records placeholder"
        
        ciphertext, nonce = encrypt_message(key, plaintext)
        decrypted = decrypt_message(key, ciphertext, nonce)
        
        self.assertEqual(plaintext, decrypted)

    def test_shamir_secret_sharing(self):
        secret = b"my_super_secret_dh_pairwise_key"  # 31 bytes
        threshold = 3
        total_shares = 5
        
        shares = split_secret(secret, threshold, total_shares)
        self.assertEqual(len(shares), total_shares)
        
        # Reconstruct with threshold shares
        reconstructed = reconstruct_secret(shares[:threshold])
        # Strip padding if any (reconstruct_secret returns 32 bytes)
        self.assertEqual(reconstructed[-len(secret):], secret)
        
        # Reconstruct with different subset of threshold shares
        subset = [shares[0], shares[2], shares[4]]
        reconstructed_sub = reconstruct_secret(subset)
        self.assertEqual(reconstructed_sub[-len(secret):], secret)


class TestGrpcTls(unittest.TestCase):
    
    def test_certificate_generation(self):
        certs = generate_self_signed_certificates()
        self.assertIn("ca_cert", certs)
        self.assertIn("server_key", certs)
        self.assertIn("server_cert", certs)
        self.assertIn("client_key", certs)
        self.assertIn("client_cert", certs)
        
        self.assertTrue(certs["ca_cert"].startswith(b"-----BEGIN CERTIFICATE-----"))
        self.assertTrue(certs["server_key"].startswith(b"-----BEGIN PRIVATE KEY-----"))


class TestDefenses(unittest.TestCase):
    
    def setUp(self):
        # Generate some mock model weight arrays
        self.updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 2.9]),
            np.array([0.9, 1.9, 3.1]),
            np.array([10.0, -10.0, 50.0]) # Malicious outlier
        ]
        
    def test_coordinate_wise_median(self):
        result = coordinate_wise_median(self.updates)
        # Median of [1.0, 1.1, 0.9, 10.0] -> [1.0] or [1.05] depending on even/odd
        # For length 4, median is average of middle 2: 1.0 and 1.1 -> 1.05
        np.testing.assert_allclose(result, np.array([1.05, 2.05, 3.0]))

    def test_trimmed_mean(self):
        # Trim beta=0.25 (1 out of 4 updates from each end sorted)
        # Sorted values:
        # col 1: [0.9, 1.0, 1.1, 10.0] -> trimmed: [1.0, 1.1] -> mean: 1.05
        # col 2: [-10.0, 1.9, 2.0, 2.1] -> trimmed: [1.9, 2.0] -> mean: 1.95
        # col 3: [2.9, 3.0, 3.1, 50.0] -> trimmed: [3.0, 3.1] -> mean: 3.05
        result = trimmed_mean(self.updates, beta=0.25)
        np.testing.assert_allclose(result, np.array([1.05, 1.95, 3.05]))

    def test_krum(self):
        # With 1 byzantine client, krum should reject the outlier and select one of the clean updates
        # n = 4, d = 1 -> neighbors = 4 - 1 - 2 = 1 neighbor
        # Krum should select the update closest to its nearest neighbor
        result = krum(self.updates, num_byzantine=1)
        # It must not select the outlier [10.0, -10.0, 50.0]
        self.assertFalse(np.array_equal(result, self.updates[3]))


if __name__ == "__main__":
    unittest.main()
