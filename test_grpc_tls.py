"""
security/tests/test_grpc_tls.py
================================
Unit tests for security.grpc_tls.

Tests cover:
* _read() raises FileNotFoundError on missing file.
* load_server_credentials() succeeds with valid cert files.
* load_channel_credentials() succeeds with valid cert files.
* load_channel_credentials() raises FileNotFoundError for a missing client cert.
* verify_cert_chain() returns True for a valid chain.
* verify_cert_chain() returns False for a mismatched CA.
* is_cert_expired() correctly identifies an expired / valid cert.

These tests are self-contained: they generate ephemeral certs in-memory
using the same `cryptography` library (no disk I/O for cert data, only
for the tempdir used to satisfy the file-based API).
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so imports work without install
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Cert generation helpers (inline, no dependency on gen_certs.py)
# ---------------------------------------------------------------------------

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

_BACKEND = default_backend()


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _pem_cert(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _pem_key(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )


def _make_ca() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1(), _BACKEND)
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name("Test CA"))
        .issuer_name(_name("Test CA"))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256(), _BACKEND)
    )
    return key, cert


def _make_leaf(
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    cn: str,
    *,
    days: int = 365,
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1(), _BACKEND)
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256(), _BACKEND)
    )
    return key, cert


def _make_expired_cert(
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    cn: str = "Expired",
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Create a certificate whose validity window ended in the past."""
    key = ec.generate_private_key(ec.SECP256R1(), _BACKEND)
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=10))
        .not_valid_after(now - datetime.timedelta(days=1))  # expired yesterday
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256(), _BACKEND)
    )
    return key, cert


# ---------------------------------------------------------------------------
# Fixture helper: write certs to a temp directory
# ---------------------------------------------------------------------------


class _CertFixture:
    """Creates a temporary cert directory with CA, server, and N clients."""

    def __init__(self, n_clients: int = 2):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.d = Path(self._tmpdir.name)

        self.ca_key, self.ca_cert = _make_ca()
        self.srv_key, self.srv_cert = _make_leaf(self.ca_key, self.ca_cert, "server")

        (self.d / "ca.pem").write_bytes(_pem_cert(self.ca_cert))
        (self.d / "server.pem").write_bytes(_pem_cert(self.srv_cert))
        (self.d / "server.key").write_bytes(_pem_key(self.srv_key))

        self.client_keys = []
        self.client_certs = []
        for i in range(n_clients):
            k, c = _make_leaf(self.ca_key, self.ca_cert, f"client_{i}")
            self.client_keys.append(k)
            self.client_certs.append(c)
            (self.d / f"client_{i}.pem").write_bytes(_pem_cert(c))
            (self.d / f"client_{i}.key").write_bytes(_pem_key(k))

    def cleanup(self):
        self._tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReadHelper(unittest.TestCase):
    def test_missing_file_raises(self):
        """_read() must raise FileNotFoundError with a helpful message."""
        from security.grpc_tls import _read

        with self.assertRaises(FileNotFoundError) as ctx:
            _read(Path("/nonexistent/path/ca.pem"))
        self.assertIn("gen_certs", str(ctx.exception))


class TestLoadServerCredentials(unittest.TestCase):
    def setUp(self):
        self.fx = _CertFixture()

    def tearDown(self):
        self.fx.cleanup()

    def test_returns_server_credentials(self):
        """load_server_credentials() should return a grpc.ServerCredentials object."""
        import grpc
        from security.grpc_tls import load_server_credentials

        creds = load_server_credentials(self.fx.d)
        self.assertIsInstance(creds, grpc.ServerCredentials)

    def test_missing_ca_raises(self):
        """Should raise FileNotFoundError if ca.pem is absent."""
        from security.grpc_tls import load_server_credentials

        (self.fx.d / "ca.pem").unlink()
        with self.assertRaises(FileNotFoundError):
            load_server_credentials(self.fx.d)

    def test_missing_server_key_raises(self):
        """Should raise FileNotFoundError if server.key is absent."""
        from security.grpc_tls import load_server_credentials

        (self.fx.d / "server.key").unlink()
        with self.assertRaises(FileNotFoundError):
            load_server_credentials(self.fx.d)


class TestLoadChannelCredentials(unittest.TestCase):
    def setUp(self):
        self.fx = _CertFixture(n_clients=2)

    def tearDown(self):
        self.fx.cleanup()

    def test_returns_channel_credentials(self):
        """load_channel_credentials() should return a grpc.ChannelCredentials object."""
        import grpc
        from security.grpc_tls import load_channel_credentials

        creds = load_channel_credentials(self.fx.d, client_id=0)
        self.assertIsInstance(creds, grpc.ChannelCredentials)

    def test_missing_client_cert_raises(self):
        """Should raise FileNotFoundError for an unknown client_id."""
        from security.grpc_tls import load_channel_credentials

        with self.assertRaises(FileNotFoundError):
            load_channel_credentials(self.fx.d, client_id=99)

    def test_all_clients_loadable(self):
        """All generated client credentials should load without error."""
        import grpc
        from security.grpc_tls import load_channel_credentials

        for i in range(2):
            creds = load_channel_credentials(self.fx.d, client_id=i)
            self.assertIsInstance(creds, grpc.ChannelCredentials)


class TestVerifyCertChain(unittest.TestCase):
    def setUp(self):
        self.ca_key, self.ca_cert = _make_ca()
        self.leaf_key, self.leaf_cert = _make_leaf(
            self.ca_key, self.ca_cert, "leaf"
        )
        # A second, unrelated CA
        self.other_ca_key, self.other_ca_cert = _make_ca()

    def test_valid_chain_returns_true(self):
        from security.grpc_tls import verify_cert_chain

        result = verify_cert_chain(
            _pem_cert(self.leaf_cert), _pem_cert(self.ca_cert)
        )
        self.assertTrue(result)

    def test_wrong_ca_returns_false(self):
        from security.grpc_tls import verify_cert_chain

        result = verify_cert_chain(
            _pem_cert(self.leaf_cert), _pem_cert(self.other_ca_cert)
        )
        self.assertFalse(result)

    def test_tampered_cert_returns_false(self):
        """Flipping bytes in the cert should fail verification."""
        from security.grpc_tls import verify_cert_chain

        cert_pem = bytearray(_pem_cert(self.leaf_cert))
        # Corrupt a byte in the middle of the DER data section
        cert_pem[len(cert_pem) // 2] ^= 0xFF
        result = verify_cert_chain(bytes(cert_pem), _pem_cert(self.ca_cert))
        self.assertFalse(result)


class TestIsCertExpired(unittest.TestCase):
    def setUp(self):
        self.ca_key, self.ca_cert = _make_ca()

    def test_valid_cert_not_expired(self):
        from security.grpc_tls import is_cert_expired

        _, cert = _make_leaf(self.ca_key, self.ca_cert, "valid", days=365)
        self.assertFalse(is_cert_expired(_pem_cert(cert)))

    def test_expired_cert_detected(self):
        from security.grpc_tls import is_cert_expired

        _, cert = _make_expired_cert(self.ca_key, self.ca_cert)
        self.assertTrue(is_cert_expired(_pem_cert(cert)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
