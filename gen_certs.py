"""
scripts/gen_certs.py
====================
Self-signed CA + mTLS certificate generator for FedMed local development.

Usage
-----
Generate certs for 3 clients (default output to certs/):

    python scripts/gen_certs.py --clients 3

Generate with a custom output directory and validity period:

    python scripts/gen_certs.py --clients 5 --out-dir /tmp/my-certs --days 90

Output files
------------
    certs/
        ca.pem          — CA certificate (trust anchor)
        ca.key          — CA private key  [DO NOT COMMIT]
        server.pem      — Server certificate
        server.key      — Server private key  [DO NOT COMMIT]
        client_0.pem    — Client 0 certificate
        client_0.key    — Client 0 private key  [DO NOT COMMIT]
        ...
        client_N.pem
        client_N.key

Security note
-------------
This script is for **local development / CI only**.  For production hospital
nodes, use a proper PKI (HSM-backed CA or HashiCorp Vault PKI engine).
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
except ImportError:
    print(
        "[gen_certs] ERROR: 'cryptography' package not found.\n"
        "Install it with:  pip install cryptography",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BACKEND = default_backend()


def _new_ec_key() -> ec.EllipticCurvePrivateKey:
    """Generate a fresh ECDSA P-256 private key."""
    return ec.generate_private_key(ec.SECP256R1(), _BACKEND)


def _key_to_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_to_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _name(cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _write(path: Path, content: bytes, mode: int = 0o600) -> None:
    path.write_bytes(content)
    # Restrict private key file permissions on POSIX systems
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass  # Windows doesn't support POSIX chmod — ignore silently


# ---------------------------------------------------------------------------
# Certificate builders
# ---------------------------------------------------------------------------


def build_ca(days: int = 365) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Create a self-signed CA certificate."""
    key = _new_ec_key()
    subject = issuer = _name("FedMed Local CA")

    now = _now_utc()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256(), _BACKEND)
    )
    return key, cert


def build_server_cert(
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    *,
    days: int = 365,
    hostname: str = "localhost",
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Create a server certificate signed by *ca_key*."""
    key = _new_ec_key()
    now = _now_utc()

    san = x509.SubjectAlternativeName(
        [
            x509.DNSName(hostname),
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(f"FedMed Server ({hostname})"))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(san, critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256(), _BACKEND)
    )
    return key, cert


def build_client_cert(
    ca_key: ec.EllipticCurvePrivateKey,
    ca_cert: x509.Certificate,
    client_id: int,
    *,
    days: int = 365,
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Create a client certificate signed by *ca_key*."""
    key = _new_ec_key()
    now = _now_utc()

    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(f"FedMed Client {client_id}"))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256(), _BACKEND)
    )
    return key, cert


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-signed mTLS certificates for FedMed local dev.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=3,
        metavar="N",
        help="Number of client certificates to generate (default: 3).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("certs"),
        metavar="DIR",
        help="Output directory for PEM files (default: certs/).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        metavar="DAYS",
        help="Certificate validity in days (default: 365).",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default="localhost",
        metavar="HOST",
        help="Server hostname for SAN (default: localhost).",
    )
    args = parser.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gen_certs] Generating certificates in: {out.resolve()}")
    print(f"[gen_certs] Validity: {args.days} days | Clients: {args.clients}")

    # ── CA ──────────────────────────────────────────────────────────────────
    ca_key, ca_cert = build_ca(days=args.days)
    _write(out / "ca.pem", _cert_to_pem(ca_cert), mode=0o644)
    _write(out / "ca.key", _key_to_pem(ca_key), mode=0o600)
    print("[gen_certs]   ✓ CA certificate and key written.")

    # ── Server ──────────────────────────────────────────────────────────────
    srv_key, srv_cert = build_server_cert(
        ca_key, ca_cert, days=args.days, hostname=args.hostname
    )
    _write(out / "server.pem", _cert_to_pem(srv_cert), mode=0o644)
    _write(out / "server.key", _key_to_pem(srv_key), mode=0o600)
    print("[gen_certs]   ✓ Server certificate and key written.")

    # ── Clients ─────────────────────────────────────────────────────────────
    for i in range(args.clients):
        cli_key, cli_cert = build_client_cert(ca_key, ca_cert, client_id=i, days=args.days)
        _write(out / f"client_{i}.pem", _cert_to_pem(cli_cert), mode=0o644)
        _write(out / f"client_{i}.key", _key_to_pem(cli_key), mode=0o600)
        print(f"[gen_certs]   ✓ Client {i} certificate and key written.")

    print(f"\n[gen_certs] Done! {3 + args.clients * 2} files written to {out.resolve()}")
    print("[gen_certs] IMPORTANT: Private key files (.key) are gitignored. Do NOT commit them.")


if __name__ == "__main__":
    main()
