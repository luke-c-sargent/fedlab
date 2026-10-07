"""Throwaway TLS material for one deployment: a CA, a server certificate, and per-node auth keys."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

VALID_DAYS = 14


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _san(host: str) -> x509.GeneralName:
    try:
        return x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        return x509.DNSName(host)


def _pem_key(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


@dataclass(frozen=True)
class NodeKeys:
    private_path: Path
    public_path: Path


def _write(path: Path, data: bytes, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    return path


def make_ca(directory: Path) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Create `ca.pem` / `ca.key` in `directory`."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name("fedlab dev CA"))
        .issuer_name(_name("fedlab dev CA"))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=VALID_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    _write(directory / "ca.key", _pem_key(key))
    _write(directory / "ca.pem", cert.public_bytes(serialization.Encoding.PEM), 0o644)
    return key, cert


def load_ca(directory: Path) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = serialization.load_pem_private_key((directory / "ca.key").read_bytes(), password=None)
    cert = x509.load_pem_x509_certificate((directory / "ca.pem").read_bytes())
    return key, cert  # type: ignore[return-value]


def issue_server_cert(
    directory: Path,
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
    hosts: list[str],
) -> tuple[Path, Path]:
    """Create `server.pem` / `server.key` valid for every host (IP or DNS name) in `hosts`."""
    ca_key, ca_cert = ca
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(hosts[0]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName([_san(h) for h in dict.fromkeys(hosts)]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path = _write(directory / "server.pem", cert.public_bytes(serialization.Encoding.PEM), 0o644)
    key_path = _write(directory / "server.key", _pem_key(key))
    return cert_path, key_path


def make_node_keys(directory: Path, name: str) -> NodeKeys:
    """Create an OpenSSH-format ECDSA keypair for SuperNode authentication (`<name>` / `<name>.pub`)."""
    key = ec.generate_private_key(ec.SECP384R1())
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()
    )
    public = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
    return NodeKeys(_write(directory / name, private), _write(directory / f"{name}.pub", public + b"\n", 0o644))
