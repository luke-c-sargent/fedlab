"""Data a client has to acquire: a manifest (what), sources (from where), and a verified local cache.

The server assigns files by putting a `Manifest` in a request. The client fetches each file through a
`Source` chosen by the URI scheme, checks its SHA-256 and size, and keeps it in a content-addressed cache,
so a file is downloaded once however often it is assigned. Nothing here knows about Flower or the file
exchange; both adapters use it the same way.

Add a source (S3, DRS, an authenticated commons) by subclassing `Source` and registering it. A source
finds its own credentials on the client; they are never part of a request.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar

_CHUNK = 1 << 20


class AssetError(Exception):
    """A file could not be fetched, or what arrived is not what the manifest promised."""


@dataclass(frozen=True)
class Asset:
    name: str  # how the client's code refers to it, unique within a manifest
    uri: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"asset {self.name!r}: sha256 must be 64 lowercase hex characters")
        if self.bytes < 0:
            raise ValueError(f"asset {self.name!r}: negative size")


@dataclass(frozen=True)
class Manifest:
    files: tuple[Asset, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.files]
        if len(set(names)) != len(names):
            raise ValueError("asset names in a manifest must be unique")

    def to_dict(self) -> dict:
        return {"files": [asdict(f) for f in self.files]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        return cls(tuple(Asset(**f) for f in data["files"]))

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls.from_dict(json.loads(text))

    @property
    def digest(self) -> str:
        """An identity for this exact assignment."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class Source(ABC):
    schemes: ClassVar[tuple[str, ...]]

    @abstractmethod
    def open(self, uri: str) -> BinaryIO:
        """A readable binary stream. Raise OSError (or anything) if it cannot be reached."""

    def exists(self, uri: str) -> bool:
        """Cheap reachability check for preflight; the default tries to open it."""
        try:
            self.open(uri).close()
            return True
        except Exception:
            return False


class FileSource(Source):
    schemes = ("file",)

    def open(self, uri: str) -> BinaryIO:
        return open(urllib.request.url2pathname(urllib.parse.urlparse(uri).path), "rb")


class HttpSource(Source):
    schemes = ("http", "https")

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def open(self, uri: str) -> BinaryIO:
        return urllib.request.urlopen(uri, timeout=self.timeout)  # type: ignore[return-value]


class Sources:
    """Sources by URI scheme."""

    def __init__(self, sources: Iterable[Source] | None = None):
        self._by_scheme: dict[str, Source] = {}
        for source in sources if sources is not None else (FileSource(), HttpSource()):
            self.register(source)

    def register(self, source: Source) -> None:
        for scheme in source.schemes:
            self._by_scheme[scheme] = source

    def for_uri(self, uri: str) -> Source:
        scheme = urllib.parse.urlparse(uri).scheme
        if scheme not in self._by_scheme:
            raise AssetError(f"no source for {scheme or 'a URI without a scheme'!r}: {uri}")
        return self._by_scheme[scheme]


class AssetCache:
    def __init__(self, root: Path, sources: Sources | None = None):
        self.root, self.sources = Path(root), sources if sources is not None else Sources()

    def path_for(self, sha256: str) -> Path:
        return self.root / "objects" / sha256[:2] / sha256

    def get(self, asset: Asset) -> Path:
        """The verified local file for an asset, fetching it if it is not cached."""
        final = self.path_for(asset.sha256)
        if final.exists() and final.stat().st_size == asset.bytes:
            return final  # only verified files are ever moved into place
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".{asset.sha256}.{os.getpid()}.tmp")
        digest, size = hashlib.sha256(), 0
        try:
            with self.sources.for_uri(asset.uri).open(asset.uri) as source, temporary.open("wb") as out:
                while chunk := source.read(_CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
            if size != asset.bytes or digest.hexdigest() != asset.sha256:
                raise AssetError(
                    f"{asset.name}: {asset.uri} is not the assigned file "
                    f"(got {size} bytes, sha256 {digest.hexdigest()[:12]}...; expected {asset.bytes} bytes, "
                    f"sha256 {asset.sha256[:12]}...)"
                )
            os.replace(temporary, final)
        except AssetError:
            raise
        except Exception as error:
            raise AssetError(f"{asset.name}: cannot fetch {asset.uri}: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)
        return final

    def fetch(self, manifest: Manifest) -> dict[str, Path]:
        """Every file of the manifest, by name. Fails on the first problem."""
        return {asset.name: self.get(asset) for asset in manifest.files}

    def preflight(self, manifest: Manifest) -> list[str]:
        """Problems that would stop `fetch`, without downloading anything. Empty means it should work."""
        problems = []
        for asset in manifest.files:
            final = self.path_for(asset.sha256)
            if final.exists() and final.stat().st_size == asset.bytes:
                continue
            try:
                if not self.sources.for_uri(asset.uri).exists(asset.uri):
                    problems.append(f"{asset.name}: cannot reach {asset.uri}")
            except AssetError as error:
                problems.append(str(error))
        return problems


def measure(name: str, uri: str, sources: Sources | None = None) -> Asset:
    """Describe a file by reading it once: for a server that is assigning data it can reach."""
    digest, size = hashlib.sha256(), 0
    try:
        with (sources or Sources()).for_uri(uri).open(uri) as stream:
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
                size += len(chunk)
    except AssetError:
        raise
    except Exception as error:
        raise AssetError(f"{name}: cannot read {uri}: {error}") from error
    return Asset(name, uri, digest.hexdigest(), size)


def reference(asset: Asset) -> dict:
    """An assignment that points at a manifest file instead of listing the files (for long lists)."""
    return {"manifest": asdict(asset)}


def load_assignment(value: str | dict, cache: AssetCache) -> Manifest:
    """An assignment as found in a request: the manifest itself, or a reference to a manifest file."""
    data = json.loads(value) if isinstance(value, str) else value
    if "manifest" in data:
        return Manifest.from_json(cache.get(Asset(**data["manifest"])).read_text())
    return Manifest.from_dict(data)
