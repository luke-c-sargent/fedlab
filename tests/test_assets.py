import functools
import hashlib
import http.server
import io
import threading

import pytest

from fedlab_async.assets import (
    Asset,
    AssetCache,
    AssetError,
    FileSource,
    Manifest,
    Source,
    Sources,
    load_assignment,
    reference,
)


def make_asset(directory, name, content: bytes) -> Asset:
    path = directory / name
    path.write_bytes(content)
    return Asset(name, path.as_uri(), hashlib.sha256(content).hexdigest(), len(content))


class CountingFiles(FileSource):
    def __init__(self):
        self.opened = []

    def open(self, uri):
        self.opened.append(uri)
        return super().open(uri)


def cache_with(tmp_path, source=None):
    return AssetCache(tmp_path / "cache", Sources([source or FileSource()]))


def test_fetch_verifies_and_caches(tmp_path):
    a = make_asset(tmp_path, "a.bin", b"hello shard")
    source = CountingFiles()
    cache = cache_with(tmp_path, source)
    path = cache.get(a)
    assert path.read_bytes() == b"hello shard" and path == cache.path_for(a.sha256)
    assert cache.get(a) == path and len(source.opened) == 1  # second time: no fetch


def test_a_file_with_the_wrong_hash_is_rejected_and_never_cached(tmp_path):
    good = make_asset(tmp_path, "a.bin", b"expected content")
    (tmp_path / "a.bin").write_bytes(b"someone changed it!")  # same name, different bytes
    cache = cache_with(tmp_path)
    with pytest.raises(AssetError, match="not the assigned file"):
        cache.get(good)
    assert not cache.path_for(good.sha256).exists()
    assert not [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]  # no temp file left behind


def test_a_truncated_file_is_rejected(tmp_path):
    good = make_asset(tmp_path, "a.bin", b"0123456789")
    (tmp_path / "a.bin").write_bytes(b"01234")
    with pytest.raises(AssetError, match="5 bytes"):
        cache_with(tmp_path).get(good)


def test_missing_file_and_unknown_scheme_are_clear_errors(tmp_path):
    cache = cache_with(tmp_path)
    gone = Asset("x", (tmp_path / "gone.bin").as_uri(), "0" * 64, 1)
    with pytest.raises(AssetError, match="cannot fetch"):
        cache.get(gone)
    with pytest.raises(AssetError, match="no source for 's3'"):
        cache.get(Asset("x", "s3://bucket/key", "0" * 64, 1))
    with pytest.raises(AssetError, match="no source for"):
        cache.get(Asset("x", "just/a/path", "0" * 64, 1))


def test_a_damaged_cache_entry_is_fetched_again(tmp_path):
    a = make_asset(tmp_path, "a.bin", b"payload")
    cache = cache_with(tmp_path)
    path = cache.get(a)
    path.write_bytes(b"x")  # wrong size
    assert cache.get(a).read_bytes() == b"payload"


def test_the_same_content_under_two_names_is_stored_once(tmp_path):
    a = make_asset(tmp_path, "a.bin", b"same")
    b = make_asset(tmp_path, "b.bin", b"same")
    cache = cache_with(tmp_path)
    paths = cache.fetch(Manifest((a, b)))
    assert paths["a.bin"] == paths["b.bin"]


def test_manifest_round_trip_identity_and_validation(tmp_path):
    a, b = make_asset(tmp_path, "a", b"1"), make_asset(tmp_path, "b", b"2")
    manifest = Manifest((a, b))
    assert Manifest.from_json(manifest.to_json()) == manifest
    assert Manifest((a, b)).digest == manifest.digest != Manifest((b, a)).digest
    with pytest.raises(ValueError, match="unique"):
        Manifest((a, a))
    with pytest.raises(ValueError, match="sha256"):
        Asset("x", "file:///x", "nothex", 1)


def test_an_assignment_can_point_at_a_manifest_file(tmp_path):
    a = make_asset(tmp_path, "a.bin", b"data")
    manifest = Manifest((a,))
    pointer = make_asset(tmp_path, "manifest.json", manifest.to_json().encode())
    cache = cache_with(tmp_path)
    assert load_assignment(reference(pointer), cache) == manifest
    assert load_assignment(manifest.to_json(), cache) == manifest  # inline, as a JSON string
    assert load_assignment(manifest.to_dict(), cache) == manifest


def test_preflight_reports_problems_without_downloading(tmp_path):
    ok = make_asset(tmp_path, "ok.bin", b"fine")
    gone = Asset("gone", (tmp_path / "gone.bin").as_uri(), "1" * 64, 3)
    odd = Asset("odd", "s3://bucket/key", "2" * 64, 3)
    source = CountingFiles()
    cache = cache_with(tmp_path, source)
    problems = cache.preflight(Manifest((ok, gone, odd)))
    assert len(problems) == 2 and "gone" in problems[0] and "no source" in problems[1]
    assert not cache.path_for(ok.sha256).exists()  # nothing was downloaded
    cache.get(ok)
    source.opened.clear()
    assert cache.preflight(Manifest((ok,))) == [] and source.opened == []  # a cached file needs no check


def test_http_source(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    (served / "shard.bin").write_bytes(b"over http")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        uri = f"http://127.0.0.1:{server.server_address[1]}/shard.bin"
        asset = Asset("shard", uri, hashlib.sha256(b"over http").hexdigest(), 9)
        cache = AssetCache(tmp_path / "cache")  # default sources include http
        assert cache.get(asset).read_bytes() == b"over http"
        assert cache.preflight(Manifest((Asset("nope", uri.replace("shard", "nope"), "1" * 64, 1),)))
    finally:
        server.shutdown()


def test_a_new_source_can_be_registered(tmp_path):
    class Memory(Source):
        schemes = ("mem",)
        objects = {"mem://bucket/one": b"in memory"}

        def open(self, uri):
            return io.BytesIO(self.objects[uri])  # a KeyError is reported as "cannot fetch"

    asset = Asset("one", "mem://bucket/one", hashlib.sha256(b"in memory").hexdigest(), 9)
    cache = AssetCache(tmp_path / "cache", Sources([FileSource(), Memory()]))
    assert cache.get(asset).read_bytes() == b"in memory"
    with pytest.raises(AssetError, match="cannot fetch"):
        cache.get(Asset("two", "mem://bucket/two", "1" * 64, 1))
