"""Server-side scaler aggregation equals the original 02_aggregate_scaler.py, and the original validators accept it.

Runs the real prep scripts on synthetic data (numpy and pandas only, no torch). Skipped if the
federated-learning-model checkout is not found: set FEDLAB_REPO to its path.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synth_compass
from fedlab_compass import aggregate as agg

REPO = Path(os.environ.get("FEDLAB_REPO", "~/code/federated-learning-model")).expanduser() / "compass_hpc_foundation_model_train"
pytestmark = pytest.mark.skipif(not REPO.is_dir(), reason="federated-learning-model checkout not found (set FEDLAB_REPO)")
SCRIPTS, CONFIG = REPO / "federated_test" / "scripts", REPO / "centralized_test" / "config" / "paper_pretraining.json"


@pytest.fixture(scope="module")
def fc():
    for path in (SCRIPTS, REPO / "centralized_test" / "scripts"):
        sys.path.insert(0, str(path))
    import federated_common

    return federated_common


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Real 01 and 02 on synthetic data."""
    root = tmp_path_factory.mktemp("reference")
    tcga, gtex = synth_compass.write_tsvs(root)
    clients = root / "prepared" / "clients"
    run = lambda *args: subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True, text=True)  # noqa: E731
    run(SCRIPTS / "01_prepare_clients.py", "--tcga", tcga, "--gtex", gtex, "--config", CONFIG, "--output-root", clients, "--seed", 42)
    run(SCRIPTS / "02_aggregate_scaler.py", "--clients-root", clients, "--output", root / "prepared" / "global_scaler.npz",
        "--manifest", root / "prepared" / "federated_manifest.json", "--config", CONFIG)
    return root / "prepared"


def reports(reference):
    return {
        client_id: agg.ClientReport(
            (reference / "clients" / directory / "manifest.json").read_bytes(),
            (reference / "clients" / directory / "scaler_stats.npz").read_bytes(),
        )
        for client_id, directory in (("TCGA", "tcga"), ("GTEx", "gtex"))
    }


def run_aggregate(fc, reference, reports_, **kwargs):
    return agg.aggregate(fc, reports_, CONFIG, str(reference / "global_scaler.npz"), **kwargs)


def normalized(manifest):
    """Drop what legitimately differs: the scaler file's hash (and everything derived from it)."""
    m = json.loads(json.dumps(manifest))
    m["scaler"].pop("sha256")
    m["prepared_artifact_hashes"].pop("global_scaler_sha256")
    m["prepared_data_identity"].pop("global_scaler_sha256")
    for key in ("prepared_data_id", "federated_dataset_id", "manifest_payload_sha256"):
        m.pop(key)
    return m


def test_matches_the_original_script(fc, reference):
    result = run_aggregate(fc, reference, reports(reference))
    original = json.loads((reference / "federated_manifest.json").read_text())
    assert normalized(result.manifest) == normalized(original)
    with np.load(io.BytesIO(result.scaler_bytes)) as mine, np.load(reference / "global_scaler.npz") as theirs:
        assert set(mine.files) == set(theirs.files)
        for name in theirs.files:
            assert mine[name].dtype == theirs[name].dtype
            np.testing.assert_array_equal(mine[name], theirs[name])


def write_result(fc, reference, tmp_path, result):
    (tmp_path / "global_scaler.npz").write_bytes(result.scaler_bytes)
    (tmp_path / "federated_manifest.json").write_bytes(result.manifest_bytes)
    return tmp_path / "federated_manifest.json", tmp_path / "global_scaler.npz"


def test_the_clients_own_metadata_check_accepts_it(fc, reference, tmp_path):
    manifest, scaler = write_result(fc, reference, tmp_path, run_aggregate(fc, reference, reports(reference)))
    fc.validate_federated_manifest_metadata(manifest, global_scaler=scaler, config_path=CONFIG, expected_split_seed=42)


def test_the_full_original_validation_accepts_it_against_the_real_client_caches(fc, reference, tmp_path):
    manifest, scaler = write_result(fc, reference, tmp_path, run_aggregate(fc, reference, reports(reference)))
    fc.validate_federated_prepared_manifest(manifest, clients_root=reference / "clients", global_scaler=scaler, config_path=CONFIG)


def test_it_is_deterministic(fc, reference):
    a = run_aggregate(fc, reference, reports(reference))
    b = run_aggregate(fc, reference, reports(reference))
    assert a.scaler_bytes == b.scaler_bytes and a.manifest_bytes == b.manifest_bytes


def forge(fc, report, stats=None, **changes):
    """A client report whose manifest is edited and re-sealed, as a client with a bug (or worse) would send."""
    manifest = json.loads(report.manifest_bytes)
    stats = report.scaler_stats_bytes if stats is None else stats
    manifest["artifact_hashes"]["scaler_stats_sha256"] = hashlib.sha256(stats).hexdigest()
    for key, value in changes.items():
        if key == "gene_order_sha256":
            manifest["artifact_hashes"][key] = value
        else:
            manifest[key] = value
    manifest["client_dataset_id"] = fc.stable_json_hash(fc.client_prepared_identity(manifest))
    return agg.ClientReport(json.dumps(manifest, indent=2, sort_keys=True).encode(), stats)


def stats_bytes(minimum, maximum):
    return agg.npz_bytes(feature_min=np.asarray(minimum, dtype=np.float64), feature_max=np.asarray(maximum, dtype=np.float64),
                         sample_count=np.asarray([1], dtype=np.int64))


def test_it_refuses_bad_reports(fc, reference, tmp_path):
    good = reports(reference)
    with pytest.raises(ValueError, match="expected reports from"):
        run_aggregate(fc, reference, {"TCGA": good["TCGA"]})

    other_config = tmp_path / "other_config.json"
    other_config.write_text(CONFIG.read_text() + "\n")
    with pytest.raises(ValueError, match="different configuration"):
        agg.aggregate(fc, good, other_config, "x")

    swapped = {"TCGA": good["GTEx"], "GTEx": good["TCGA"]}
    with pytest.raises(ValueError, match="belongs to"):
        run_aggregate(fc, reference, swapped)

    tampered = json.loads(good["TCGA"].manifest_bytes)
    tampered["samples"] += 1
    with pytest.raises(ValueError, match="identity does not match"):
        run_aggregate(fc, reference, {**good, "TCGA": agg.ClientReport(json.dumps(tampered).encode(), good["TCGA"].scaler_stats_bytes)})

    flipped = bytearray(good["GTEx"].scaler_stats_bytes)
    flipped[-1] ^= 1
    with pytest.raises(ValueError, match="not the ones its manifest attests"):
        run_aggregate(fc, reference, {**good, "GTEx": agg.ClientReport(good["GTEx"].manifest_bytes, bytes(flipped))})

    with pytest.raises(ValueError, match="other than the assigned"):
        run_aggregate(fc, reference, good, expected_source_sha256={"TCGA": "0" * 64})
    assigned = {c: json.loads(r.manifest_bytes)["source_sha256"] for c, r in good.items()}
    run_aggregate(fc, reference, good, expected_source_sha256=assigned)  # the real hashes pass


def test_it_refuses_inconsistent_clients(fc, reference):
    good = reports(reference)
    genes = len(np.load(io.BytesIO(good["TCGA"].scaler_stats_bytes))["feature_min"])
    ok = np.zeros(genes)

    bad_range = forge(fc, good["GTEx"], stats_bytes(ok + 1.0, ok))  # max below min
    with pytest.raises(ValueError, match="Invalid extrema"):
        run_aggregate(fc, reference, {**good, "GTEx": bad_range})

    not_finite = forge(fc, good["GTEx"], stats_bytes(np.where(np.arange(genes) == 0, np.nan, 0.0), ok + 1.0))
    with pytest.raises(ValueError, match="Invalid extrema"):
        run_aggregate(fc, reference, {**good, "GTEx": not_finite})

    other_genes = forge(fc, good["GTEx"], stats_bytes(np.zeros(genes - 1), np.ones(genes - 1)))
    with pytest.raises(ValueError, match="feature counts differ"):
        run_aggregate(fc, reference, {**good, "GTEx": other_genes})

    with pytest.raises(ValueError, match="gene orders differ"):
        run_aggregate(fc, reference, {**good, "GTEx": forge(fc, good["GTEx"], gene_order_sha256="1" * 64)})

    with pytest.raises(ValueError, match="different split seeds"):
        run_aggregate(fc, reference, {**good, "GTEx": forge(fc, good["GTEx"], split_seed=7)})


# ---- the per-site half of 01, and the code identity -------------------------------------------------

from fedlab_compass import identity, prepare  # noqa: E402

CODE_ROOT = REPO
SEED = 42


def prepared_sites(reference, tmp_path):
    source = {"tcga": reference.parent / "tcga.tsv", "gtex": reference.parent / "gtex.tsv"}
    return {s: prepare.prepare_site(CODE_ROOT, s, source[s], tmp_path / "clients" / s, SEED) for s in source}


def test_a_site_prepared_alone_equals_what_01_wrote_for_it(fc, reference, tmp_path):
    prepared_sites(reference, tmp_path)
    for site in ("tcga", "gtex"):
        mine, theirs = tmp_path / "clients" / site, reference / "clients" / site
        assert {p.name for p in mine.iterdir()} == {p.name for p in theirs.iterdir()}
        for path in theirs.iterdir():
            if path.name not in ("scaler_stats.npz", "manifest.json"):  # npz embeds a timestamp; see below
                assert (mine / path.name).read_bytes() == path.read_bytes(), path.name
        with np.load(mine / "scaler_stats.npz") as a, np.load(theirs / "scaler_stats.npz") as b:
            assert set(a.files) == set(b.files)
            for name in b.files:
                np.testing.assert_array_equal(a[name], b[name])
        drop = lambda m: {k: v for k, v in m.items() if k not in ("client_dataset_id",)} | {  # noqa: E731
            "artifact_hashes": {k: v for k, v in m["artifact_hashes"].items() if k != "scaler_stats_sha256"}}
        assert drop(json.loads((mine / "manifest.json").read_text())) == drop(json.loads((theirs / "manifest.json").read_text()))


def test_the_original_validator_accepts_a_site_prepared_alone(fc, reference, tmp_path):
    prepared_sites(reference, tmp_path)
    for site, client_id in (("tcga", "TCGA"), ("gtex", "GTEx")):
        fc.validate_client_prepared_manifest(
            tmp_path / "clients" / site, expected_client_id=client_id,
            expected_config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(), expected_split_seed=SEED,
        )


def test_sites_prepared_alone_aggregate_into_a_federated_cache_the_original_accepts(fc, reference, tmp_path):
    """The whole exchange-0 data path, without 01 or 02: clients prepare alone, the server aggregates their reports."""
    sites = prepared_sites(reference, tmp_path)
    result = agg.aggregate(fc, {"TCGA": sites["tcga"], "GTEx": sites["gtex"]}, CONFIG, str(tmp_path / "global_scaler.npz"),
                           expected_source_sha256={"TCGA": hashlib.sha256((reference.parent / "tcga.tsv").read_bytes()).hexdigest(),
                                                   "GTEx": hashlib.sha256((reference.parent / "gtex.tsv").read_bytes()).hexdigest()})
    manifest, scaler = write_result(fc, reference, tmp_path, result)
    fc.validate_federated_prepared_manifest(manifest, clients_root=tmp_path / "clients", global_scaler=scaler, config_path=CONFIG)


def test_preparing_alone_refuses_bad_input(reference, tmp_path):
    with pytest.raises(ValueError, match="unknown site"):
        prepare.prepare_site(CODE_ROOT, "lung", reference.parent / "tcga.tsv", tmp_path / "x", SEED)
    short = tmp_path / "short.tsv"
    short.write_text("sample_id\tcancer_type\tcancer_code\tG1\tG2\n")
    with pytest.raises(ValueError, match="Expected 15,672 genes; found 2"):
        prepare.prepare_site(CODE_ROOT, "tcga", short, tmp_path / "x", SEED)
    with pytest.raises(FileNotFoundError):
        prepare.prepare_site(CODE_ROOT, "tcga", tmp_path / "missing.tsv", tmp_path / "x", SEED)


def make_code(root):
    for tree in identity.TREES:
        (root / tree / "scripts").mkdir(parents=True)
        (root / tree / "scripts" / "a.py").write_text("x = 1\n")
        (root / tree / "config.json").write_text("{}\n")
    return root


def test_code_identity_tracks_code_and_config_only(tmp_path):
    root = make_code(tmp_path / "code")
    base = identity.code_identity(root)
    assert identity.code_identity(root) == base
    (root / "federated_test" / "README.md").write_text("notes")
    (root / "federated_test" / "scripts" / "__pycache__").mkdir()
    (root / "federated_test" / "scripts" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\0")
    (root / "federated_test" / "slurm.sbatch").write_text("#!/bin/bash")
    assert identity.code_identity(root) == base  # docs, bytecode and job scripts do not count
    (root / "federated_test" / "scripts" / "a.py").write_text("x = 2\n")
    changed = identity.code_identity(root)
    assert changed["trees"]["federated_test"] != base["trees"]["federated_test"]
    assert changed["trees"]["centralized_test"] == base["trees"]["centralized_test"]


def test_code_identity_notices_added_renamed_and_edited_config_files(tmp_path):
    root = make_code(tmp_path / "code")
    base = identity.code_identity(root)["trees"]["centralized_test"]
    (root / "centralized_test" / "config.json").write_text('{"a": 1}\n')
    edited = identity.tree_digest(root / "centralized_test")
    (root / "centralized_test" / "config.json").rename(root / "centralized_test" / "other.json")
    renamed = identity.tree_digest(root / "centralized_test")
    (root / "centralized_test" / "extra.py").write_text("")
    added = identity.tree_digest(root / "centralized_test")
    assert len({base, edited, renamed, added}) == 4


def test_check_code_identity_names_what_differs(tmp_path):
    root = make_code(tmp_path / "code")
    expected = identity.code_identity(root) | {"compass_commit": "abc123"}
    identity.check_code_identity(expected, expected)
    identity.check_code_identity({}, expected)  # nothing expected, nothing checked
    (root / "federated_test" / "scripts" / "a.py").write_text("x = 3\n")
    actual = identity.code_identity(root) | {"compass_commit": "def456"}
    with pytest.raises(identity.CodeMismatch) as error:
        identity.check_code_identity(expected, actual)
    message = str(error.value)
    assert "federated_test/ differs" in message and "centralized_test" not in message
    assert "COMPASS commit is def456, expected abc123" in message


def test_compass_commit_comes_from_git(tmp_path):
    repo = tmp_path / "compass"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (repo / "f").write_text("x")
    git("add", "f")
    git("commit", "-q", "-m", "c")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert identity.compass_commit(repo) == head
    assert identity.compass_commit(tmp_path) is None  # not a repository
    root = make_code(tmp_path / "code")
    assert identity.code_identity(root, repo)["compass_commit"] == head


def test_the_real_training_code_has_a_stable_identity():
    assert identity.code_identity(CODE_ROOT) == identity.code_identity(CODE_ROOT)


def test_preparing_alone_works_in_a_clean_process(tmp_path):
    """Regression: the script imports its siblings by name, so prepare_site must set up sys.path itself."""
    import fedlab_compass

    app_dir = Path(fedlab_compass.__file__).parent.parent
    synth_compass.write_tsvs(tmp_path)
    code = (
        "from pathlib import Path\n"
        "from fedlab_compass.prepare import prepare_site\n"
        f"report = prepare_site(Path({str(CODE_ROOT)!r}), 'gtex', Path({str(tmp_path / 'gtex.tsv')!r}), Path({str(tmp_path / 'out')!r}), 42)\n"
        "print(len(report.manifest_bytes) > 0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(app_dir)})
    assert done.returncode == 0 and done.stdout.strip().endswith("True"), done.stderr[-800:]
