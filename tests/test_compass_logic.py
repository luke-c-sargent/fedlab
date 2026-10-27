"""The framework-neutral pieces of the COMPASS strategy: early stopping, optimizer-state packing, OOM advice."""

import numpy as np

from fedlab_compass import logic
from fedlab_compass.contract import OptimizerState


def test_early_stopper_patience_counts_only_non_improvements():
    stopper = logic.EarlyStopper(patience=2)
    arrays = [np.zeros(1)]
    assert [stopper.update(r, loss, arrays, {}) for r, loss in enumerate([3.0, 2.0, 2.5, 1.0, 1.5, 1.6], 1)] == [
        False, False, False, False, False, True,
    ]
    assert stopper.best.round == 4


def test_optimizer_state_round_trips_through_packing():
    state = {
        "state": {0: {"step": np.float32(7), "exp_avg": np.arange(6, dtype=np.float32).reshape(2, 3), "exp_avg_sq": np.ones(4)}},
        "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999), "params": [0]}],
    }
    packed = logic.pack_optimizer_state(state)
    restored = logic.unpack_optimizer_state(OptimizerState(packed.arrays, packed.meta))
    assert restored["state"][0]["exp_avg"].shape == (2, 3)
    assert restored["state"][0]["step"].shape == ()
    assert float(restored["state"][0]["step"]) == 7
    np.testing.assert_array_equal(restored["state"][0]["exp_avg"], state["state"][0]["exp_avg"])
    assert restored["param_groups"][0]["betas"] == [0.9, 0.999]  # tuples become lists; Adam accepts both


def test_oom_advice_names_a_smaller_divisor_and_keeps_the_effective_batch():
    assert logic.is_cuda_oom(type("OutOfMemoryError", (RuntimeError,), {})("x"))
    assert logic.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 246 MiB"))
    assert not logic.is_cuda_oom(ValueError("something else"))
    advice = logic.oom_advice(64, 128)
    assert "micro_batch_size=64" in advice and "try 32" in advice and "stays 128" in advice
    assert "try 8" in logic.oom_advice(16, 128) and "try 4" in logic.oom_advice(8, 128)


def test_weighted_mean_is_the_weighted_average():
    mean = logic.weighted_mean([([np.full(3, 1.0, np.float32)], 3), ([np.full(3, 5.0, np.float32)], 1)])
    np.testing.assert_allclose(mean[0], 2.0)
    assert mean[0].dtype == np.float32


def _git(repo, *args):
    import subprocess

    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True, capture_output=True, text=True).stdout.strip()


def test_compass_commit_without_a_git_binary_matches_git(tmp_path, monkeypatch):
    import types

    from fedlab_compass import identity

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    def without_git():  # only the module under test loses git; this test's own helper keeps it
        return monkeypatch.context()

    repo = tmp_path / "c"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("x")
    _git(repo, "add", "f")
    _git(repo, "commit", "-q", "-m", "m")
    want = _git(repo, "rev-parse", "HEAD")
    assert identity.compass_commit(repo) == want  # on a branch
    _git(repo, "checkout", "-q", "--detach")
    assert identity.compass_commit(repo) == want  # detached, like the staged checkout
    with without_git() as m:
        m.setattr(identity, "subprocess", types.SimpleNamespace(run=no_git))
        assert identity.compass_commit(repo) == want
    _git(repo, "checkout", "-q", "-")
    _git(repo, "pack-refs", "--all")
    with without_git() as m:
        m.setattr(identity, "subprocess", types.SimpleNamespace(run=no_git))
        assert identity.compass_commit(repo) == want  # branch ref in packed-refs
        assert identity.compass_commit(tmp_path) is None  # not a checkout
