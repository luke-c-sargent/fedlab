# Tests

[Back to README](../README.md)

Run `uv run pytest`. The tests use mocks (moto for AWS), a fake Flower grid, and fake nodes (directories on this machine). They make no cloud calls. The tests of the COMPASS preparation and of the file-based runs run the original scripts, so they need a checkout of `federated-learning-model` (`FEDLAB_REPO`, default `~/code/federated-learning-model`). They are skipped without it.

Slow tests are opt-in:

- `FEDLAB_E2E=1` runs the full lifecycle on local fake nodes with real Flower (TLS, node auth). It needs network access.
- `FEDLAB_E2E_REAL=1` runs the real COMPASS training on synthetic data on CPU, in three forms: live (`tests/test_e2e_compass_real.py`), as files (`tests/test_e2e_compass_real_async.py`), and as files moved by the `fedlab` module (same file). The docstrings list the setup and how to compare the losses with a live run.

Run the real tests one after the other. The live test reinstalls torch into `FEDLAB_REAL_ENV`, and any other process that uses that environment at the same time can fail while it does.
