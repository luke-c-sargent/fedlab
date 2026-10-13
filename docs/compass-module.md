# The COMPASS module

[Back to README](../README.md)

The TCGA client holds 33 cancer contexts. The GTEx client holds the normal-tissue context. FedAvg weights them 33 to 1. Training stops early when the validation loss does not improve for `patience` rounds.

Options (`module_options`):

| option | default | meaning |
|---|---|---|
| `compass_repo` | required | Your checkout of `federated-learning-model`. fedlab uploads `compass_hpc_foundation_model_train/{centralized_test,federated_test}` from it. |
| `tcga_tsv`, `gtex_tsv` | required for `prepare` | The source expression TSVs: the `processed/*_compass_tpm.tsv` files of the original project. They are inputs only. |
| `prepared_root` | `.fedlab/compass/prepared` | Output of `prepare`: the per-site caches, the scaler, and the manifest. Use a new directory if you change `seed`. |
| `prep_python` | this Python | Interpreter with numpy and pandas for `prepare`. |
| `rounds`, `patience`, `local_epochs`, `seed` | 100, 10, 1, 42 | Training schedule. |
| `micro_batch_size`, `num_workers`, `cpu_threads` | 16, 0, 8 | Client runtime. `micro_batch_size` must divide 128. Gradients are accumulated, so the effective batch stays 128. The original used 64, which exhausts a 16 GB T4. |
| `device` | `cuda` | Use `cpu` for tests. |
| `backend` | `compass` | `stub` runs a numpy stand-in that writes `stub_result.json`. It tests the cloud plumbing without data or GPU work. |

**Data.** `prepare` runs two original scripts on your machine. The first converts each TSV to a memory-mapped float32 array, a sample list, a validation split, and per-gene min and max values. The second combines the min and max values into one shared scaler. During `deploy`, the TCGA cache goes only to the TCGA client and the GTEx cache only to the GTEx client. Every node receives the manifest, the scaler, and the gene list. Each node then checks that its data matches the manifest.

**Results.** `collect` copies `run-<id>/` from the server. It holds `pretrainer_federated_tcga_gtex.pt` (the COMPASS pretrainer), `best_model.pth`, `history.tsv`, and `rounds.json` (per-round losses, times, and GPU memory). It also copies the SuperLink and SuperNode logs.

**Differences from the original scripts.** The port uses Flower 1.33 (Message API) instead of 1.8.0.

- Flower 1.33 needs Python 3.11, so the module uses `torch 2.0.1` and `torchvision 0.15.2` (see `requirements.txt`).
- Each message runs in a new process. The Adam state is saved in a file on the node between rounds. It is too large for the 4 MiB `context.state` channel of Flower. The dropout RNG is seeded again each round.
- Each node checks its prepared data once during `deploy`. Every reply carries the hash of the federated manifest the client used, and the server rejects a reply that differs from its own. The original's three extra dataset ids are not sent.
- Only the default path is ported: historical protocol, scratch initialization, random negatives, and the full GTEx set. Other settings raise an error.
