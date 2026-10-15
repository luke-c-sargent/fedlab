# File-based runs (`fedlab-async`)

[Back to README](../README.md)

`fedlab-async` runs the same federated training without any connection between the server and the sites. The server writes **request files**. Each site runs its request and writes a **response file**. The server reads the responses, aggregates, and writes the next requests. You move the files by any means: `rsync`, a shared bucket, a USB drive.

"File-based" and "async" mean the same thing here: the CLI is `fedlab-async` and the module is `compass_tcga_gtex_async`.

There are two ways to use it:

- **Directly.** `fedlab-async` needs no cloud account, no SSH, and no `fedlab` install. Each site runs `fedlab-async client` on its own machine. Use this between institutions that cannot reach each other.
- **Through `fedlab`.** For a fleet you own, the module `compass_tcga_gtex_async` creates the VMs, installs everything, runs the same `fedlab-async` commands on them over SSH, and copies the files for you. Use this for testing on real machines. See [Run it on a fleet with `fedlab`](#run-it-on-a-fleet-with-fedlab).

## Server

A run directory holds `exchange/` (the only part you move), `private/` (state, lock, audit log, copies of outstanding requests) and `results/`. Never share `private/`: its state file is a pickle, so whoever can write it controls the server.

```sh
uv run fedlab-async init run1 --config run.yaml   # create the run
uv run fedlab-async emit run1                     # write request files; move them to the sites
uv run fedlab-async status run1                   # who is outstanding
uv run fedlab-async collect run1                  # read responses; advances only when every site has answered
```

Repeat `emit` and `collect` until `status` says `finished`. `emit` and `collect` are separate, so you can wait days between them. A response that fails a check (it belongs to another run or round, it was cut short, or its content is invalid) is moved to `exchange/rejected/` with a reason, and that site stays outstanding.

If a request file is lost or damaged in transit, `fedlab-async reissue run1` puts it back from the server's private copy, with the same nonce. `fedlab-async inspect FILE` shows the header of any exchange file and whether its payload is intact.

## Site

```sh
uv run fedlab-async client --exchange exchange/ --state-dir ~/fedlab-state \
    --site-file site.yaml --client-factory fedlab_compass.neutral_client:make_client
```

Without request files on the command line, `client` runs this node's pending requests (`node_id` in the site file). Running a request again is safe. A request older than the last one the node ran is refused. If the client fails, it writes no response and keeps its state. For a plain client function (`fedlab_core.ClientFn`), use `--client package.module:function`.

Put the directories that hold the program and client classes on the import path with `--path DIR` (repeatable) or `FEDLAB_ASYNC_PATH` (separated by `:`).

## COMPASS example

`run.yaml` (server):

```yaml
program: fedlab_compass.async_run:build
options:
  code_root: ~/code/federated-learning-model/compass_hpc_foundation_model_train
  backend: compass            # or stub, to test the plumbing
  rounds: 100
  patience: 10
  seed: 42
  # compass_source: ~/COMPASS # a checkout; puts the COMPASS commit in the code identity (or give compass_commit)
  nodes:                      # sha256 and bytes are measured by reading the file, if you leave them out
    1: {site: tcga, source: https://data.example.org/tcga_compass_tpm.tsv}
    2: {site: gtex, source: file:///data/gtex_compass_tpm.tsv}
```

`site.yaml` (at each site). `backend` and `seed` must match `run.yaml`:

```yaml
node_id: 1
node: {site: tcga, workdir: ~/fedlab-site, cache: ~/fedlab-cache, code_root: ~/code/federated-learning-model/compass_hpc_foundation_model_train}
run: {backend: compass, seed: 42, device: cuda, micro-batch-size: 16, num-workers: 0, cpu-threads: 8, local-epochs: 1}
```

The first step is **prepare**: each site fetches its raw file, prepares its own data, and reports per-gene minimum and maximum values, its manifest, and the gene list (no sample rows). The server combines them into the global scaler and sends it with every later request. See [The COMPASS module](compass-module.md) for what the scaler is.

Every site must install the training code itself: the two directories of `federated-learning-model`, the pinned COMPASS source, and the Python environment from `requirements.txt` plus `typer` and `pyyaml`. Each request carries a hash of the expected code: every `*.py` and `*.json` file in those directories, and the COMPASS commit if you set `compass_source` or `compass_commit`. A site whose code differs refuses to run.

Data sources: `file://` and `http(s)://`. For S3 or DRS, add a `Source` (see `src/fedlab_async/assets.py`).

## Run it on a fleet with `fedlab`

Set `module: compass_tcga_gtex_async` in `config.yaml`. The usual commands then work as for the live module: `fedlab up`, `deploy`, `run`, `collect`, `destroy`, or `fedlab experiment` for all of them.

```yaml
module: compass_tcga_gtex_async
module_options:
  compass_repo: ~/code/federated-learning-model
  tcga_tsv: ~/data/TCGA_dataset/processed/tcga_compass_tpm.tsv   # raw files; the sites prepare their own data from them
  gtex_tsv: ~/data/GTEx_dataset/processed/gtex_compass_tpm.tsv
  rounds: 100               # patience, seed, device, and the other training options are as for compass_tcga_gtex
  # backend: stub           # real data preparation, numpy training
  # install_environment: false   # the nodes already have ~/fl
```

- `deploy` installs the environment and the code on every node, uploads each site's raw file, writes the run and site files, and runs `fedlab-async init` on the server. It starts no Flower services, and `fedlab prepare` is not needed.
- `run` repeats one step until `fedlab-async status` says `finished`: `emit` on the server, copy each request to its site, run `fedlab-async client` on both sites in parallel, copy the responses back, `collect`.
- `collect` copies `results/` (the model files and `rounds.json`) and the audit log. There are no service logs, so `fedlab logs` has nothing to show. To debug, run `fedlab ssh <node>` and use `fedlab-async status` or `inspect` there.
