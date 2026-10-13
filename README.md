# fedlab

fedlab runs federated-learning experiments on AWS and GCP. It starts the VMs, installs an experiment, runs it, copies the results to your machine, and removes the VMs.

An experiment is a **module**. The module `compass_tcga_gtex` trains the COMPASS foundation model on TCGA and GTEx with [Flower](https://flower.ai). The module `compass_tcga_gtex_async` runs the same training as files instead of connections (see [File-based runs](#file-based-runs-fedlab-async)). `smoke` is a tiny numpy FedAvg that checks provisioning, TLS, and result collection.

## Default fleet

One server and two GPU clients. Each VM has a 200 GB disk and runs Ubuntu 22.04 on-demand.

| node | role | site | where | type | spec |
|---|---|---|---|---|---|
| `fedlearn-server-aws-eu-west-1` | server | - | AWS eu-west-1 | r6i.large | 2 vCPU / 16 GB |
| `fedlearn-client-aws-us-east-1` | client | tcga | AWS us-east-1 | g4dn.4xlarge | 16 vCPU / 64 GB, 1x T4 |
| `fedlearn-client-gcp-us-central1-a` | client | gtex | GCP us-central1-a | n1-highmem-8 + T4 | 8 vCPU / 52 GB, 1x T4 |

Exactly one node must have `role: server`. A client with a `site` holds the data of that site. Destroy the fleet when it is idle.

## Setup

1. Make sure that `ssh` and `rsync` are on your machine.
2. Install the packages: `uv sync`
3. Copy `.env.example` to `.env`. Add the AWS keys (or `AWS_PROFILE`) and `GCP_PROJECT`.
4. Log in to GCP: `gcloud auth application-default login`
5. Copy `config.example.yaml` to `config.yaml`. Set `module` and `module_options`.

Environment variables `FEDLAB_<KEY>` and `.env` override `config.yaml`.

## Run an experiment

```sh
uv run fedlab check        # credentials, quotas, and the local data of the module
uv run fedlab prepare      # module-specific local step (COMPASS: build the data caches)
uv run fedlab experiment   # up -> deploy -> run -> collect -> destroy
```

`experiment` destroys the fleet only after `collect` succeeds. If a step fails, it keeps the fleet, so you can read the logs. Then run `fedlab destroy`. Options:

- `--keep` leaves the fleet running after success.
- `--destroy-on-failure` also destroys the fleet after a failure.
- `--yes` skips the confirmation.

## Commands

Run each command with `uv run fedlab <command>`.

| command | action |
|---|---|
| `check` | Test credentials, permissions, quotas, instance types, and the configuration of the module. It creates nothing. |
| `modules` | List the available modules. |
| `prepare` | Run the local preparation step of the module. It uses no cloud. |
| `up` | Create all VMs, start stopped VMs, and wait for SSH. You can run it again safely. |
| `deploy` | Wait for the VMs to finish first boot. Install the software and data of the module. Start the federation. |
| `run` | Run the deployed experiment to completion. |
| `collect` | Copy the results and service logs to `results/<run_name>/<timestamp>/`. Use `--out` to choose the directory, or set `results_dir` in the config. |
| `logs <node> [superlink\|supernode]` | Show the Flower service log of a node. `--lines` sets the length. |
| `experiment` | `up`, `deploy`, `run`, `collect`, `destroy`. |
| `status`, `ping` | Show node states and IPs. Test SSH login on each node. |
| `start`, `stop` | Start or stop VMs. Stopped VMs still pay for their disks. |
| `ssh <node> [command]` | Open SSH to a node with the repo key. |
| `destroy` | Delete all resources of this run. Add `--dry-run` to list them only. |

`up`, `start`, `stop`, `ping`, and `check` accept `-n <node>` (repeat it for more nodes). All other commands act on the whole fleet. Every command except `modules` accepts `-c <file>` for a config file.

## The COMPASS module

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
- Each node checks its prepared data once during `deploy`. Every reply also carries one value, the hash of the federated manifest the client used, and the server rejects a reply that differs from its own. The original sent three more dataset ids with each reply. They are not ported.
- Only the default path is ported: historical protocol, scratch initialization, random negatives, and the full GTEx set. Other settings raise an error.

## Add a module

A module is a class that implements `FLModule` (`src/fedlab/modules/base.py`):

| method | job |
|---|---|
| `validate(nodes)` | Local checks. Return a list of problems. |
| `prepare(log)` | Optional local step that builds inputs. |
| `stage(d)` | Install software and upload code and data. |
| `start(d)` | Start the long-running services. |
| `run(d)` | Run to completion. Return a run id. Raise on failure. |
| `collect(d, dest)` | Copy results to the local directory `dest`. |
| `stop(d)` | Stop the services. |

`d` is a `Deployment`: the server and clients, each with a `Host` for commands, file copies, and services. For a Flower app, subclass `FlowerModule` (`src/fedlab/modules/flower_module.py`). It handles the Python 3.11 environment, TLS, node authentication, `flwr run`, and result collection. You supply the app directory, a requirements file, and the run config. `src/fedlab/modules/smoke/` is a small example. Register the class in `src/fedlab/modules/__init__.py`.

## File-based runs (`fedlab-async`)

`fedlab-async` runs the same federated training without any connection between the server and the sites. The server writes **request files**. Each site runs its request and writes a **response file**. The server reads the responses, aggregates, and writes the next requests. You move the files by any means: `rsync`, a shared bucket, a USB drive.

There are two ways to use it:

- **Directly.** `fedlab-async` needs no cloud account, no SSH, and no `fedlab` install. Each site runs `fedlab-async client` on its own machine. Use this between institutions that cannot reach each other.
- **Through `fedlab`.** For a fleet you own, the module `compass_tcga_gtex_async` creates the VMs, installs everything, runs the same `fedlab-async` commands on them over SSH, and copies the files for you. Use this for testing on real machines. See [Run it on a fleet with `fedlab`](#run-it-on-a-fleet-with-fedlab).

The code is in `src/fedlab_async/`. It does not import `fedlab`, and its core does not import Flower. Flower strategies and `ClientApp`s run through an adapter. A plain adapter runs ordinary Python functions.

### Server

A run directory holds `exchange/` (the only part you move), `private/` (state, lock, audit log, copies of outstanding requests) and `results/`. Never share `private/`: its state file is a pickle, so whoever can write it controls the server.

```sh
uv run fedlab-async init run1 --config run.yaml   # create the run
uv run fedlab-async emit run1                     # write request files; move them to the sites
uv run fedlab-async status run1                   # who is outstanding
uv run fedlab-async collect run1                  # read responses; advances only when every site has answered
```

Repeat `emit` and `collect` until `status` says `finished`. `emit` and `collect` are separate, so you can wait days between them. A response that fails a check (it belongs to another run or round, it was cut short, or its content is invalid) is moved to `exchange/rejected/` with a reason, and that site stays outstanding.

If a request file is lost or damaged in transit, `fedlab-async reissue run1` puts it back from the server's private copy, with the same nonce. `fedlab-async inspect FILE` shows the header of any exchange file and whether its payload is intact.

### Site

```sh
uv run fedlab-async client --exchange exchange/ --state-dir ~/fedlab-state \
    --site-file site.yaml --app fedlab_compass.async_client:app
```

Without request files on the command line, `client` runs this node's pending requests (`node_id` in the site file). Running a request again is safe. A request older than the last one the node ran is refused. If the client fails, it writes no response and keeps its state. For plain functions, use `--plain package.module:function`.

Put the directories that hold the program and client classes on the import path with `--path DIR` (repeatable) or `FEDLAB_ASYNC_PATH` (separated by `:`).

### COMPASS example

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

The first step is **prepare**. Each site fetches its assigned raw file and checks its hash, then prepares its own data. It reports its per-gene minimum and maximum values, its prepared-data manifest (sample counts and hashes), and the gene list. It sends no sample rows. The server combines the reports into the global scaler and the federated manifest, and sends both with every later request. Then training runs as usual.

Every site must install the training code itself: the two directories of `federated-learning-model`, the pinned COMPASS source, and the Python environment from `requirements.txt` plus `typer` and `pyyaml`. Each request carries a hash of the expected code: every `*.py` and `*.json` file in those directories, and the COMPASS commit if you set `compass_source` or `compass_commit`. A site whose code differs refuses to run.

Data sources for `file://` and `http(s)://` exist today. For S3 or DRS, add a `Source` (see `src/fedlab_async/assets.py`).

**Status.** Real COMPASS training has run as files on CPU with synthetic data, and its losses and weights were identical to a live run. GPU runs, real TCGA and GTEx data, and runs over real SSH connections have not been tried.

### Run it on a fleet with `fedlab`

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

This fits fleets you control. It does not fit institutions that do not trust each other, because `fedlab` holds SSH access to every node. There, use `fedlab-async` directly. On CPU, with the same data, seed, and thread count, the results matched the live module exactly. GPU training is not bit-reproducible in general.

## Flower deployment (live modules)

- The server runs the SuperLink and submits runs. Each client runs a SuperNode.
- `deploy` creates a new CA, a server certificate for the current IP of the server, and one key per SuperNode. Only registered SuperNodes can join. Stop and start change the IPs, so run `deploy` again afterward.
- The Control and ServerAppIo APIs listen only on the loopback address of the server.
- Python environments are in `~/fl` on each node. The results are in `~/results` on the server.

## Local files

Public IPs change after each stop and start. After `up`, `start`, `stop`, `status`, and `destroy`, fedlab rewrites these files:

- `.fedlab/nodes.json` contains the role, site, IP, SSH user, SSH key, and open ports of each node.
- `~/.ssh/config.d/fedlab` lets you use `ssh <node>`. On first run, fedlab asks before it adds an `Include` line to `~/.ssh/config`.

The same directory holds the SSH keypair `id_ed25519`. `destroy` keeps it.

## GPUs

- **AWS:** `gpu: true` selects the "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)". It includes the driver and CUDA.
- **GCP:** The image is stock Ubuntu. cloud-init runs `ubuntu-drivers install --gpgpu`, then reboots the node once. `deploy` waits for the driver.
- **GCP:** `accelerator:` attaches a GPU to an N1 machine. fedlab sets the maintenance policy to terminate, because GPU VMs cannot live-migrate.

`check` tests the AWS vCPU quota with a launch dry run. It also tests the GCP GPU quota.

**GCP capacity.** Google can refuse a launch when a zone is out of GPU machines (`ZONE_RESOURCE_POOL_EXHAUSTED`). No call can tell you that in advance, so `check` cannot see it. For a GCP node, list other zones to try in order:

```yaml
- {role: client, site: gtex, provider: gcp, location: us-central1-a, fallback_locations: [us-central1-b, us-central1-c], machine_type: g2-standard-4, gpu: true}
```

`up` tries `location` first and moves to the next zone only on this error. Quota and permission errors stop at once, because another zone would not help. The node keeps its name (it still ends in `us-central1-a`), and `status`, `start`, `stop`, and `destroy` look in every listed zone. `check` tests the machine type in each zone. Fallback zones are for GCP only. For AWS, a launch in a region that is out of capacity is not retried elsewhere.

## Ports and security

| role | open ports (from 0.0.0.0/0) |
|---|---|
| server | 22, 9092 (set with `ports`) |
| client | 22 |

SSH accepts the repo key only. Flower traffic uses TLS, and SuperNodes must be registered. The generated SSH config does not check host keys, because IPs are reused.

### SSH uses your SSH configuration

fedlab runs plain `ssh` and `rsync`, so your `~/.ssh/config` applies (`ProxyJump`, multiplexing, per-host rules). It overrides only the repo key (`IdentitiesOnly`), `BatchMode`, host-key checking, and timeouts. If `fedlab ping` fails but another `ssh` works, compare with `ssh -G ubuntu@<node-ip>`.

## Teardown

fedlab tags every resource with `fedlab-run=<run_name>`. `destroy` removes only tagged resources, in the locations of your config. These are instances, disks, firewall rules or security groups, and AWS key pairs.

GCP firewall rules cannot carry labels. `destroy` removes them by name: `<run_name>-server` and `<run_name>-client`.

- Use a unique `run_name`. Two stacks with the same name delete each other.
- If you remove a region from the config, destroy its resources first.

`status` warns when a node runs for more than `warn_after_days` (default 3).

## Tests

Run `uv run pytest`. The tests use mocks (moto for AWS), a fake Flower grid, and fake nodes (directories on this machine). They make no cloud calls. The tests of the COMPASS preparation and of the file-based runs run the original scripts, so they need a checkout of `federated-learning-model` (`FEDLAB_REPO`, default `~/code/federated-learning-model`). They are skipped without it.

Slow tests are opt-in:

- `FEDLAB_E2E=1` runs the full lifecycle on local fake nodes with real Flower (TLS, node auth). It needs network access.
- `FEDLAB_E2E_REAL=1` runs the real COMPASS training on synthetic data on CPU, in three forms: live (`tests/test_e2e_compass_real.py`), as files (`tests/test_e2e_compass_real_async.py`), and as files moved by the `fedlab` module (same file). The docstrings list the setup and how to compare the losses with a live run.

Run the real tests one after the other. The live test reinstalls torch into `FEDLAB_REAL_ENV`, and any other process that uses that environment at the same time can fail while it does.
