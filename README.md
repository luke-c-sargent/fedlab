# fedlab

Harness for launching (and cleanly tearing down) VMs on AWS and GCP for federated-learning tests with [Flower](https://flower.ai).

Default fleet (3 nodes, each 4 vCPU / 16 GB / 200 GB, Ubuntu 22.04, on-demand):

| node | where | type |
|---|---|---|
| `fedlearn-aws-us-east-1` | AWS us-east-1 | m5.xlarge, gp3 |
| `fedlearn-aws-eu-west-1` | AWS eu-west-1 | m5.xlarge, gp3 |
| `fedlearn-gcp-us-central1-a` | GCP us-central1-a | n2-standard-4, pd-balanced |

## Setup

```sh
uv sync
cp .env.example .env              # AWS keys (or AWS_PROFILE) + GCP_PROJECT
cp config.example.yaml config.yaml  # optional; defaults match the table above
gcloud auth application-default login   # GCP auth (ADC)
```

## Usage

```sh
uv run fedlab up                  # create everything (idempotent), wait for SSH
uv run fedlab status              # states, IPs, uptime; refreshes local files
uv run fedlab stop [-n NODE]      # stop VMs (disks still bill)
uv run fedlab start [-n NODE]     # start VMs; new IPs are picked up automatically
uv run fedlab ssh fedlearn-aws-us-east-1
uv run fedlab destroy --dry-run   # list what would be deleted
uv run fedlab destroy             # delete everything tagged for this run
```

### Changing IPs

Public IPs are ephemeral (no Elastic/static IPs, so no idle-IP cost) and change after stop/start. After `up`, `start`, `stop` and `status`, fedlab regenerates:

- `.fedlab/nodes.json` – per-node IP, SSH user/key, and Flower addresses (Fleet API `:9092`, ServerAppIo `:9091`, Exec API `:9093`). Read this from your Flower scripts.
- `~/.ssh/config.d/fedlab` – so `ssh fedlearn-aws-us-east-1` works. On first run fedlab asks before adding `Include ~/.ssh/config.d/fedlab` to the top of `~/.ssh/config`.

### Provisioning

cloud-init installs Python 3, `uv`, `git`, and a venv at `~/fl` with the latest `flwr` (`source ~/fl/bin/activate`). It runs after SSH comes up; wait for `/var/lib/fedlab-ready` (`ssh <node> 'cloud-init status --wait'`).

### Security

Ports 22, 9091–9093 are open to **0.0.0.0/0**. SSH is key-only (a dedicated ed25519 key in `.fedlab/`). Flower gRPC is plaintext/unauthenticated unless you enable TLS in Flower. Host-key checking is disabled in the generated SSH config because IPs are reused.

### Teardown & scope

Every resource is tagged/labelled `fedlab-run=<run_name>`. `destroy` only touches tagged resources, in the locations listed in your config (if you remove a region from the config, destroy it first). It is idempotent. The local keypair is kept.

Costs: stopped VMs still pay for their 200 GB disks.

## Tests

```sh
uv run pytest      # mocked (moto for AWS); no cloud calls
```
