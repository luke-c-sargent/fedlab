# fedlab

fedlab starts and removes VMs on AWS and GCP for federated-learning tests with [Flower](https://flower.ai).

The default fleet has one server and two GPU clients. Each VM has a 200 GB disk and runs Ubuntu 22.04 on-demand.

| node | role | where | type | spec |
|---|---|---|---|---|
| `fedlearn-server-aws-eu-west-1` | server | AWS eu-west-1 | r6i.large | 2 vCPU / 16 GB |
| `fedlearn-client-aws-us-east-1` | client | AWS us-east-1 | g6e.2xlarge | 8 vCPU / 64 GB, 1x L40S |
| `fedlearn-client-gcp-us-central1-a` | client | GCP us-central1-a | n1-highmem-8 + T4 | 8 vCPU / 52 GB, 1x T4 |

Exactly one node must have `role: server`. A g6e.2xlarge costs about $2.2 per hour. Stop or destroy GPU nodes when they are idle.

## Setup

1. Install the packages: `uv sync`
2. Copy `.env.example` to `.env`. Add the AWS keys (or `AWS_PROFILE`) and `GCP_PROJECT`.
3. Log in to GCP: `gcloud auth application-default login`
4. Optional: copy `config.example.yaml` to `config.yaml` to change the nodes. The defaults match the table.

Environment variables `FEDLAB_<KEY>` and `.env` override `config.yaml`.

## Commands

Run each command with `uv run fedlab <command>`.

| command | action |
|---|---|
| `check` | Test credentials, permissions, quotas, and instance types. It creates nothing. |
| `up` | Create all VMs, start stopped VMs, and wait for SSH. You can run it again safely. |
| `status` | Show state, IP, and uptime. It also refreshes the local files. |
| `ping` | Test SSH login on each node. |
| `start`, `stop` | Start or stop VMs. Stopped VMs still pay for their disks. |
| `ssh <node> [command]` | Open SSH to a node with the repo key. |
| `destroy` | Delete all resources of this run. Add `--dry-run` to list them only. |

`up`, `start`, `stop`, `ping`, and `check` accept `-n <node>` (repeat it for more nodes). `destroy` ignores `-n`. All commands accept `-c <file>` for a config file.

Always run `destroy --dry-run` first and read the list.

## Local files

Public IPs change after each stop and start. After `up`, `start`, `stop`, and `status`, fedlab rewrites these files:

- `.fedlab/nodes.json` contains the role, IP, SSH user, SSH key, and open ports of each node.
- `~/.ssh/config.d/fedlab` lets you use `ssh <node>`. On first run, fedlab asks before it adds an `Include` line to `~/.ssh/config`.

The same directory holds the SSH keypair `id_ed25519`. `destroy` keeps it.

## Flower endpoints

Only the server entry in `nodes.json` has a `flower` block. The block holds the addresses of the three Flower APIs of the SuperLink:

| key | port | used by |
|---|---|---|
| `fleet_api` | 9092 | SuperNodes on the clients |
| `serverappio_api` | 9091 | ServerApp processes |
| `exec_api` | 9093 | `flwr run` |

For clients, `flower` is `null`. Clients connect out to `fleet_api` and accept no Flower connections.

fedlab does not start Flower. The server IP changes after each stop and start, so read `nodes.json` again.

## Provisioning

cloud-init installs Python 3, `uv`, `git`, and a venv with the latest `flwr` at `~/fl`. It runs after SSH starts.

To wait for it, run `ssh <node> 'cloud-init status --wait'`. The file `/var/lib/fedlab-ready` shows that it is done.

## GPUs

- **AWS:** `gpu: true` selects the "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)". It includes the driver and CUDA.
- **GCP:** The image is stock Ubuntu. cloud-init runs `ubuntu-drivers install --gpgpu`, then reboots the node once. This starts about one minute after first boot. After the reboot, run `nvidia-smi`.
- **GCP:** `accelerator:` attaches a GPU to an N1 machine. fedlab sets the maintenance policy to terminate, because GPU VMs cannot live-migrate.

New accounts often have no GPU quota. `check` tests the AWS vCPU quota with a launch dry run. It also tests the GCP GPU quota.

## Ports and security

| role | open ports (from 0.0.0.0/0) |
|---|---|
| server | 22, 9091, 9092, 9093 (set with `ports`) |
| client | 22 |

SSH accepts the repo key only. Flower gRPC is not encrypted and has no login, unless you enable TLS in Flower. The generated SSH config does not check host keys, because IPs are reused.

## Teardown

fedlab tags every resource with `fedlab-run=<run_name>`. `destroy` removes only tagged resources, in the locations of your config. These are instances, disks, firewall rules or security groups, and AWS key pairs.

GCP firewall rules cannot carry labels. `destroy` removes them by name: `<run_name>-server` and `<run_name>-client`.

- Use a unique `run_name`. Two stacks with the same name delete each other.
- If you remove a region from the config, destroy its resources first.
- `destroy` can run again safely.

`status` warns when a node runs for more than `warn_after_days` (default 3).

## Tests

Run `uv run pytest`. The tests use mocks (moto for AWS) and make no cloud calls.
