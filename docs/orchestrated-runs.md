# Orchestrated runs

In an orchestrated run, no SuperLink runs on the server VM. A separate **orchestrator** VM runs one SuperLink container for each request. All other nodes only dial out to the orchestrator on port 443.

Use this mode to test how the orchestrator copes with real traffic, and to run nodes that cannot accept inbound connections. The other two modes are not changed: `federation: direct` (the default, see the [README](../README.md)) and the file-based mode ([File-based runs](file-based-runs.md)).

## How it works

| node | what runs there |
|---|---|
| orchestrator | nginx on 443, the orchestrator service, one SuperLink container for each request |
| `server` node | the ServerApp host: a `flower-superexec` container |
| each client | a `flower-supernode` container |

The orchestrator is AWS only. The other nodes can be on AWS or GCP. fedlab creates the client VMs and points them at the orchestrator.

- `fedlab orchestrator up` makes the VM with Terraform. It also makes an Elastic IP and a new CA for this orchestrator. Requests are reached as `<id>.<ip>.sslip.io`.
- A request returns the endpoint, the CA, one private key for each client, and the SuperExec secret. fedlab copies them to the nodes.
- nginx forwards only `/v1/fleet` and `/v1/runtime`. Both need a registered node key or the SuperExec secret. The orchestrator API listens on the VM's loopback address. fedlab reaches it over SSH.
- The nodes run the image `quay.io/sargentl/fedlab-compass`. The image has only the toolchain (and an unused copy of the pinned COMPASS source, which has no `.git`). fedlab copies the training scripts, the data, and the module code to each node, fetches the pinned COMPASS commit into a git checkout there (the module's check needs a real checkout), and mounts the home directory in the container at the same path.

## Settings

| key | default | meaning |
|---|---|---|
| `federation` | `direct` | `direct` or `orchestrated` |
| `orchestrator_region` | `us-west-2` | where the orchestrator VM is made |
| `orchestrator_instance_type` | `t3.medium` | its instance type |
| `orchestrator_auto_terminate_hours` | `4` | the VM terminates itself after this time |
| `image_cpu`, `image_gpu` | pinned digests | the node images. GPU nodes use `image_gpu`. |

Set `ports: [22]` for orchestrated runs. Nodes need no inbound port except SSH.

## Run

```
uv run fedlab experiment -c config.orchestrated-smoke.yaml --yes   # stub backend, CPU nodes
uv run fedlab experiment -c config.orchestrated.yaml --yes         # real COMPASS on GPU nodes
```

If the orchestrator is not up, `experiment` makes it and removes it afterwards. An orchestrator that was already up stays up. To control it yourself, use `fedlab orchestrator up`, `status`, and `down`. Always run `down` when you finish: the orchestrator bills while it exists, until its auto-terminate time.

## The images

`fedlab build-image` builds the images for `linux/amd64` and pushes them to `quay.io/sargentl/fedlab-compass` (`cpu` and `cu126`). Log in first with `docker login quay.io`. It uses a temporary `buildx` builder and removes its build cache afterwards (`--keep-cache` keeps it).

After a build, copy the printed digests to `image_cpu` and `image_gpu` (or to `config.py`). Nodes pull by digest. The repositories must be public, or the nodes need a login.

The build runs under QEMU on Apple silicon. The Dockerfile uses `pip` because `uv` crashes there. COMPASS needs `matplotlib < 3.9`.

## Security notes

- Port 443 is open to the whole internet on the orchestrator. Control API calls are blocked by nginx, but the SuperLink Fleet and Runtime APIs are reachable. Node keys and the SuperExec secret protect them.
- Port 22 is open only to the address that ran `fedlab orchestrator up`.
- The Terraform state holds the CA private key. It is in `.fedlab/.../orchestrator/` and is not committed. Each `up` makes a new CA.
