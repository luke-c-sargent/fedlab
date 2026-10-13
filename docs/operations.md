# Operations

[Back to README](../README.md)

## Flower deployment (live modules)

- The server runs the SuperLink and submits runs. Each client runs a SuperNode.
- `deploy` creates a new CA, a server certificate for the current IP of the server, and one key per SuperNode. Only registered SuperNodes can join. Stop and start change the IPs, so run `deploy` again afterward.
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

- `up` tries `location` first and moves to the next zone only on this error. Quota and permission errors stop at once, because another zone would not help.
- The node keeps its name (it still ends in `us-central1-a`). `status`, `start`, `stop`, and `destroy` look in every listed zone.
- `check` tests the machine type in each zone.
- Fallback zones are for GCP only. An AWS launch in a region that is out of capacity is not retried elsewhere.

## Ports and security

Open ports are set with `ports` in the config. By default the server opens 22 and 9092 and the clients open 22, all from 0.0.0.0/0. See [Architecture](../README.md#architecture) for what each port carries.

SSH accepts the repo key only. Flower traffic uses TLS, and SuperNodes must be registered. The generated SSH config does not check host keys, because IPs are reused.

### SSH uses your SSH configuration

fedlab runs plain `ssh` and `rsync`, so your `~/.ssh/config` applies (`ProxyJump`, multiplexing, per-host rules). It overrides only the repo key (`IdentitiesOnly`), `BatchMode`, host-key checking, and timeouts. If `fedlab ping` fails but another `ssh` works, compare with `ssh -G ubuntu@<node-ip>`.

## Teardown

fedlab tags every resource with `fedlab-run=<run_name>`. `destroy` removes only tagged resources, in the locations of your config. These are instances, disks, firewall rules or security groups, and AWS key pairs.

GCP firewall rules cannot carry labels. `destroy` removes them by name: `<run_name>-server` and `<run_name>-client`.

- Use a unique `run_name`. Two stacks with the same name delete each other.
- To run experiments at the same time, give each its own `run_name`, `name_prefix`, and `state_dir`.
- GCP instance names are shared by the whole project, so two experiments with the same `name_prefix` and zone would get the same VM name. `up`, `start`, `stop`, `status`, and `check` refuse a VM that carries another run's label (or none) and name it in the error.
- AWS nodes are matched by tag as well as name, so a different `run_name` is enough there.
- `fedlab` does not lock against two commands on the same config at once. The vCPU and GPU quotas are shared by the whole account.
- If you remove a region from the config, destroy its resources first.

`status` warns when a node runs for more than `warn_after_days` (default 3).
