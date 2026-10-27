# Metrics

Every run records metrics on every node, with no setting needed. After `collect`, look in `results/<run_name>/<timestamp>/metrics/`.

| file | content |
|---|---|
| `<node>.csv` | one row per second: CPU, memory, GPU, and cumulative disk and network byte counters |
| `summary.json` | the totals below, for each host |
| `transfers.jsonl` | async mode: the size and time of each request and response file |
| `steps.jsonl` | async mode: the time of each step |
| `fedlab-nginx.log` | orchestrated mode: the orchestrator's access log |

## summary.json

- `phases`: seconds for `up`, `first_boot`, `stage`, `start`, `run`, and (orchestrated) `orchestrator_up`. `events` has the raw times.
- `hosts.<node>.whole` and `.run`: the whole sampling span, and the time between `run_start` and `run_end`. Each has CPU average and peak, memory peak, disk read and write (MB), network received and sent (MB), network peaks (Mbit/s), and, for GPUs, utilization, memory, power, and energy (Wh).
- `egress_by_provider_mb`: bytes sent by the hosts of each provider. Treat it as an upper bound of billed egress: the counters cannot tell traffic between clouds from traffic inside one cloud.
- `rtt`: TCP connect time from each client to the server (direct) or to the orchestrator (orchestrated), as min, median, and max in ms.
- `image_pulls` (orchestrated): time and size of each node's image pull.
- `orchestrator_traffic` (orchestrated): bytes that nginx received (`bytes_in`) and sent (`bytes_out`) for each request, split into the `fleet` and `runtime` APIs.
- `async_steps` (async mode): the time of `emit` and `collect`, the compute time of each site, and the time each site waited for the slowest site.
- `transfers` (async mode): files, bytes, and seconds for each direction, for example `fedlab-server -> fedlab-client`.
- `run_info`: Flower and Python versions, the fedlab commit, a hash of the configuration, and the nodes.

## Compare runs

```
uv run fedlab report results/fedmx-direct/<time> results/fedorch/<time> results/fedmx-async/<time> -o results/comparison
```

With no arguments it reads every run under `results/`. It writes four files to the output directory:

| file | content |
|---|---|
| `comparison.csv`, `comparison.json` | one row per run, one column per number (for a spreadsheet or your own plots) |
| `comparison.md` | a short table for a README or a pull request |
| `comparison.html` | one page with charts: where the time went, then one bar chart per number. It needs no scripts and no network. |

Bars are coloured by execution style (`direct`, `orchestrated`, `async`). To compare the styles fairly, use the same nodes and settings: `config.matrix-direct.yaml`, `config.orchestrated.yaml` and `config.matrix-async.yaml` do this.

## Limits

- The sampler reads `/proc`, so disk and network values are blank on non-Linux hosts.
- Network counters include all traffic of the host: installs, image pulls, and SSH. Use the `run` window for the run alone.
- In async mode fedlab moves the files itself, so `transfers` is the traffic through your machine.
