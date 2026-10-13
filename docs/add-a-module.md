# Add a module

[Back to README](../README.md)

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
