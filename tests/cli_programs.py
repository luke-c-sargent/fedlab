"""Program factories for the CLI tests (importable by `package.module:name`)."""

from test_async_strategy import MomentumFedAvg, client_fn


def build(options, run_dir):
    return MomentumFedAvg(int(options["rounds"]))  # a bare Strategy: `init` wraps it


def client_factory(node_config, run_config):
    assert node_config == {"x": 1} and run_config == {"y": 2}, (node_config, run_config)  # both come from the site file
    return client_fn
