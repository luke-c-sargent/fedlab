"""Program factories for the CLI tests (importable by `package.module:name`)."""

from fedlab_async.adapters.plain import PlainProgram
from test_async_plain import MomentumFedAvg


def build(options, run_dir):
    return PlainProgram(MomentumFedAvg(), int(options["rounds"]))
