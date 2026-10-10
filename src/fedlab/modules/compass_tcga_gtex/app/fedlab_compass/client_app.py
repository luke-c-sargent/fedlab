"""ClientApp: each message runs in a fresh process, so optimizer state is stored between rounds.

State goes to `$FEDLAB_STATE_DIR` on the node when that is set, else to `context.state`. The
SuperNode passes `context.state` over a gRPC channel limited to 4 MiB; COMPASS's Adam state is about 8 MiB.
"""

import os
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.clientapp import ClientApp

from .apps_common import client_backend
from .contract import OptimizerState
from .orchestrator import pack_reply

app = ClientApp()
OPT_ARRAYS, OPT_META = "optimizer_arrays", "optimizer_meta"


def _state_file(context: Context) -> Path | None:
    directory = os.environ.get("FEDLAB_STATE_DIR")
    return Path(directory) / f"optimizer-run{context.run_id}-node{context.node_id}.npz" if directory else None


def load_optimizer(context: Context) -> OptimizerState | None:
    path = _state_file(context)
    if path is not None:
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as saved:
            count = int(saved["count"])
            return OptimizerState([saved[f"a{i}"] for i in range(count)], str(saved["meta"]))
    if OPT_META not in context.state:
        return None
    arrays = context.state[OPT_ARRAYS].to_numpy_ndarrays() if OPT_ARRAYS in context.state else []
    return OptimizerState(arrays, str(context.state[OPT_META]["json"]))


def save_optimizer(context: Context, optimizer: OptimizerState | None) -> None:
    if optimizer is None:
        return
    path = _state_file(context)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in path.parent.glob(f"optimizer-run*-node{context.node_id}.npz"):
            if old != path:
                old.unlink(missing_ok=True)  # state from earlier runs is never reused
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, count=len(optimizer.arrays), meta=optimizer.meta, **{f"a{i}": a for i, a in enumerate(optimizer.arrays)})
        os.replace(temporary, path)
        return
    context.state[OPT_ARRAYS] = ArrayRecord(optimizer.arrays)
    context.state[OPT_META] = ConfigRecord({"json": optimizer.meta})


@app.train()
def train(msg: Message, context: Context) -> Message:
    backend = client_backend(context.run_config, context.node_config)
    result = backend.fit(
        msg.content["arrays"].to_numpy_ndarrays(), dict(msg.content["config"]), load_optimizer(context)
    )
    save_optimizer(context, result.optimizer)
    reply = pack_reply(result.arrays, result.metrics, backend.weight)
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    backend = client_backend(context.run_config, context.node_config)
    metrics = backend.evaluate(msg.content["arrays"].to_numpy_ndarrays(), dict(msg.content["config"]))
    return Message(content=pack_reply(None, metrics, backend.weight), reply_to=msg)
