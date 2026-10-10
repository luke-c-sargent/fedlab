"""Framework-neutral pieces of the learning strategy: early stopping and optimizer-state packing."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .contract import BestModel, OptimizerState


class EarlyStopper:
    """Tracks the best validation loss and stops after `patience` rounds without improvement."""

    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.best: BestModel | None = None
        self.stale = 0

    def update(self, server_round: int, loss: float, arrays: list[np.ndarray], validation: dict) -> bool:
        """Record a round. Returns True if training should stop now."""
        if self.best is None or loss < self.best.loss:
            self.best = BestModel(server_round, loss, [a.copy() for a in arrays], validation)
            self.stale = 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def is_cuda_oom(error: BaseException) -> bool:
    return type(error).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(error)


def oom_advice(micro_batch: int, effective_batch: int) -> str:
    """What to do about a CUDA out-of-memory error: gradient accumulation keeps the effective batch fixed."""
    smaller = [m for m in (32, 16, 8, 4) if m < micro_batch and effective_batch % m == 0]
    return (
        f"CUDA ran out of memory with micro_batch_size={micro_batch}. Lower module_options.micro_batch_size "
        f"(try {smaller[0] if smaller else 'a smaller divisor'}; it must divide {effective_batch}). "
        f"The effective batch size stays {effective_batch}, because gradients are accumulated over "
        f"{effective_batch}/micro_batch_size micro-batches."
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pack_optimizer_state(state_dict: dict) -> OptimizerState:
    """Flatten a torch-style optimizer `state_dict()` into arrays plus JSON metadata."""
    arrays: list[np.ndarray] = []
    layout: list[list] = []  # [param_index, [state names], [original shapes]]
    for index in sorted(state_dict["state"]):
        entry = state_dict["state"][index]
        names, shapes = sorted(entry), []
        for name in names:
            array = _to_numpy(entry[name])
            shapes.append(list(array.shape))
            arrays.append(np.atleast_1d(array))  # keep every array at least 1-D for the wire format
        layout.append([int(index), names, shapes])
    return OptimizerState(arrays, json.dumps({"layout": layout, "param_groups": state_dict["param_groups"]}))


def unpack_optimizer_state(packed: OptimizerState) -> dict:
    """Inverse of `pack_optimizer_state`; values come back as numpy arrays."""
    meta = json.loads(packed.meta)
    state: dict[int, dict[str, np.ndarray]] = {}
    position = 0
    for index, names, shapes in meta["layout"]:
        state[index] = {}
        for name, shape in zip(names, shapes):
            state[index][name] = np.asarray(packed.arrays[position]).reshape(shape)
            position += 1
    if position != len(packed.arrays):
        raise ValueError("optimizer state has more arrays than its layout describes")
    return {"state": state, "param_groups": meta["param_groups"]}
