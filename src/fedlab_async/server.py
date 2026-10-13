"""The framework-neutral server: a run is a sequence of exchanges, each one a file per node out and back.

A `Program` holds the learning logic and its state. The `Server` owns everything about files and steps:
writing requests, checking responses, waiting for all nodes, locking, saving, and rolling back a failed step.

    emit():    program.plan()   -> one request file per node
    collect(): when every node has a valid response -> program.absorb() -> next exchange, or done

Payloads are bytes; how they are encoded is the adapter's business. The Program is pickled with the run
state, so it is trusted input: keep the state path out of the exchange directory, and keep the Program's
state on the instance (class attributes and module globals do not survive a restart).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import random
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .envelope import EnvelopeError, Header, new_nonce, read, write
from .layout import Exchange


@dataclass(frozen=True)
class Slot:
    """Which response is being checked."""

    run_id: str
    round: int
    phase: str
    node_id: int
    seq: int


@dataclass
class Plan:
    """One exchange. (round, phase) must be different for every exchange of a run: it names the files."""

    round: int
    phase: str
    requests: dict[int, bytes]  # node id -> request payload


@dataclass(frozen=True)
class Timing:
    elapsed_seconds: float  # first request of the round to the last response, including all waiting
    server_seconds: float  # time spent in server steps this round, excluding waiting


class Program(ABC):
    @property
    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def position(self) -> tuple[int, str]:
        """(round, phase) of the next exchange; used for `status` before it is planned."""

    @abstractmethod
    def plan(self, run_id: str) -> Plan:
        """The next exchange. Only called when not done."""

    def validate(self, slot: Slot, payload: bytes) -> None:
        """Raise ValueError to reject one response (it is quarantined and the node stays outstanding)."""

    @abstractmethod
    def absorb(self, run_id: str, round: int, phase: str, responses: dict[int, bytes], timing: Timing) -> None:
        """All nodes answered: update the program's state. Raise to abort the step; nothing is saved."""


@dataclass
class _RunState:
    run_id: str
    program: Program
    seq: int = 0
    round: int = 0
    phase: str = ""
    emitted: bool = False
    pending: dict[int, str] = field(default_factory=dict)  # node id -> nonce of its outstanding request
    rng_state: tuple | None = None  # (random, numpy) global RNG state, restored around each step
    round_marker: int = -1
    round_started_at: float = 0.0
    server_seconds: float = 0.0


@dataclass
class Status:
    run_id: str
    round: int
    phase: str  # "done" when finished
    done: bool
    emitted: bool  # request files exist for this exchange
    waiting_on: list[int]  # nodes with no response file yet
    received: list[int]  # nodes whose response file is present (not yet validated)


@dataclass
class CollectResult:
    advanced: bool
    missing: list[int]
    rejected: dict[str, str]  # file name -> reason


class Server:
    def __init__(self, state_path: Path, exchange: Exchange, validate: Callable[[Slot, bytes], None] | None = None):
        self.state_path, self.exchange, self.validate = Path(state_path), exchange, validate
        self.state: _RunState = pickle.loads(self.state_path.read_bytes())

    @classmethod
    def create(cls, state_path: Path, exchange: Exchange, program: Program, **kwargs) -> Server:
        state = _RunState(uuid.uuid4().hex, program, rng_state=(random.getstate(), np.random.get_state()))
        _save(Path(state_path), state)
        return cls(state_path, exchange, **kwargs)

    @property
    def program(self) -> Program:
        return self.state.program

    @property
    def done(self) -> bool:
        return self.state.program.done

    def status(self) -> Status:
        s = self.state = pickle.loads(self.state_path.read_bytes())
        if s.program.done:
            return Status(s.run_id, s.round, "done", True, False, [], [])
        round, phase = (s.round, s.phase) if s.emitted else s.program.position()
        present = [n for n in sorted(s.pending) if self.exchange.response(s.round, s.phase, n).exists()]
        return Status(s.run_id, round, phase, False, s.emitted, [n for n in sorted(s.pending) if n not in present], present)

    def emit(self) -> list[Path]:
        """Write this exchange's request files (once). Calling it again returns the same files."""
        with self._locked():
            s = self.state
            if s.program.done:
                return []
            if not s.emitted:
                with self._step():
                    self._emit()
                self._audit(
                    {"event": "request", "node_id": n, "nonce": s.pending[n], "sha256": self._sent[n]} for n in sorted(s.pending)
                )
            return [self.exchange.request(s.round, s.phase, n) for n in sorted(s.pending)]

    def reissue(self, node_id: int | None = None, force: bool = False) -> list[Path]:
        """Put back outstanding request files that are missing or damaged, from the server's private copies.

        Use it when a transport moved a request away instead of copying it, or a file was lost or corrupted on
        the way. The restored file is the original, nonce included, so the client's work is unaffected.
        `force` replaces files that look fine.
        """
        with self._locked():
            s = self.state
            restored = []
            for node in sorted(s.pending):
                if node_id is not None and node != node_id:
                    continue
                target = self.exchange.request(s.round, s.phase, node)
                if not force and self._request_is_intact(target, s.pending[node]):
                    continue
                stored = self._copies / target.name
                if not stored.exists():
                    raise RuntimeError(f"no stored copy of {target.name}; it cannot be reissued")
                header, payload = read(stored)  # the copy is checked before it is trusted
                if header.nonce != s.pending[node]:
                    raise RuntimeError(f"the stored copy of {target.name} is not for the outstanding request")
                write(target, header, payload)
                restored.append(target)
            return restored

    def collect(self) -> CollectResult:
        """Read the responses for the outstanding requests; absorb and advance once all are valid."""
        with self._locked():
            return self._collect()

    # -- internals ---------------------------------------------------------------------------------

    @contextmanager
    def _locked(self):
        """Only one server step at a time per run, always on the latest saved state.

        The state is reloaded after the lock is taken, so a `Server` kept from before another process
        advanced the run does not act on old state and fork the run.
        """
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path.with_name(self.state_path.name + ".lock"), "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another server step is running on this run") from None
            self.state = pickle.loads(self.state_path.read_bytes())
            yield

    @contextmanager
    def _step(self):
        """One unit of server work: the program's RNG stream is restored and saved, time is counted, the new
        state is saved on success, and the saved state is reloaded on failure so nothing half-done is kept."""
        s = self.state
        caller = (random.getstate(), np.random.get_state())
        if s.rng_state is not None:
            random.setstate(s.rng_state[0])
            np.random.set_state(s.rng_state[1])
        self._step_started = time.perf_counter()
        try:
            yield
            s.rng_state = (random.getstate(), np.random.get_state())
            s.server_seconds += time.perf_counter() - self._step_started
            _save(self.state_path, s)
        except BaseException:
            self.state = pickle.loads(self.state_path.read_bytes())
            raise
        finally:
            random.setstate(caller[0])
            np.random.set_state(caller[1])

    @property
    def _copies(self) -> Path:
        return self.state_path.with_name(self.state_path.name + ".requests")

    @staticmethod
    def _request_is_intact(path: Path, nonce: str) -> bool:
        try:
            return read(path)[0].nonce == nonce
        except (OSError, EnvelopeError):
            return False

    def _emit(self) -> None:
        s = self.state
        plan = s.program.plan(s.run_id)
        if plan.round != s.round_marker:
            s.round_marker, s.round_started_at, s.server_seconds = plan.round, time.time(), 0.0
        s.seq += 1
        s.round, s.phase, s.pending, self._sent = plan.round, plan.phase, {}, {}
        for node_id, payload in plan.requests.items():
            self._sent[node_id] = hashlib.sha256(payload).hexdigest()
            nonce = new_nonce()
            write(
                self.exchange.request(s.round, s.phase, node_id),
                Header("request", s.run_id, s.round, s.phase, node_id, nonce, seq=s.seq),
                payload,
            )
            s.pending[node_id] = nonce
            path = self.exchange.request(s.round, s.phase, node_id)
            self._copies.mkdir(exist_ok=True)
            shutil.copyfile(path, self._copies / path.name)  # a private copy, so a lost request can be reissued
        s.emitted = True

    def _collect(self) -> CollectResult:
        s = self.state
        if s.program.done or not s.emitted:
            return CollectResult(False, sorted(s.pending), {})
        payloads: dict[int, bytes] = {}
        missing: list[int] = []
        rejected: dict[str, str] = {}
        for node_id in sorted(s.pending):
            path = self.exchange.response(s.round, s.phase, node_id)
            if not path.exists():
                missing.append(node_id)
                continue
            try:
                payloads[node_id] = self._accept(path, node_id)
            except (EnvelopeError, ValueError) as error:
                rejected[path.name] = str(error)
                self.exchange.quarantine(path, str(error))
                missing.append(node_id)
        if missing:
            return CollectResult(False, missing, rejected)
        with self._step():
            timing = Timing(
                time.time() - s.round_started_at, s.server_seconds + time.perf_counter() - self._step_started
            )
            s.program.absorb(s.run_id, s.round, s.phase, payloads, timing)
            accepted = {n: (s.round, s.phase, s.pending[n]) for n in payloads}
            for node in payloads:
                (self._copies / self.exchange.request(s.round, s.phase, node).name).unlink(missing_ok=True)
            s.emitted, s.pending = False, {}
        self._audit(
            {"event": "response", "node_id": n, "round": r, "phase": p, "nonce": nonce,
             "sha256": hashlib.sha256(payloads[n]).hexdigest()}
            for n, (r, p, nonce) in sorted(accepted.items())
        )
        return CollectResult(True, [], rejected)

    def _audit(self, events) -> None:
        """Append what was sent and what was accepted (with payload hashes) to `<state>.audit.jsonl`.

        This is a best-effort record for provenance: the state file is the authority, and a crash between
        saving the state and writing here can lose a line.
        """
        s = self.state
        with open(self.state_path.with_name(self.state_path.name + ".audit.jsonl"), "a") as log:
            for event in events:
                round, phase = event.pop("round", s.round), event.pop("phase", s.phase)
                log.write(json.dumps({"time": time.time(), "run_id": s.run_id, "seq": s.seq, "round": round,
                                      "phase": phase, **event}, sort_keys=True) + "\n")

    def _accept(self, path: Path, node_id: int) -> bytes:
        s = self.state
        header, payload = read(path)
        expected = Header("response", s.run_id, s.round, s.phase, node_id, s.pending[node_id], seq=s.seq)
        for name in ("kind", "run_id", "round", "phase", "node_id", "nonce", "seq"):
            if getattr(header, name) != getattr(expected, name):
                raise ValueError(
                    f"{path.name}: {name} is {getattr(header, name)!r}, expected "
                    f"{getattr(expected, name)!r}" + (" (a file from another run?)" if name == "run_id" else "")
                )
        slot = Slot(s.run_id, s.round, s.phase, node_id, s.seq)
        s.program.validate(slot, payload)
        if self.validate is not None:
            self.validate(slot, payload)
        return payload


def _save(path: Path, state: _RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(pickle.dumps(state))
    os.replace(temporary, path)
