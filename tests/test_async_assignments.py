"""The server assigns each client its own data files; clients fetch, verify and train on them."""

import hashlib
import json

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.serde import message_from_proto
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.serverapp.strategy import FedAvg

from fedlab_async.adapters.flower_client import run_client
from fedlab_async.adapters.flower_server import FlowerServer, Hooks
from fedlab_async.assets import Asset, AssetCache, AssetError, FileSource, Manifest, Sources, load_assignment
from fedlab_async.envelope import read
from fedlab_async.layout import Exchange


class CountingFiles(FileSource):
    opened: list = []

    def open(self, uri):
        CountingFiles.opened.append(uri)
        return super().open(uri)


app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Shift the model by the mean of the assigned shard; the weight is the shard's size."""
    cache = AssetCache(context.node_config["cache"], Sources([CountingFiles()]))
    manifest = load_assignment(dict(msg.content["config"])["assignment"], cache)
    shard = np.load(cache.fetch(manifest)["shard"])
    arrays = [a + shard.mean() for a in msg.content["arrays"].to_numpy_ndarrays()]
    reply = RecordDict({"arrays": ArrayRecord(arrays), "metrics": MetricRecord({"num-examples": len(shard)})})
    return Message(reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    return Message(RecordDict({"metrics": MetricRecord({"num-examples": 1, "loss": 0.0})}), reply_to=msg)


class Assigner(Hooks):
    def __init__(self, manifests):
        self.manifests = manifests

    def node_config(self, server_round, phase, node_id):
        return {"assignment": self.manifests[node_id].to_json()}


def make_shard(tmp_path, node, values):
    path = tmp_path / "remote" / f"shard-{node}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64))
    raw = path.read_bytes()
    return path, Asset("shard", path.as_uri(), hashlib.sha256(raw).hexdigest(), len(raw))


@pytest.fixture
def shards(tmp_path):
    CountingFiles.opened = []
    values = {1: [1.0, 2.0, 3.0], 2: [10.0]}  # node 1 has 3 rows, node 2 has 1
    made = {n: make_shard(tmp_path, n, v) for n, v in values.items()}
    return values, {n: path for n, (path, _) in made.items()}, {n: Manifest((asset,)) for n, (_, asset) in made.items()}


def setup(tmp_path, manifests, rounds=2):
    ex = Exchange(tmp_path / "exchange")
    state = tmp_path / "private" / "state.pkl"
    strategy = FedAvg(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2)
    FlowerServer.create(state, ex, strategy, ArrayRecord([np.zeros(2)]), [1, 2], rounds, hooks=Assigner(manifests))
    return state, ex


def client(tmp_path, ex, request):
    node = int(request.stem.rsplit("-n", 1)[1])
    return run_client(request, app, ex, tmp_path / f"clients-{node}", {"cache": str(tmp_path / f"cache-{node}")})


def run_all(tmp_path, state, ex):
    while not FlowerServer(state, ex).done:
        for request in FlowerServer(state, ex).emit():
            client(tmp_path, ex, request)
        assert FlowerServer(state, ex).collect().advanced
    return FlowerServer(state, ex)


def test_each_node_trains_on_its_own_assignment(tmp_path, shards):
    values, _, manifests = shards
    state, ex = setup(tmp_path, manifests, rounds=2)
    server = run_all(tmp_path, state, ex)
    means = {n: np.mean(v) for n, v in values.items()}
    per_round = (3 * means[1] + 1 * means[2]) / 4  # weighted by shard size
    np.testing.assert_allclose(server.result.arrays.to_numpy_ndarrays()[0], [2 * per_round] * 2)


def test_a_node_only_sees_its_own_assignment(tmp_path, shards):
    _, _, manifests = shards
    state, ex = setup(tmp_path, manifests)
    FlowerServer(state, ex).emit()
    seen = {}
    for node in (1, 2):
        _, payload = read(ex.request(1, "train", node))
        config = dict(message_from_proto(ProtoMessage.FromString(payload)).content["config"])
        seen[node] = Manifest.from_json(config["assignment"])
        assert config["server-round"] == 1  # the strategy's own config entries are kept
    assert seen[1] == manifests[1] and seen[2] == manifests[2] and seen[1] != seen[2]


def test_a_file_is_fetched_once_across_rounds(tmp_path, shards):
    _, _, manifests = shards
    state, ex = setup(tmp_path, manifests, rounds=3)
    run_all(tmp_path, state, ex)
    assert len(CountingFiles.opened) == 2  # one fetch per node, though 3 rounds ran


def test_an_unreachable_file_fails_the_client_step_and_a_rerun_works(tmp_path, shards):
    _, paths, manifests = shards
    state, ex = setup(tmp_path, manifests, rounds=1)
    requests = FlowerServer(state, ex).emit()
    hidden = paths[2].with_suffix(".hidden")
    paths[2].rename(hidden)
    client(tmp_path, ex, requests[0])
    with pytest.raises(AssetError, match="cannot fetch"):
        client(tmp_path, ex, requests[1])
    assert not ex.response(1, "train", 2).exists()
    assert FlowerServer(state, ex).collect().missing == [2]
    hidden.rename(paths[2])
    client(tmp_path, ex, requests[1])
    assert FlowerServer(state, ex).collect().advanced


def test_a_changed_remote_file_is_refused(tmp_path, shards):
    _, paths, manifests = shards
    state, ex = setup(tmp_path, manifests, rounds=1)
    requests = FlowerServer(state, ex).emit()
    np.save(paths[1], np.array([99.0, 99.0, 99.0]))  # the remote data changed after it was assigned
    with pytest.raises(AssetError, match="not the assigned file"):
        client(tmp_path, ex, requests[0])
    assert not ex.response(1, "train", 1).exists()


def audit_lines(state):
    return [json.loads(line) for line in state.with_name(state.name + ".audit.jsonl").read_text().splitlines()]


def test_the_audit_log_records_what_was_sent_and_accepted(tmp_path, shards):
    _, _, manifests = shards
    state, ex = setup(tmp_path, manifests, rounds=1)
    server = FlowerServer(state, ex)
    requests = server.emit()
    server.emit()  # a repeated emit must not log twice
    first = audit_lines(state)
    assert [e["event"] for e in first] == ["request", "request"]
    for e, request in zip(first, requests):
        _, payload = read(request)
        assert e["sha256"] == hashlib.sha256(payload).hexdigest() and e["round"] == 1 and e["phase"] == "train"
    for request in requests:
        client(tmp_path, ex, request)
    sent_responses = {n: hashlib.sha256(read(ex.response(1, "train", n))[1]).hexdigest() for n in (1, 2)}
    assert FlowerServer(state, ex).collect().advanced
    responses = [e for e in audit_lines(state) if e["event"] == "response"]
    assert {e["node_id"]: e["sha256"] for e in responses} == sent_responses
    assert all(e["phase"] == "train" and e["round"] == 1 for e in responses)


def test_a_collect_that_waits_logs_nothing(tmp_path, shards):
    _, _, manifests = shards
    state, ex = setup(tmp_path, manifests, rounds=1)
    requests = FlowerServer(state, ex).emit()
    client(tmp_path, ex, requests[0])
    assert not FlowerServer(state, ex).collect().advanced
    assert [e["event"] for e in audit_lines(state)] == ["request", "request"]
