"""One neutral strategy, three runners: in process, as files, and on (a stand-in for) live Flower. Same answer."""

import numpy as np
import pytest
from flwr.app import Context, RecordDict

from fedlab_async.adapters.flower_server import _RecordingGrid
from fedlab_core.flower.runner import make_client_app, serve
from test_async_strategy import MomentumFedAvg, client_fn, in_memory, run_files

FLOWER_IDS = {1: 1001, 2: 2002, 3: 3003}  # Flower gives each SuperNode a random id; the strategy plans for 1, 2, 3


class LiveGrid(_RecordingGrid):
    """In-process stand-in for the SuperLink: delivers each message to the ClientApp of the SuperNode it is for."""

    def __init__(self, app, node_ids=FLOWER_IDS):
        super().__init__(list(node_ids.values()), 1)
        self.app, self.node_ids = app, node_ids

    def send_and_receive(self, messages, *, timeout=None):
        by_flower_id = {v: k for k, v in self.node_ids.items()}
        replies = []
        for m in messages:
            flower_id = m.metadata.dst_node_id
            reply = self.app(m, Context(7, flower_id, {"node-id": by_flower_id[flower_id]}, RecordDict(), {}))
            replies.append(reply)
        return replies


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDLAB_STATE_DIR", str(tmp_path / "state"))
    return make_client_app(lambda context: client_fn)


def test_live_flower_matches_in_process_and_files(app, tmp_path):
    live = serve(MomentumFedAvg(8), LiveGrid(app))
    reference = in_memory(8)
    files, _ = run_files(tmp_path / "files", 8)
    np.testing.assert_array_equal(live.w, reference.w)
    np.testing.assert_allclose(files.strategy.w, reference.w)
    assert live.rounds_seen == reference.rounds_seen == files.strategy.rounds_seen
    assert live.rounds_seen[-1] == {1: 8, 2: 8, 3: 8}  # client state survived between messages


def test_a_missing_node_is_a_clear_error(app):
    grid = LiveGrid(app, {1: 1001, 2: 2002})  # node 3 never came online
    with pytest.raises(RuntimeError, match=r"node\(s\) \[3\] are not online"):
        serve(MomentumFedAvg(2), grid)


def test_duplicate_node_ids_are_refused(app):
    grid = LiveGrid(app)  # two SuperNodes both started with node-id=1
    grid.send_and_receive = lambda messages, timeout=None: [
        app(m, Context(7, m.metadata.dst_node_id, {"node-id": 1}, RecordDict(), {})) for m in messages
    ]
    with pytest.raises(RuntimeError, match="same node-id"):
        serve(MomentumFedAvg(2), grid)
