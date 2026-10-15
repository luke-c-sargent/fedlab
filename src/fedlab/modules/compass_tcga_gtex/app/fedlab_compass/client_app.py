"""ClientApp: a site's COMPASS client function (see `neutral_client`). Each message runs in a fresh process, so the
optimizer state between rounds is kept in `$FEDLAB_STATE_DIR` on the node (the gRPC state channel is limited to 4 MiB;
COMPASS's Adam state is about 8 MiB)."""

from fedlab_core.flower.runner import make_client_app

from .neutral_client import make_client

app = make_client_app(lambda context: make_client(dict(context.node_config), dict(context.run_config)))
