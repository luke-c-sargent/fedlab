"""Smoke ClientApp: each round, move the model by 1 and count calls in `context.state`."""

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    arrays = [a + 1.0 for a in msg.content["arrays"].to_numpy_ndarrays()]
    previous = context.state.get("calls")
    calls = int(previous["n"]) + 1 if previous else 1  # proves state survives between rounds
    context.state["calls"] = ConfigRecord({"n": calls})
    metrics = {"num-examples": 1, "calls": calls}
    return Message(content=RecordDict({"arrays": ArrayRecord(arrays), "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    metrics = {"num-examples": 1, "loss": 0.0}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
