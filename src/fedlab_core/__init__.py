"""Framework-neutral federated-learning contract. Imports no runner and no training framework."""

from .loader import load_object
from .payload import Payload, decode, encode
from .strategy import ClientFn, Plan, Strategy, run_in_process

__all__ = ["ClientFn", "Payload", "Plan", "Strategy", "decode", "encode", "load_object", "run_in_process"]
