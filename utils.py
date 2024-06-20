from base64 import b64decode, b64encode
import pickle
from typing import Any


def half(x: int, lb: int, ub: int):
    h = x // 2
    if h > ub:
        return ub
    return h if h >= lb else lb


def double(x: int, lb: int, ub: int):
    if x < lb:
        return lb
    d = x * 2
    return d if d <= ub else ub


def serialize(obj) -> str:
    return b64encode(pickle.dumps(obj)).decode()


def deserialize(obj) -> Any:
    return pickle.loads(b64decode(obj))
