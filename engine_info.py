from dataclasses import dataclass, asdict, is_dataclass
import json
from typing import Callable


@dataclass
class EngineInfo:
    min_batch: int
    opt_batch: int
    max_batch: int
    min_height: int
    opt_height: int
    max_height: int
    min_width: int
    opt_width: int
    max_width: int
    min_context_len: int
    opt_context_len: int
    max_context_len: int
    in_channels: int
    context_dim: int
    y_dim: int
    dtype: str
    model_config_init: str  # base64 encoded pickle
    model_init: str  # base64 encoded pickle

    def verify(self, other: "EngineInfo"):
        return (
            other.dtype == self.dtype
            and other.in_channels == self.in_channels
            and other.y_dim == self.y_dim
            and other.context_dim == self.context_dim
            # NOTE: not checking batch size because it can be splitted
            # and self.min_batch <= other.opt_batch <= self.max_batch
            and self.min_height <= other.opt_height <= self.max_height
            and self.min_width <= other.opt_width <= self.max_width
            and self.min_context_len <= other.opt_context_len <= self.max_context_len
        )

    def _get_shapes(self, batch_size: int, height: int, width: int, context_len: int):
        s = (
            (
                batch_size,
                self.in_channels,
                height // 8,
                width // 8,
            ),
            (batch_size,),
            (batch_size, context_len * self.opt_context_len, self.context_dim),
        )
        if self.y_dim > 0:
            return s + ((batch_size, self.y_dim),)
        return s

    def min_shapes(self):
        return self._get_shapes(
            self.min_batch, self.min_height, self.min_width, self.min_context_len
        )

    def opt_shapes(self):
        return self._get_shapes(
            self.opt_batch, self.opt_height, self.opt_width, self.opt_context_len
        )

    def max_shapes(self):
        return self._get_shapes(
            self.max_batch, self.max_height, self.max_width, self.max_context_len
        )

    @classmethod
    def load(cls, info_path: str):
        with open(info_path, "r") as f:
            return cls(**json.load(f))

    def dump(self, info_path: str):
        with open(info_path, "w") as f:
            json.dump(self, f, indent=2, cls=EngineInfoJsonEncoder)


class EngineInfoJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)
