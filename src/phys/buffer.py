from abc import ABC

__all__ = ["Buffer"]


class Buffer(ABC):
    __slots__ = ("buffer",)

    def __init__(self, **kwargs):
        self.buffer = {}

    def flush_buffer(self):
        for key, value in self.buffer.items():
            setattr(self, key, value)
