from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

class Buffer(ABC):
    def __init__ (self, **kwargs):
        self.buffer = {}

    def flush (self):
        for attr, val in self.buffer.items():
            if hasattr(self, attr):
                setattr(self, attr, val)
            else:
                raise AttributeError(
                    f"Attempted to flush buffer property '{attr}'"
                     "to no corresponding object property"
                )
        self.buffer = {}




