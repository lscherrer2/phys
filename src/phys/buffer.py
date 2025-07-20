from abc import ABC

class Buffer (ABC):
    def __init__ (self, **kwargs):
       self.buffer = {}

    def flush_buffer (self):
        for key, value in self.buffer.items():
            setattr(self, key, value)
