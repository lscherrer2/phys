from unittest import TestCase
from phys.buffer import Buffer

class BufferedObject (Buffer):
    def __init__ (self, a: str, b: str):
        self.a = a
        self.b = b

class TestBuffer (TestCase):

    def test_set_buffer (self):
        obj = BufferedObject('old_a', 'old_b')
        obj.buffer['a'], obj.buffer['b'] = 'new_a', 'new_b'
        self.assertEqual(obj.a, 'old_a')
        self.assertEqual(obj.b, 'old_b')
        self.assertEqual(obj.buffer['a'], 'new_a')
        self.assertEqual(obj.buffer['b'], 'new_b')

    def test_flush_buffer (self):
        obj = BufferedObject('old_a', 'old_b')
        obj.buffer['a'], obj.buffer['b'] = 'new_a', 'new_b'

        obj.flush()
        self.assertEqual(obj.a, 'new_a')
        self.assertEqual(obj.b, 'new_b')





