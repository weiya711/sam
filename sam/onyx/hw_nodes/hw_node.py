from enum import Enum


class HWNodeType(Enum):
    GLB = 1
    Buffet = 2
    Memory = 3
    ReadScanner = 4
    WriteScanner = 5
    Intersect = 6
    Reduce = 7
    Lookup = 8
    Merge = 9
    Repeat = 10
    Compute = 11
    Broadcast = 12
    RepSigGen = 13


class HWNode():
    def __init__(self, name=None) -> None:
        self._dot_node = None
        if name is not None:
            self.name = name
        else:
            self.name = "default_name"

    def connect(self, other):
        pass

    def configure(self, **kwargs):
        pass

    def get_name(self):
        return self.name