

class HWNode():
    def __init__(self) -> None:
        self._dot_node = None

    def connect(self, other):
        pass

    def configure(self, **kwargs):
        pass


class BuffetNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass

    def configure(self, **kwargs):
        pass


class MemoryNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class ReadScannerNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):

        if type(other) == ReadScannerNode:
            raise NotImplementedError
        elif type(other) == ReadScannerNode:
            pass


class WriteScannerNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class IntersectNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class ReduceNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class LookupNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class MergeNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class RepeatNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass


class ComputeNode(HWNode):
    def __init__(self) -> None:
        super().__init__()

    def connect(self, other):
        pass