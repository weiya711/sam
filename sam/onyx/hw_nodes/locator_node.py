from dis import code_info
from sam.onyx.hw_nodes.hw_node import *


class LocatorNode(HWNode):
    def __init__(self, name=None, outer=None, inner=None, locate_lvl=None, locate_dim_size=None) -> None:
        super().__init__(name=name)
        self.locate_lvl = locate_lvl
        self.locate_dim_size = locate_dim_size

    def connect(self, other, edge, kwargs=None):

        locator = self.get_name()

        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.pass_through_node import PassThroughNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        new_conns = None
        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            locator = self.get_name()
            new_conns = {
                'locator_to_rd_scan': [
                    ([(locator, "addr_out"), (rd_scan, "us_pos_in")], 17),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == PassThroughNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')
        elif other_type == FiberAccessNode:
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            init_conns = self.connect(that_flavor, edge)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns
        else:
            raise NotImplementedError(f'Cannot connect LocatorNode to {other_type}')

    def configure(self, attributes):
        locate_lvl = self.locate_lvl
        locate_dim_size = self.locate_dim_size
        # 0 for compression, 1 for crddrop
        cfg_kwargs = {
            'locate_lvl': locate_lvl,
            'locate_dim_size': locate_dim_size,
        }
        return (locate_lvl, locate_dim_size), cfg_kwargs
