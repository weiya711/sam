from sam.onyx.hw_nodes.hw_node import *


class WriteScannerNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)

    def connect(self, other, edge):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode

        wr_scan = self.get_name()

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == BuffetNode:
            buffet = other.get_name()
            new_conns = {
                'buffet_to_wr_scan': [
                    # wr op/data
                    ([(wr_scan, "data_out"), (buffet, "wr_data")], 16),
                    ([(wr_scan, "op_out"), (buffet, "wr_op")], 1),
                    ([(buffet, "wr_data_ready"), (wr_scan, "data_out_ready_in")], 1),
                    ([(wr_scan, "data_out_valid_out"), (buffet, "wr_data_valid")], 1),
                    # addr
                    ([(wr_scan, "addr_out"), (buffet, "wr_addr")], 16),
                    ([(buffet, "wr_addr_ready"), (wr_scan, "addr_out_ready_in")], 1),
                    ([(wr_scan, "addr_out_valid_out"), (buffet, "wr_addr_valid")], 1),

                    # id
                    ([(wr_scan, "ID_out"), (buffet, "wr_ID")], 16),
                    ([(buffet, "wr_ID_ready"), (wr_scan, "ID_out_ready_in")], 1),
                    ([(wr_scan, "ID_out_valid_out"), (buffet, "wr_ID_valid")], 1),
                ]
            }
            return new_conns
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == WriteScannerNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect WriteScannerNode to {other_type}')

    def configure(self, attributes):
        inner_offset = 0
        # compressed = int(attributes['format'] == 'compressed')
        if 'format' in attributes:
            compressed = int(attributes['format'].strip('"') == 'compressed')
        # elif attributes['type'].strip('"') == 'arrayvals':
        else:
            compressed = 1

        # compressed = int(attributes['format'] == 'compressed')
        if attributes['type'].strip('"') == 'arrayvals':
            lowest_level = 1
            stop_lvl = 0
        elif attributes['mode'].strip('"') == 'vals':
            lowest_level = 1
            stop_lvl = 0
        else:
            lowest_level = 0
            stop_lvl = int(attributes['mode'].strip('"'))
        block_mode = int(attributes['type'].strip('"') == 'fiberlookup')
        cfg_tuple = (inner_offset, compressed, lowest_level, stop_lvl, block_mode)
        return cfg_tuple
