from sam.onyx.hw_nodes.hw_node import *


class IntersectNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self.num_inputs = 2
        self.num_inputs_connected = 0
        self.num_outputs = 3
        self.num_outputs_connected = 0

    def connect(self, other, edge):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode

        isect = self.get_name()

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            out_conn = 0
            print(edge)
            comment = edge.get_attributes()['comment'].strip('"')
            if "C" in comment:
                out_conn = 1
            new_conns = {
                f'isect_to_rd_scan': [
                    # send output to rd scanner
                    ([(isect, f"pos_out_{out_conn}"), (rd_scan, f"us_pos_in")], 16),
                    ([(isect, f"eos_out_{1 + out_conn}"), (rd_scan, f"us_eos_in")], 1),
                    ([(rd_scan, f"us_ready_out"), (isect, f"ready_in_{1 + out_conn}")], 1),
                    ([(isect, f"valid_out_{1 + out_conn}"), (rd_scan, f"us_valid_in")], 1),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            new_conns = {
                f'isect_to_rd_scan': [
                    # send output to rd scanner
                    ([(isect, f"pos_out_0"), (wr_scan, f"data_in_0")], 16),
                    ([(isect, f"eos_out_0"), (wr_scan, f"eos_in_0")], 1),
                    ([(wr_scan, f"ready_out_0"), (isect, f"ready_in_0")], 1),
                    ([(isect, f"valid_out_0"), (wr_scan, f"valid_in_0")], 1),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            # TODO
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect IntersectNode to {other_type}')

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def configure(self, attributes):
        print("INTERSECT CONFIGURE")
        print(attributes)
        cmrg_enable = 0
        cmrg_stop_lvl = 0
        return (cmrg_enable, cmrg_stop_lvl)
