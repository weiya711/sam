from sam.onyx.hw_nodes.hw_node import *


class MergeNode(HWNode):
    def __init__(self, name=None, outer=None, inner=None) -> None:
        super().__init__(name=name)
        self.outer = outer
        self.inner = inner

    def get_outer(self):
        return self.outer

    def get_inner(self):
        return self.inner

    def connect(self, other, edge):

        merge = self.get_name()

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode

        new_conns = None
        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == ReadScannerNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            conn = 0
            comment = edge.get_attributes()['comment'].strip('"')
            print("MERGE TO WR SCAN")
            print(comment)
            if 'outer' in comment:
                conn = 1
            new_conns = {
                f'merge_{conn}_to_wr_scan': [
                    ([(merge, f"cmrg_coord_out_{conn}"), (wr_scan, f"data_in")], 17),
                    # ([(merge, f"cmrg_eos_out_{conn}"), (wr_scan, f"eos_in_0")], 1),
                    # ([(wr_scan, f"ready_out_0"), (merge, f"cmrg_ready_in_{conn}")], 1),
                    # ([(merge, f"cmrg_valid_out_{conn}"), (wr_scan, f"valid_in_0")], 1),
                ]
            }

            return new_conns
        elif other_type == IntersectNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == MergeNode:
            out_conn = 1
            in_conn = 1

            other_merge = other.get_name()
            # Use inner to process outer
            merge_outer = other.get_outer()
            merge_inner = other.get_inner()
            conn = 0
            print(edge)
            print("INTERSECT TO MERGE")
            comment = edge.get_attributes()['comment'].strip('"')
            print(comment)
            print(merge_outer)
            print(merge_inner)
            if merge_outer in comment:
                conn = 1
            new_conns = {
                f'isect_to_merger_{conn}': [
                    # Send isect row and isect col to merger inside isect_col
                    ([(merge, f"cmrg_coord_out_{out_conn}"), (other_merge, f"cmrg_coord_in_{in_conn}")], 17),
                    # ([(isect, "eos_out_0"), (merge, f"cmrg_eos_in_{conn}")], 1),
                    # ([(merge, f"cmrg_ready_out_{conn}"), (isect, "ready_in_0")], 1),
                    # ([(isect, "valid_out_0"), (merge, f"cmrg_valid_in_{conn}")], 1),
                ]
            }

        elif other_type == RepeatNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == ComputeNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect MergeNode to {other_type}')

        return new_conns

    def configure(self, attributes):
        print("MERGE CONFIGURE")
        print(attributes)
        cmrg_enable = 1
        # TODO what is this supposed to be?
        cmrg_stop_lvl = 1
        op = 0
        cfg_kwargs = {
            'cmrg_enable': cmrg_enable,
            'cmrg_stop_lvl': cmrg_stop_lvl,
            'op': op
        }
        return (cmrg_enable, cmrg_stop_lvl, op), cfg_kwargs
