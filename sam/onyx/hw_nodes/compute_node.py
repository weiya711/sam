from sam.onyx.hw_nodes.hw_node import *


class ComputeNode(HWNode):
    def __init__(self, name=None) -> None:
        super().__init__(name=name)
        self.num_inputs = 2
        self.num_outputs = 1
        self.num_inputs_connected = 0
        self.num_outputs_connected = 0

    def connect(self, other, edge):

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
        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode

        other_type = type(other)

        if other_type == GLBNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == BuffetNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            pe = self.get_name()
            new_conns = {
                'pe_to_rd_scan': [
                    # send output to rd scanner
                    ([(pe, "data_out"), (rd_scan, "us_pos_in")], 17),
                    # ([(pe, "eos_out"), (rd_scan, "us_eos_in")], 1),
                    # ([(rd_scan, "us_ready_out"), (pe, "ready_in")], 1),
                    # ([(pe, "valid_out"), (rd_scan, "us_valid_in")], 1),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            pe = self.get_name()
            new_conns = {
                'pe_to_wr_scan': [
                    # send output to rd scanner
                    ([(pe, "data_out"), (wr_scan, "data_in")], 17),
                    # ([(pe, "eos_out"), (wr_scan, "eos_in_0")], 1),
                    # ([(wr_scan, "ready_out_0"), (pe, "ready_in")], 1),
                    # ([(pe, "valid_out"), (wr_scan, "valid_in_0")], 1),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            in_str = "pos_in"
            offset = 1
            if 'crd' in edge.get_comment():
                in_str = "coord_in"
                offset = 0
            isect = other.get_name()
            pe = self.get_name()
            # isect_conn = other.get_num_inputs()
            isect_conn = 0
            if edge.get_tensor() == "C":
                isect_conn = 1
            new_conns = {
                f'pe_to_isect_{in_str}_{isect_conn}': [
                    # send output to rd scanner
                    ([(pe, "data_out"), (isect, f"{in_str}_{isect_conn}")], 17),
                    # ([(pe, "eos_out"), (isect, f"eos_in_{isect_conn * 2 + offset}")], 1),
                    # ([(isect, f"ready_out_{isect_conn * 2 + offset}"), (pe, "ready_in")], 1),
                    # ([(pe, "valid_out"), (isect, f"valid_in_{isect_conn * 2 + offset}")], 1),
                ]
            }
            other.update_input_connections()
            return new_conns
        elif other_type == ReduceNode:
            other_red = other.get_name()
            pe = self.get_name()
            new_conns = {
                f'pe_to_reduce': [
                    # send output to rd scanner
                    ([(pe, "data_out"), (other_red, f"data_in")], 17),
                    # ([(pe, "eos_out"), (other_red, f"eos_in")], 1),
                    # ([(other_red, f"ready_out"), (pe, "ready_in")], 1),
                    # ([(pe, "valid_out"), (other_red, f"valid_in")], 1),
                ]
            }
            return new_conns
        elif other_type == LookupNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == MergeNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == RepeatNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == ComputeNode:
            other_pe = other.get_name()
            other_conn = other.get_num_inputs()
            pe = self.get_name()
            new_conns = {
                f'pe_to_pe_{other_conn}': [
                    # send output to rd scanner
                    ([(pe, "data_out"), (other_pe, f"data_in_{other_conn}")], 17),
                    # ([(pe, "eos_out"), (other_pe, f"eos_in_{other_conn}")], 1),
                    # ([(other_pe, f"ready_out_{other_conn}"), (pe, "ready_in")], 1),
                    # ([(pe, "valid_out"), (other_pe, f"valid_in_{other_conn}")], 1),
                ]
            }
            other.update_input_connections()
            return new_conns
        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == RepSigGenNode:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == CrdHoldNode:
            raise NotImplementedError(f'Cannot connect GLBNode to {other_type}')
        else:
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def configure(self, attributes):
        print("PE CONFIGURE")
        print(attributes)
        c_op = attributes['type'].strip('"')
        print(c_op)
        op_code = 0
        if c_op == 'mul':
            op_code = 1
        elif c_op == 'add':
            op_code = 0
        cfg_kwargs = {
            'op': op_code
        }
        return op_code, cfg_kwargs
