from numpy import block
from sam.onyx.hw_nodes.hw_node import *


class ReadScannerNode(HWNode):
    def __init__(self, name=None, tensor=None) -> None:
        super().__init__(name=name)
        self.tensor = tensor

    def get_tensor(self):
        return self.tensor

    def connect(self, other, edge):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode

        new_conns = None
        rd_scan = self.get_name()
        other_type = type(other)

        if other_type == GLBNode:
            other_data = other.get_data()
            other_ready = other.get_ready()
            other_valid = other.get_valid()
            new_conns = {
                'rd_scan_to_glb': [
                    # send output to rd scanner
                    ([(rd_scan, "coord_out"), (other_data, "f2io_16")], 16),
                    ([(other_ready, "io2f_1"), (rd_scan, "ready_in_0")], 1),
                    ([(rd_scan, "valid_out_0"), (other_valid, "f2io_1")], 1),
                ]
            }
            return new_conns
        elif other_type == BuffetNode:
            buffet = other.get_name()
            new_conns = {
                'buffet_to_rd_scan': [
                    # rd rsp
                    ([(buffet, "rd_rsp_data"), (rd_scan, "rd_rsp_data_in")], 16),
                    ([(rd_scan, "rd_rsp_ready_out"), (buffet, "rd_rsp_ready")], 1),
                    ([(buffet, "rd_rsp_valid"), (rd_scan, "rd_rsp_valid_in")], 1),
                    # addr
                    ([(rd_scan, "addr_out"), (buffet, "rd_addr")], 16),
                    ([(buffet, "rd_addr_ready"), (rd_scan, "addr_out_ready_in")], 1),
                    ([(rd_scan, "addr_out_valid_out"), (buffet, "rd_addr_valid")], 1),

                    # op
                    ([(rd_scan, "op_out"), (buffet, "rd_op_op")], 16),
                    ([(buffet, "rd_op_ready"), (rd_scan, "op_out_ready_in")], 1),
                    ([(rd_scan, "op_out_valid_out"), (buffet, "rd_op_valid")], 1),

                    # id
                    ([(rd_scan, "ID_out"), (buffet, "rd_ID")], 16),
                    ([(buffet, "rd_ID_ready"), (rd_scan, "ID_out_ready_in")], 1),
                    ([(rd_scan, "ID_out_valid_out"), (buffet, "rd_ID_valid")], 1),
                ]
            }
            return new_conns
        elif other_type == MemoryNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == ReadScannerNode:
            # send the ref to the next rd scanner
            other_rd_scan = other.get_name()
            new_conns = {
                'rd_scan_to_rd_scan': [
                    # send output to rd scanner
                    ([(rd_scan, "pos_out"), (other_rd_scan, "us_pos_in")], 16),
                    ([(rd_scan, "eos_out_1"), (other_rd_scan, "us_eos_in")], 1),
                    ([(other_rd_scan, "us_ready_out"), (rd_scan, "ready_in_1")], 1),
                    ([(rd_scan, "valid_out_1"), (other_rd_scan, "us_valid_in")], 1),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            # send the crd to write scanner
            wr_scan = other.get_name()
            new_conns = {
                'rd_scan_to_wr_scan': [
                    # send output to rd scanner
                    ([(rd_scan, "coord_out"), (wr_scan, "data_in_0")], 16),
                    ([(rd_scan, "eos_out_0"), (wr_scan, "eos_in_0")], 1),
                    ([(wr_scan, "ready_out_0"), (rd_scan, "ready_in_0")], 1),
                    ([(rd_scan, "valid_out_0"), (wr_scan, "valid_in_0")], 1),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            # Send both....
            isect = other.get_name()
            isect_conn = 0
            print(edge)
            print("CHECKING READ TENSOR - INTERSECT")
            print(self.get_tensor())
            # comment = edge.get_attributes()['comment'].strip('"')
            if self.get_tensor() == 'C':
                isect_conn = 1
            e_type = edge.get_attributes()['type'].strip('"')
            if "crd" in e_type:
                new_conns = {
                    f'rd_scan_to_isect_{isect_conn}_crd': [
                        # send output to rd scanner
                        ([(rd_scan, "coord_out"), (isect, f"coord_in_{isect_conn}")], 16),
                        ([(rd_scan, "eos_out_0"), (isect, f"eos_in_{isect_conn * 2}")], 1),
                        ([(isect, f"ready_out_{isect_conn * 2}"), (rd_scan, "ready_in_0")], 1),
                        ([(rd_scan, "valid_out_0"), (isect, f"valid_in_{isect_conn * 2}")], 1),
                    ]
                }
            elif 'ref' in e_type:
                new_conns = {
                    f'rd_scan_to_isect_{isect_conn}_pos': [
                        # send output to rd scanner
                        ([(rd_scan, "pos_out"), (isect, f"pos_in_{isect_conn}")], 16),
                        ([(rd_scan, "eos_out_1"), (isect, f"eos_in_{isect_conn * 2 + 1}")], 1),
                        ([(isect, f"ready_out_{isect_conn * 2 + 1}"), (rd_scan, "ready_in_1")], 1),
                        ([(rd_scan, "valid_out_1"), (isect, f"valid_in_{isect_conn * 2 + 1}")], 1),
                    ]
                }
            else:
                raise NotImplementedError(f'Only accept ref or crd types to intersect....you used {type}')
            # other.update_input_connections()
            return new_conns
        elif other_type == ReduceNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == LookupNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == MergeNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == RepeatNode:
            repeat = other.get_name()
            new_conns = {
                'rd_scan_to_repeat': [
                    # send output to rd scanner
                    ([(rd_scan, "pos_out"), (repeat, "proc_data_in")], 16),
                    ([(rd_scan, "eos_out_1"), (repeat, "proc_eos_in")], 1),
                    ([(repeat, "proc_ready_out"), (rd_scan, "ready_in_1")], 1),
                    ([(rd_scan, "valid_out_1"), (repeat, "proc_valid_in")], 1),
                ]
            }
            return new_conns
        elif other_type == ComputeNode:
            compute = other.get_name()
            compute_conn = 0
            print(edge)
            print("CHECKING READ TENSOR - COMPUTE")
            print(self.get_tensor())
            if self.get_tensor() == 'C':
                compute_conn = 1
            new_conns = {
                f'rd_scan_to_compute_{compute_conn}': [
                    # send output to rd scanner
                    ([(rd_scan, "coord_out"), (compute, f"data_in_{compute_conn}")], 16),
                    ([(rd_scan, "eos_out_0"), (compute, f"eos_in_{compute_conn}")], 1),
                    ([(compute, f"ready_out_{compute_conn}"), (rd_scan, "ready_in_0")], 1),
                    ([(rd_scan, "valid_out_0"), (compute, f"valid_in_{compute_conn}")], 1),
                ]
            }
            return new_conns

        elif other_type == BroadcastNode:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')
        elif other_type == RepSigGenNode:
            rsg = other.get_name()
            new_conns = {
                f'rd_scan_to_rsg': [
                    ([(rd_scan, "coord_out"), (rsg, f"base_data_in")], 16),
                    ([(rd_scan, "eos_out_0"), (rsg, f"base_eos_in")], 1),
                    ([(rsg, f"base_ready_out"), (rd_scan, "ready_in_0")], 1),
                    ([(rd_scan, "valid_out_0"), (rsg, f"base_valid_in")], 1),
                ]
            }
            return new_conns
        else:
            raise NotImplementedError(f'Cannot connect ReadScannerNode to {other_type}')

        return new_conns

    def configure(self, attributes):
        inner_offset = 0
        max_outer_dim = 0
        strides = [0]
        ranges = [1]
        if attributes['type'].strip('"') == 'fiberwrite':
            # in fiberwrite case, we are in block mode
            mode = attributes['mode'].strip('"')
            if mode == 'vals' or int(mode) != 0:
                is_root = 0
            else:
                is_root = 1
        elif attributes['type'].strip('"') == 'arrayvals':
            is_root = 0
        else:
            is_root = int(attributes['root'].strip('"') == 'true')
        do_repeat = 0
        repeat_outer = 0
        repeat_factor = 0
        if attributes['type'].strip('"') == 'arrayvals':
            stop_lvl = 0
            lookup = 1
        elif attributes['mode'].strip('"') == 'vals':
            stop_lvl = 0
            lookup = 1
        else:
            stop_lvl = int(attributes['mode'].strip('"'))
            lookup = 0
        block_mode = int(attributes['type'].strip('"') == 'fiberwrite')
        if attributes['type'].strip('"') == 'fiberwrite':
            lookup = 0

        cfg_kwargs = {
            'inner_offset': inner_offset,
            'max_out': max_outer_dim,
            'strides': strides,
            'ranges': ranges,
            'root': is_root,
            'do_repeat': do_repeat,
            'repeat_outer': repeat_outer,
            'repeat_factor': repeat_factor,
            'stop_lvl': stop_lvl,
            'block_mode': block_mode,
            'lookup': lookup
        }

        return (inner_offset, max_outer_dim, strides, ranges, is_root, do_repeat,
                repeat_outer, repeat_factor, stop_lvl, block_mode, lookup), cfg_kwargs
