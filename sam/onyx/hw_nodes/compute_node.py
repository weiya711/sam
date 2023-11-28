from sam.onyx.hw_nodes.hw_node import *
from lassen.utils import float2bfbin


class ComputeNode(HWNode):
    def __init__(self, name=None, op=None) -> None:
        super().__init__(name=name)
        self.num_inputs = 2
        self.num_outputs = 1
        self.num_inputs_connected = 0
        self.num_outputs_connected = 0
        self.op = op

    def connect(self, other, edge, kwargs=None):

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
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

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
                    ([(pe, "res"), (rd_scan, "us_pos_in")], 17),
                ]
            }
            return new_conns
        elif other_type == WriteScannerNode:
            wr_scan = other.get_name()
            pe = self.get_name()
            new_conns = {
                'pe_to_wr_scan': [
                    ([(pe, "res"), (wr_scan, "data_in")], 17),
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

            if 'vector_reduce_mode' in edge.get_attributes():
                if edge.get_attributes()['vector_reduce_mode']:
                    isect_conn = 0
            else:
                if 'tensor' not in edge.get_attributes():
                    # Taking some liberties here - but technically this is the combo val
                    # isect_conn = other.get_connection_from_tensor('B')
                    isect_conn = other.get_connection_from_tensor('C')
                else:
                    isect_conn = other.get_connection_from_tensor(edge.get_tensor())

            new_conns = {
                f'pe_to_isect_{in_str}_{isect_conn}': [
                    ([(pe, "res"), (isect, f"{in_str}_{isect_conn}")], 17),
                ]
            }
            other.update_input_connections()
            return new_conns
        elif other_type == ReduceNode:
            other_red = other.get_name()
            pe = self.get_name()
            new_conns = {
                f'pe_to_reduce': [
                    ([(pe, "res"), (other_red, f"reduce_data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == LookupNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == MergeNode:
            # TODO
            # raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
            # Hack just use inner for now
            crddrop = other.get_name()
            pe = self.get_name()
            conn = 0
            if 'outer' in edge.get_comment():
                conn = 1

            new_conns = {
                f'pe_to_crddrop_res_to_{conn}': [
                    ([(pe, "res"), (crddrop, f"cmrg_coord_in_{conn}")], 17),
                ]
            }
            return new_conns

        elif other_type == RepeatNode:
            # TODO
            raise NotImplementedError(f'Cannot connect ComputeNode to {other_type}')
        elif other_type == ComputeNode:
            other_pe = other.get_name()
            other_conn = other.get_num_inputs()
            pe = self.get_name()
            # TODO: remove hack eventually
            if 'Max' in other.op:
                other_conn = 1
            else:
                other_conn = other.get_num_inputs()
            new_conns = {
                f'pe_to_pe_{other_conn}': [
                    ([(pe, "res"), (other_pe, f"data{other_conn}")], 17),
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
        elif other_type == FiberAccessNode:
            print("COMPUTE TO FIBER ACCESS")
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            print(kwargs)
            init_conns = self.connect(that_flavor, edge)
            print(init_conns)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns
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
        comment = attributes['comment'].strip('"')
        print(c_op)
        op_code = 0
        # configuring via sam, it is a sparse app
        use_dense = False
        # mapping to pe only, configuring only the pe, ignore the reduce
        pe_only = True
        # data I/O should interface with other primitive outside of the cluster
        pe_in_external = 1
        if c_op == 'mul':
            op_code = 1
        elif c_op == 'add' and 'sub=1' not in comment:
            op_code = 0
        elif c_op == 'add' and 'sub=1' in comment:
            op_code = 2
        elif c_op == 'max':
            op_code = 4
        elif c_op == 'and':
            op_code = 5
        elif c_op == 'fp_mul':
            op_code = 6
        elif c_op == 'fgetfint':
            op_code = 7
        elif c_op == 'fgetffrac':
            op_code = 8
        elif c_op == 'faddiexp':
            op_code = 9

        rb_const = None
        if "rb_const" in attributes:
            # the b operand of the op is a constant
            rb_const = attributes["rb_const"].strip('"')
            if "." in rb_const:
                # constant is a floating point 
                rb_const = float(rb_const)
                rb_const = int(float2bfbin(rb_const), 2)
            else:
                # it is a int
                rb_const = int(rb_const)

        cfg_kwargs = {
            'op': op_code,
            'use_dense': use_dense,
            'pe_only': pe_only,
            'pe_in_external': pe_in_external,
            'rb_const': rb_const
        }
        return (op_code, use_dense, pe_only, pe_in_external, rb_const), cfg_kwargs
