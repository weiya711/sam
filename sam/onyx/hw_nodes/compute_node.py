from sam.onyx.hw_nodes.hw_node import *
from lassen.utils import float2bfbin
import coreir
import subprocess
import json
import os


class ComputeNode(HWNode):
    def __init__(self, name=None, op=None, sam_graph_node_id=None) -> None:
        super().__init__(name=name)
        self.num_inputs = 2
        self.num_outputs = 1
        self.num_inputs_connected = 0
        self.num_outputs_connected = 0
        self.mapped_input_ports = []
        self.op = op
        self.opcode = None
        # parse the mapped coreir file to get the input ports and opcode
        self.parse_mapped_json("/aha/alu_mapped.json", sam_graph_node_id)
        assert len(self.mapped_input_ports) > 0
        assert self.opcode is not None

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
            if 'Max 0' in other.op:
                other_conn = 1
            elif 'Faddiexp' in other.op:
                comment = edge.get_attributes()["comment"].strip('"')
                if 'fp' in comment:
                    other_conn = 0
                elif 'exp' in comment:
                    other_conn = 1
                else:
                    assert 0 & "edge connected to faddiexp has to have comment specified to either 'exp' or 'fp'"
            else: 
                other_conn = other.mapped_input_ports[other_conn]
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

    def parse_mapped_json(self, filename, node_id):
        with open(filename, 'r') as alu_mapped_file:
            alu_mapped = json.load(alu_mapped_file)
        # parse out the mapped opcode
        opcode = alu_mapped["namespaces"]["global"]["modules"]["ALU_" + node_id + "_mapped"]["instances"]["c0"]["modargs"]["value"][1]
        opcode = "0x" + opcode.split('h')[1]
        # parse out the mapped input ports
        for connection in alu_mapped["namespaces"]["global"]["modules"]["ALU_" + node_id + "_mapped"]["connections"]:
            src, dest = connection
            # if the connection is to the data port of alu
            if "self.in" in src:
                # get the port name of the alu
                self.mapped_input_ports.append(dest.split(".")[1].strip("data"))
        self.opcode = int(opcode, 0)

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
        # according to the mapped input ports generate input port config
        num_sparse_inputs = list("000")
        for port in self.mapped_input_ports:
            num_sparse_inputs[2 - int(port)] = '1'
        print("".join(num_sparse_inputs))
        num_sparse_inputs = int("".join(num_sparse_inputs), 2)

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
            'op': self.opcode,
            'use_dense': use_dense,
            'pe_only': pe_only,
            'pe_in_external': pe_in_external,
            'rb_const': rb_const,
            'num_sparse_inputs': num_sparse_inputs
        }
        return (op_code, use_dense, pe_only, pe_in_external, rb_const, num_sparse_inputs), cfg_kwargs
