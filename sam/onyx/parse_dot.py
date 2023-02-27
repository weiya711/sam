import argparse
from numpy import broadcast
import pydot
from sam.onyx.hw_nodes.hw_node import HWNodeType


class SAMDotGraphLoweringError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SAMDotGraph():

    def __init__(self, filename=None, local_mems=True, use_fork=False, use_fa=False) -> None:
        assert filename is not None, "filename is None"
        self.graphs = pydot.graph_from_dot_file(filename)
        self.graph = self.graphs[0]
        self.mode_map = {}
        self.get_mode_map()
        self.mapped_graph = {}
        self.seq = 0
        self.local_mems = local_mems
        self.use_fork = use_fork
        self.use_fa = use_fa
        self.fa_color = 0

        # Rewrite each 3-input joiners to 3 2-input joiners
        self.rewrite_tri_to_binary()
        self.rewrite_spacc_1()

        # Passes to lower to CGRA
        self.rewrite_lookup()
        self.rewrite_arrays()
        # If using real fork, we don't rewrite the rsg broadcast in the same way
        if self.use_fork:
            self.rewrite_broadcast()
        else:
            self.rewrite_rsg_broadcast()
        self.map_nodes()

    def get_mode_map(self):
        sc = self.graph.get_comment().strip('"')
        for tensor in sc.split(","):
            tensor_name = tensor.split("=")[0]
            self.mode_map[tensor_name] = {}
            tensor_format = tensor.split("=")[1]
            if tensor_format == 'none':
                continue
            for mode, tf_subspec in enumerate(tensor_format[0:len(tensor_format) // 2]):
                actual_mode = int(tensor_format[mode + len(tensor_format) // 2])
                self.mode_map[tensor_name][actual_mode] = (mode, tf_subspec)
                print(self.mode_map)
        self.mode_map_list = []
        self.tensor_list = []
        for tensor, mappings in self.mode_map.items():
            self.tensor_list.append(tensor)
            mappings_list = []
            for idx in range(len(mappings.keys())):
                mappings_list.append(mappings[idx])
            self.mode_map_list.append(mappings_list)

        self.remaining = {}

        for tensor, mappings in self.mode_map.items():
            self.remaining[tensor] = tuple(mappings.items())
        # return self.tensor_list, self.mode_map_list
        # return self.mode_map
        return self.remaining

    def map_nodes(self):
        '''
        Iterate through the nodes and map them to the proper HWNodes
        '''

        for node in self.graph.get_nodes():
            # Simple write the HWNodeType attribute
            if 'hwnode' not in node.get_attributes():
                node.create_attribute_methods(node.get_attributes())
                n_type = node.get_type().strip('"')
                assert n_type != "fiberwrite", "fiberwrite should have been rewritten out..."
                assert n_type != "fiberlookup", "fiberlookup should have been rewritten out..."
                assert n_type != "arrayvals", "arrayvals should have been rewritten out..."

                hw_nt = None
                if n_type == "broadcast":
                    hw_nt = f"HWNodeType.Broadcast"
                elif n_type == "repsiggen":
                    hw_nt = f"HWNodeType.RepSigGen"
                elif n_type == "repeat":
                    hw_nt = f"HWNodeType.Repeat"
                elif n_type == "mul" or n_type == "add":
                    hw_nt = f"HWNodeType.Compute"
                elif n_type == "reduce":
                    hw_nt = f"HWNodeType.Reduce"
                elif n_type == "intersect" or n_type == "union":
                    if n_type == "union":
                        print("UNION BLOCK")
                    hw_nt = f"HWNodeType.Intersect"
                elif n_type == "crddrop":
                    hw_nt = f"HWNodeType.Merge"
                elif n_type == "crdhold":
                    hw_nt = f"HWNodeType.CrdHold"
                elif n_type == "spaccumulator":
                    hw_nt = f"HWNodeType.SpAccumulator"
                else:
                    print(n_type)
                    raise SAMDotGraphLoweringError(f"Node is of type {n_type}")

                node.get_attributes()['hwnode'] = hw_nt

    def get_next_seq(self):
        ret = self.seq
        self.seq += 1
        return ret

    def find_node_by_name(self, name):
        for node in self.graph.get_nodes():
            if node.get_name() == name:
                return node
        assert False

    def rewrite_spacc_1(self):

        # Get the spacc node and the resulting fiberwrites
        nodes_to_proc = []
        for node in self.graph.get_nodes():
            node_type = node.get_attributes()['type'].strip('"')
            if 'spaccumulator' in node_type and '1' in node.get_attributes()['order'].strip('"'):
                # nodes_to_proc.append(node.get_name())
                nodes_to_proc.append(node)

        for spacc_node in nodes_to_proc:

            attrs = spacc_node.get_attributes()
            og_label = attrs['label'].strip('"')
            del attrs['label']

            # TODO: Get redux crd
            output_crd = attrs['in0'].strip('"')
            input_crd = None

            incoming_edges = [edge for edge in self.graph.get_edges() if edge.get_destination() == spacc_node.get_name()]
            outgoing_edges = [edge for edge in self.graph.get_edges() if edge.get_source() == spacc_node.get_name()]

            in_val_node = None
            in_output_node = None
            in_input_node = None

            # Keep these for the edges
            in_edge_attrs = {}

            # Figure out the other coordinate edge/delete the incoming edges to replace
            for incoming_edge_ in incoming_edges:
                edge_attr = incoming_edge_.get_attributes()
                if edge_attr['type'].strip('"') == 'val':
                    in_val_node = incoming_edge_.get_source()
                    in_edge_attrs[in_val_node] = edge_attr
                elif edge_attr['type'].strip('"') == 'crd':
                    edge_comment = edge_attr['comment'].strip('"')
                    if output_crd in edge_comment:
                        in_output_node = incoming_edge_.get_source()
                        in_edge_attrs[in_output_node] = edge_attr
                    else:
                        input_crd = edge_comment
                        in_input_node = incoming_edge_.get_source()
                        in_edge_attrs[in_input_node] = edge_attr
                self.graph.del_edge(incoming_edge_.get_source(), incoming_edge_.get_destination())

            # Delete the outgoing edges/attached nodes
            # output_nodes_ = []
            # Now delete the outputs of this
            # for edge_ in outgoing_edges:
            #     # output_nodes_.append(edge_.get_destination())
            #     self.graph.del_node(edge_.get_destination())
            #     # self.graph.del_edge(edge_)
            #     self.graph.del_edge(edge_.get_source(), edge_.get_destination())

            print(attrs)
            og_type = attrs['type']
            del attrs['type']

            rsg = pydot.Node(f"spacc1_rsg_{self.get_next_seq()}",
                             **attrs, label=f"{og_label}_rsg", hwnode=f"{HWNodeType.RepSigGen}",
                             type=og_type)

            repeat = pydot.Node(f"spacc1_repeat_{self.get_next_seq()}",
                                **attrs, label=f"{og_label}_repeat", hwnode=f"{HWNodeType.Repeat}",
                                root="true", type=og_type, spacc="true")

            union = pydot.Node(f"spacc1_union_{self.get_next_seq()}",
                               **attrs, label=f"{og_label}_union", hwnode=f"{HWNodeType.Intersect}",
                               type="union")

            add = pydot.Node(f"spacc1_add_{self.get_next_seq()}",
                             **attrs, label=f"{og_label}_add", hwnode=f"{HWNodeType.Compute}",
                             type=og_type)

            crd_buffet = pydot.Node(f"spacc1_crd_buffet_{self.get_next_seq()}",
                                    **attrs, label=f"{og_label}_crd_buffet", hwnode=f"{HWNodeType.Buffet}",
                                    type=og_type, fa_color=self.fa_color)

            crd_rd_scanner = pydot.Node(f"spacc1_crd_rd_scanner_{self.get_next_seq()}",
                                        **attrs, label=f"{og_label}_crd_rd_scanner", hwnode=f"{HWNodeType.ReadScanner}",
                                        tensor="x", type=og_type, root="false", format="compressed",
                                        mode="0", index=f"{output_crd}", spacc="true", stop_lvl="0",
                                        fa_color=self.fa_color)

            crd_wr_scanner = pydot.Node(f"spacc1_crd_wr_scanner_{self.get_next_seq()}",
                                        **attrs, label=f"{og_label}_crd_wr_scanner", hwnode=f"{HWNodeType.WriteScanner}",
                                        type=og_type, mode="0", format="compressed", spacc="true", stop_lvl="0",
                                        fa_color=self.fa_color)

            self.fa_color += 1

            # glb_crd = pydot.Node(f"spacc1_crd_glb_{self.get_next_seq()}", **attrs,
            #                      label=f"{og_label}_glb_crd_read", hwnode=f"{HWNodeType.GLB}",
            #                      tensor="x", mode="0", format="compressed", type=og_type)

            vals_buffet = pydot.Node(f"spacc1_vals_buffet_{self.get_next_seq()}",
                                     **attrs, label=f"{og_label}_vals_buffet", hwnode=f"{HWNodeType.Buffet}",
                                     type=og_type, fa_color=self.fa_color)

            vals_rd_scanner = pydot.Node(f"spacc1_vals_rd_scanner_{self.get_next_seq()}",
                                         **attrs, label=f"{og_label}_vals_rd_scanner", hwnode=f"{HWNodeType.ReadScanner}",
                                         tensor="x", type=og_type, root="false", format="vals",
                                         mode="vals", spacc="true", stop_lvl="0", fa_color=self.fa_color)

            vals_wr_scanner = pydot.Node(f"spacc1_vals_wr_scanner_{self.get_next_seq()}",
                                         **attrs, label=f"{og_label}_vals_wr_scanner", hwnode=f"{HWNodeType.WriteScanner}",
                                         type=og_type, mode="vals", format="compressed", spacc="true",
                                         stop_lvl="0", fa_color=self.fa_color)

            # glb_vals = pydot.Node(f"spacc1_crd_vals_{self.get_next_seq()}", **attrs,
            #                       label=f"{og_label}_glb_vals_read", hwnode=f"{HWNodeType.GLB}",
            #                       tensor="x", mode="vals", format="vals", type=og_type)

            self.fa_color += 1

            self.graph.add_node(rsg)
            self.graph.add_node(repeat)
            self.graph.add_node(union)
            self.graph.add_node(add)
            self.graph.add_node(crd_buffet)
            self.graph.add_node(crd_rd_scanner)
            self.graph.add_node(crd_wr_scanner)
            # self.graph.add_node(glb_crd)
            self.graph.add_node(vals_buffet)
            self.graph.add_node(vals_rd_scanner)
            self.graph.add_node(vals_wr_scanner)
            # self.graph.add_node(glb_vals)

            print(in_edge_attrs[in_input_node])
            print(in_edge_attrs[in_output_node])
            print(in_edge_attrs[in_val_node])

            del in_edge_attrs[in_output_node]['comment']
            del in_edge_attrs[in_val_node]['type']

            # Edges
            input_to_rsg_edge = pydot.Edge(src=in_input_node, dst=rsg, **in_edge_attrs[in_input_node])
            rsg_to_repeat = pydot.Edge(src=rsg, dst=repeat)
            repeat_to_crd_rd_scan = pydot.Edge(src=repeat, dst=crd_rd_scanner)
            crd_rd_scan_to_val_rd_scan = pydot.Edge(src=crd_rd_scanner, dst=vals_rd_scanner)
            output_to_union_edge = pydot.Edge(src=in_output_node, dst=union, **in_edge_attrs[in_output_node], comment=f"in-B")
            val_to_union = pydot.Edge(src=in_val_node, dst=union, **in_edge_attrs[in_val_node],
                                      type="ref", comment=f"in-B", val="true")
            crd_rd_scan_to_union = pydot.Edge(src=crd_rd_scanner, dst=union, type="crd", comment="in-x")
            val_rd_scan_to_union = pydot.Edge(src=vals_rd_scanner, dst=union, type="ref", comment="in-x", val="true")
            union_crd_to_crd_wr_scan = pydot.Edge(src=union, dst=crd_wr_scanner, type="crd")
            union_val0_to_alu = pydot.Edge(src=union, dst=add, comment='out-B')
            union_val1_to_alu = pydot.Edge(src=union, dst=add, comment='out-x')
            add_to_val_wr_scan = pydot.Edge(src=add, dst=vals_wr_scanner)
            crd_wr_scan_to_buffet = pydot.Edge(src=crd_wr_scanner, dst=crd_buffet)
            val_wr_scan_to_buffet = pydot.Edge(src=vals_wr_scanner, dst=vals_buffet)
            crd_rd_scan_to_buffet = pydot.Edge(src=crd_rd_scanner, dst=crd_buffet)
            vals_rd_scan_to_buffet = pydot.Edge(src=vals_rd_scanner, dst=vals_buffet)

            # Match the crd/vals outputs
            crd_edge = [edge_ for edge_ in outgoing_edges if
                        self.find_node_by_name(edge_.get_destination()).get_attributes()['mode'].strip('"') != "vals"][0]
            val_edge = [edge_ for edge_ in outgoing_edges if
                        self.find_node_by_name(edge_.get_destination()).get_attributes()['mode'].strip('"') == "vals"][0]
            dst_crd = crd_edge.get_destination()
            dst_vals = val_edge.get_destination()

            crd_edge_attr = crd_edge.get_attributes()
            val_edge_attr = val_edge.get_attributes()

            self.graph.del_edge(crd_edge.get_source(), crd_edge.get_destination())
            self.graph.del_edge(val_edge.get_source(), val_edge.get_destination())

            crd_rd_scan_to_glb = pydot.Edge(src=crd_rd_scanner, dst=dst_crd, **crd_edge_attr, use_alt_out_port="1")
            val_rd_scan_to_glb = pydot.Edge(src=vals_rd_scanner, dst=dst_vals, **val_edge_attr, use_alt_out_port="1")

            self.graph.add_edge(input_to_rsg_edge)
            self.graph.add_edge(rsg_to_repeat)
            self.graph.add_edge(repeat_to_crd_rd_scan)
            self.graph.add_edge(crd_rd_scan_to_val_rd_scan)
            self.graph.add_edge(output_to_union_edge)
            self.graph.add_edge(val_to_union)
            self.graph.add_edge(crd_rd_scan_to_union)
            self.graph.add_edge(val_rd_scan_to_union)
            self.graph.add_edge(union_crd_to_crd_wr_scan)
            self.graph.add_edge(union_val0_to_alu)
            self.graph.add_edge(union_val1_to_alu)
            self.graph.add_edge(add_to_val_wr_scan)
            self.graph.add_edge(crd_wr_scan_to_buffet)
            self.graph.add_edge(val_wr_scan_to_buffet)
            self.graph.add_edge(crd_rd_scan_to_buffet)
            self.graph.add_edge(vals_rd_scan_to_buffet)
            self.graph.add_edge(crd_rd_scan_to_glb)
            self.graph.add_edge(val_rd_scan_to_glb)

            self.graph.del_node(spacc_node)

    def rewrite_tri_to_binary(self):
        '''
        Rewrites any 3-input joiner node into three binary joiner nodes
        '''
        joiner_ninputs = dict()
        nodes_to_proc = []
        for node in self.graph.get_nodes():
            if "intersect" in node.get_attributes()['type'].strip('"') or \
                    "union" in node.get_attributes()['type'].strip('"'):
                joiner_ninputs[node.get_name()] = 0
                nodes_to_proc.append(node)
        for edge in self.graph.get_edges():
            if edge.get_destination() in joiner_ninputs and "crd" in edge.get_attributes()['type'].strip('"'):
                joiner_ninputs[edge.get_destination()] += 1

        # Only tri_to_binary implemented. Not n_to_binary...
        assert all([v <= 3 for k, v in joiner_ninputs.items()])

        nodes_to_proc = [n for n in nodes_to_proc if joiner_ninputs[n.get_name()] == 3]
        print("NODES TO REWRITE FOR BINARY", [n.get_name() for n in nodes_to_proc])

        for node in nodes_to_proc:
            attrs = node.get_attributes()
            og_label = attrs['label']
            del attrs['label']
            name = "intersect" if "intersect" in attrs['type'].strip('"') else "union"

            joiner12 = pydot.Node(f"{name}_{self.get_next_seq()}",
                                  **attrs, label=f"{og_label}_12")
            joiner13 = pydot.Node(f"{name}_{self.get_next_seq()}",
                                  **attrs, label=f"{og_label}_13")
            joiner23 = pydot.Node(f"{name}_{self.get_next_seq()}",
                                  **attrs, label=f"{og_label}_23")

            broadcast3_crd = pydot.Node(f"broadcast_crd_{self.get_next_seq()}",
                                        label=f"broadcast_{og_label}_3", type=f"broadcast", comment=f"broadcast")
            broadcast3_ref = pydot.Node(f"broadcast_ref_{self.get_next_seq()}",
                                        label=f"broadcast_{og_label}_3", type=f"broadcast", comment=f"broadcast")

            input_crd_edges = dict()
            input_ref_edges = dict()
            output_ref_edges = dict()
            output_crd_edge = 0
            for edge in self.graph.get_edges():
                if edge.get_destination() == node.get_name():
                    edge_name = edge.get_attributes()['comment'].strip('"').split('-')[1]
                    edge_type = edge.get_attributes()['type'].strip('"')
                    if 'crd' in edge_type:
                        input_crd_edges[edge_name] = edge
                    elif 'ref' in edge_type:
                        input_ref_edges[edge_name] = edge
                elif edge.get_source() == node.get_name():
                    edge_type = edge.get_attributes()['type'].strip('"')
                    if 'crd' in edge_type:
                        output_crd_edge = edge
                    elif 'ref' in edge_type:
                        edge_name = edge.get_attributes()['comment'].strip('"').split('-')[1]
                        output_ref_edges[edge_name] = edge

            # Add in the new joiner nodes
            self.graph.add_node(joiner12)
            self.graph.add_node(joiner23)
            self.graph.add_node(joiner13)
            self.graph.add_node(broadcast3_crd)
            self.graph.add_node(broadcast3_ref)

            # Rewire the edges
            assert set(input_crd_edges.keys()) == set(input_ref_edges.keys()) and set(input_ref_edges.keys()) == set(
                output_ref_edges.keys())
            keys = sorted(input_crd_edges.keys())

            joiner12_crd1_edge_tmp = pydot.Edge(src=input_crd_edges[keys[0]].get_source(), dst=joiner12,
                                                **input_crd_edges[keys[0]].get_attributes())
            joiner12_ref1_edge_tmp = pydot.Edge(src=input_ref_edges[keys[0]].get_source(), dst=joiner12,
                                                **input_ref_edges[keys[0]].get_attributes())
            joiner12_crd2_edge_tmp = pydot.Edge(src=input_crd_edges[keys[1]].get_source(), dst=joiner12,
                                                **input_crd_edges[keys[1]].get_attributes())
            joiner12_ref2_edge_tmp = pydot.Edge(src=input_ref_edges[keys[1]].get_source(), dst=joiner12,
                                                **input_ref_edges[keys[1]].get_attributes())

            self.graph.add_edge(joiner12_crd1_edge_tmp)
            self.graph.add_edge(joiner12_ref1_edge_tmp)
            self.graph.add_edge(joiner12_crd2_edge_tmp)
            self.graph.add_edge(joiner12_ref2_edge_tmp)

            broadcast3_input_crd = pydot.Edge(src=input_crd_edges[keys[2]].get_source(), dst=broadcast3_crd,
                                              label=f"crd")
            broadcast3_input_ref = pydot.Edge(src=input_ref_edges[keys[2]].get_source(), dst=broadcast3_ref,
                                              label=f"ref")

            joiner13_crd1_edge_tmp = pydot.Edge(src=joiner12, dst=joiner13,
                                                **input_crd_edges[keys[0]].get_attributes())
            joiner13_ref1_edge_tmp = pydot.Edge(src=joiner12, dst=joiner13,
                                                **input_ref_edges[keys[0]].get_attributes())

            joiner13_crd2_edge_tmp = pydot.Edge(src=broadcast3_crd, dst=joiner13,
                                                **input_crd_edges[keys[2]].get_attributes())
            joiner13_ref2_edge_tmp = pydot.Edge(src=broadcast3_ref, dst=joiner13,
                                                **input_ref_edges[keys[2]].get_attributes())

            joiner13_ref1_out_edge_tmp = pydot.Edge(src=joiner13, dst=output_ref_edges[keys[0]].get_destination(),
                                                    **output_ref_edges[keys[0]].get_attributes())
            joiner13_ref2_out_edge_tmp = pydot.Edge(src=joiner13, dst=output_ref_edges[keys[2]].get_destination(),
                                                    **output_ref_edges[keys[2]].get_attributes())
            joiner13_crd_out_edge_tmp = pydot.Edge(src=joiner13, dst=output_crd_edge.get_destination(),
                                                   **output_crd_edge.get_attributes())

            self.graph.add_edge(broadcast3_input_crd)
            self.graph.add_edge(broadcast3_input_ref)
            self.graph.add_edge(joiner13_crd1_edge_tmp)
            self.graph.add_edge(joiner13_ref1_edge_tmp)
            self.graph.add_edge(joiner13_crd2_edge_tmp)
            self.graph.add_edge(joiner13_ref2_edge_tmp)
            self.graph.add_edge(joiner13_ref1_out_edge_tmp)
            self.graph.add_edge(joiner13_ref2_out_edge_tmp)
            self.graph.add_edge(joiner13_crd_out_edge_tmp)

            joiner23_crd1_edge_tmp = pydot.Edge(src=joiner12, dst=joiner23,
                                                **input_crd_edges[keys[1]].get_attributes())
            joiner23_ref1_edge_tmp = pydot.Edge(src=joiner12, dst=joiner23,
                                                **input_ref_edges[keys[1]].get_attributes())

            joiner23_crd2_edge_tmp = pydot.Edge(src=broadcast3_crd, dst=joiner23,
                                                **input_crd_edges[keys[2]].get_attributes())
            joiner23_ref2_edge_tmp = pydot.Edge(src=broadcast3_ref, dst=joiner23,
                                                **input_ref_edges[keys[2]].get_attributes())

            joiner23_ref1_out_edge_tmp = pydot.Edge(src=joiner23, dst=output_ref_edges[keys[1]].get_destination(),
                                                    **output_ref_edges[keys[1]].get_attributes())
            # joiner23_ref2_out_edge_tmp = pydot.Edge(src=joiner23, dst=output_ref_edges[keys[2]].get_destination(),
            #                                         **output_ref_edges[keys[2]].get_attributes())

            self.graph.add_edge(joiner23_crd1_edge_tmp)
            self.graph.add_edge(joiner23_ref1_edge_tmp)
            self.graph.add_edge(joiner23_crd2_edge_tmp)
            self.graph.add_edge(joiner23_ref2_edge_tmp)
            self.graph.add_edge(joiner23_ref1_out_edge_tmp)
            # self.graph.add_edge(joiner23_ref2_out_edge_tmp)

            # Delete original edges and nodes
            for k, v in input_crd_edges.items():
                self.graph.del_edge(v.get_source(), v.get_destination())
            for k, v in input_ref_edges.items():
                self.graph.del_edge(v.get_source(), v.get_destination())
            for k, v in output_ref_edges.items():
                self.graph.del_edge(v.get_source(), v.get_destination())
            self.graph.del_edge(output_crd_edge.get_source(), output_crd_edge.get_destination())
            self.graph.del_node(node)

    def rewrite_n_to_binary(self):
        raise NotImplementedError

    def rewrite_broadcast(self):
        '''
        Rewrites the broadcast going into an RSG and makes it n or more separate connections
        '''
        nodes_to_proc = []
        for node in self.graph.get_nodes():
            if 'broadcast' in node.get_attributes()['type'].strip('"'):
                nodes_to_proc.append(node.get_name())
        # Now we have the broadcast node - want to find the incoming edge and redirect to the destinations
        for broadcast_node in nodes_to_proc:
            # broadcast_node = self.graph.get_node(broadcast_node)
            attrs = node.get_attributes()
            og_label = attrs['label']
            # del attrs['label']
            # Find the upstream broadcast node
            in_src = None
            in_edge = None
            out_dsts = []
            out_edges = []
            for edge in self.graph.get_edges():
                source_node = edge.get_source()
                # source_node = self.graph.get_node(source_node)[0]
                # If this is the destination, mark the src node
                # if broadcast_node == self.graph.get_node(edge.get_destination())[0]:
                if broadcast_node == edge.get_destination():
                    in_src = source_node
                    in_edge = edge
                # If this is the src, mark the destination node
                elif broadcast_node == source_node:
                    out_dsts.append(self.graph.get_node(edge.get_destination())[0])
                    out_edges.append(edge)

            # Replace the edges
            for i in range(len(out_edges)):
                # Here we need to copy and replace the edges
                tmp_edge = out_edges[i]
                fork_edge_tmp = pydot.Edge(src=in_src, dst=tmp_edge.get_destination(), **tmp_edge.get_attributes())
                self.graph.del_edge(tmp_edge.get_source(), tmp_edge.get_destination())
                self.graph.add_edge(fork_edge_tmp)

            # Now delete the broadcast and all original edge
            ret = self.graph.del_edge(in_edge.get_source(), in_edge.get_destination())
            ret = self.graph.del_node(broadcast_node)

    def rewrite_rsg_broadcast(self):
        '''
        Rewrites the broadcast going into an RSG and uses it with passthru
        '''
        nodes_to_proc = []
        for node in self.graph.get_nodes():
            if 'repsiggen' in node.get_attributes()['type'].strip('"'):
                nodes_to_proc.append(node)
        # Now we have the rep sig gen - want to find the incoming edge from the broadcast node
        # then rip that out and wire the original edge to the rsg and the edge to the other guy from rsg to it
        for rsg_node in nodes_to_proc:
            attrs = node.get_attributes()
            og_label = attrs['label']
            # del attrs['label']
            # Find the upstream broadcast node
            broadcast_nodes = []
            for edge in self.graph.get_edges():
                source_node = edge.get_source()
                source_node = self.graph.get_node(source_node)[0]
                if "broadcast" in source_node.get_attributes()['type'].strip('"') and \
                        edge.get_destination() == rsg_node.get_name():
                    broadcast_nodes.append(source_node)

            # Leave early.
            if len(broadcast_nodes) == 0:
                return
            bc_node = broadcast_nodes[0]
            # Now that we have the broadcast node, get the incoming edge and other outgoing edge
            incoming_edge = [edge for edge in self.graph.get_edges() if edge.get_destination() == bc_node.get_name()][0]
            outgoing_edge = [edge for edge in self.graph.get_edges() if edge.get_source() == bc_node.get_name() and
                             edge.get_destination() == rsg_node.get_name()][0]
            other_outgoing_edge = \
                [edge for edge in self.graph.get_edges() if edge.get_source() == bc_node.get_name() and
                 edge.get_destination() != rsg_node.get_name()][0]
            # Now, connect the original source to the rsg and the rsg to the original other branch
            og_to_rsg = pydot.Edge(src=incoming_edge.get_source(), dst=rsg_node, **incoming_edge.get_attributes())
            rsg_to_branch = pydot.Edge(src=rsg_node, dst=other_outgoing_edge.get_destination(),
                                       **other_outgoing_edge.get_attributes())

            # Now delete the broadcast and all original edge
            ret = self.graph.del_edge(incoming_edge.get_source(), incoming_edge.get_destination())
            ret = self.graph.del_edge(outgoing_edge.get_source(), outgoing_edge.get_destination())
            ret = self.graph.del_edge(other_outgoing_edge.get_source(), other_outgoing_edge.get_destination())
            ret = self.graph.del_node(bc_node)

            # ...and add in the new edges
            self.graph.add_edge(og_to_rsg)
            self.graph.add_edge(rsg_to_branch)

    def rewrite_lookup(self):
        '''
        Rewrites the lookup nodes to become (wr_scan, rd_scan, buffet) triples
        '''
        nodes_to_proc = [node for node in self.graph.get_nodes() if 'fiberlookup' in node.get_comment() or
                         'fiberwrite' in node.get_comment()]

        for node in nodes_to_proc:
            if 'fiberlookup' in node.get_comment():
                # Rewrite this node to a read
                root = bool(node.get_root())
                root = False
                if 'true' in node.get_root():
                    root = True
                attrs = node.get_attributes()
                og_label = attrs['label']
                del attrs['label']
                rd_scan = pydot.Node(f"rd_scan_{self.get_next_seq()}",
                                     **attrs, label=f"{og_label}_rd_scan", hwnode=f"{HWNodeType.ReadScanner}",
                                     fa_color=self.fa_color)
                wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}",
                                     **attrs, label=f"{og_label}_wr_scan", hwnode=f"{HWNodeType.WriteScanner}",
                                     fa_color=self.fa_color)
                buffet = pydot.Node(f"buffet_{self.get_next_seq()}",
                                    **attrs, label=f"{og_label}_buffet", hwnode=f"{HWNodeType.Buffet}",
                                    fa_color=self.fa_color)

                self.fa_color += 1

                glb_write = pydot.Node(f"glb_write_{self.get_next_seq()}",
                                       **attrs, label=f"{og_label}_glb_write", hwnode=f"{HWNodeType.GLB}")
                if self.local_mems is False:
                    memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs,
                                        label=f"{og_label}_SRAM", hwnode=f"{HWNodeType.Memory}")

                crd_out_edge = [edge for edge in self.graph.get_edges() if edge.get_source() == node.get_name() and
                                "crd" in edge.get_label()][0]
                ref_out_edge = [edge for edge in self.graph.get_edges() if edge.get_source() == node.get_name() and
                                "ref" in edge.get_label()][0]
                ref_in_edge = None
                if not root:
                    # Then we have ref in edge...
                    ref_in_edge = [edge for edge in self.graph.get_edges() if edge.get_destination() == node.get_name() and
                                   "ref" in edge.get_label()][0]
                # Now add the nodes and move the edges...
                self.graph.add_node(rd_scan)
                self.graph.add_node(wr_scan)
                self.graph.add_node(buffet)
                self.graph.add_node(glb_write)
                if self.local_mems is False:
                    self.graph.add_node(memory)
                # Glb to WR
                glb_to_wr = pydot.Edge(src=glb_write, dst=wr_scan, label=f"glb_to_wr_{self.get_next_seq()}",
                                       style="bold")
                self.graph.add_edge(glb_to_wr)
                # write + read to buffet
                wr_to_buff = pydot.Edge(src=wr_scan, dst=buffet, label=f'wr_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(wr_to_buff)
                rd_to_buff = pydot.Edge(src=rd_scan, dst=buffet, label=f'rd_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(rd_to_buff)
                if self.local_mems is False:
                    # Mem to buffet
                    mem_to_buff = pydot.Edge(src=buffet, dst=memory, label=f'mem_to_buff_{self.get_next_seq()}')
                    self.graph.add_edge(mem_to_buff)
                # Now inject the read scanner to other nodes...
                rd_to_down_crd = pydot.Edge(src=rd_scan, dst=crd_out_edge.get_destination(),
                                            **crd_out_edge.get_attributes())
                rd_to_down_ref = pydot.Edge(src=rd_scan, dst=ref_out_edge.get_destination(),
                                            **ref_out_edge.get_attributes())
                self.graph.add_edge(rd_to_down_crd)
                self.graph.add_edge(rd_to_down_ref)
                if ref_in_edge is not None:
                    up_to_ref = pydot.Edge(src=ref_in_edge.get_source(), dst=rd_scan, **ref_in_edge.get_attributes())
                    self.graph.add_edge(up_to_ref)

                # Delte old stuff...
                ret = self.graph.del_node(node)
                ret = self.graph.del_edge(crd_out_edge.get_source(), crd_out_edge.get_destination())
                self.graph.del_edge(ref_out_edge.get_source(), ref_out_edge.get_destination())
                if ref_in_edge is not None:
                    self.graph.del_edge(ref_in_edge.get_source(), ref_in_edge.get_destination())

            elif 'fiberwrite' in node.get_comment():
                # Rewrite this node to a write
                # root = 'root' in node.get_name()
                attrs = node.get_attributes()
                node.create_attribute_methods(attrs)
                og_label = attrs['label']
                del attrs['label']
                rd_scan = pydot.Node(f"rd_scan_{self.get_next_seq()}", **attrs,
                                     label=f"{og_label}_rd_scan", hwnode=f"{HWNodeType.ReadScanner}",
                                     fa_color=self.fa_color)
                wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}", **attrs,
                                     label=f"{og_label}_wr_scan", hwnode=f"{HWNodeType.WriteScanner}",
                                     fa_color=self.fa_color)
                buffet = pydot.Node(f"buffet_{self.get_next_seq()}", **attrs,
                                    label=f"{og_label}_buffet", hwnode=f"{HWNodeType.Buffet}",
                                    fa_color=self.fa_color)

                self.fa_color += 1

                glb_read = pydot.Node(f"glb_read_{self.get_next_seq()}", **attrs,
                                      label=f"{og_label}_glb_read", hwnode=f"{HWNodeType.GLB}")
                if self.local_mems is False:
                    memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs,
                                        label=f"{og_label}_SRAM", hwnode=f"{HWNodeType.Memory}")
                vals = 'vals' in node.get_mode()
                in_edge = None
                if vals:
                    in_edge = [edge for edge in self.graph.get_edges() if edge.get_destination() == node.get_name() and
                               "val" in edge.get_label()][0]
                else:
                    in_edge = [edge for edge in self.graph.get_edges() if edge.get_destination() == node.get_name() and
                               "crd" in edge.get_label()][0]

                # Now add the nodes and move the edges...
                self.graph.add_node(rd_scan)
                self.graph.add_node(wr_scan)
                self.graph.add_node(buffet)
                self.graph.add_node(glb_read)
                if self.local_mems is False:
                    self.graph.add_node(memory)
                # RD to GLB
                rd_to_glb = pydot.Edge(src=rd_scan, dst=glb_read, label=f"glb_to_wr_{self.get_next_seq()}",
                                       style="bold")
                self.graph.add_edge(rd_to_glb)
                # write + read to buffet
                wr_to_buff = pydot.Edge(src=wr_scan, dst=buffet, label=f'wr_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(wr_to_buff)
                rd_to_buff = pydot.Edge(src=rd_scan, dst=buffet, label=f'rd_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(rd_to_buff)
                if self.local_mems is False:
                    # Mem to buffet
                    mem_to_buff = pydot.Edge(src=buffet, dst=memory, label=f'mem_to_buff_{self.get_next_seq()}')
                    self.graph.add_edge(mem_to_buff)
                # Now inject the write scanner to other nodes...
                up_to_wr = pydot.Edge(src=in_edge.get_source(), dst=wr_scan, **in_edge.get_attributes())
                self.graph.add_edge(up_to_wr)

                # Delte old stuff...
                self.graph.del_node(node)
                self.graph.del_edge(in_edge.get_source(), in_edge.get_destination())

    def rewrite_arrays(self):
        '''
        Rewrites the array nodes to become (lookup, buffet) triples
        '''
        nodes_to_proc = [node for node in self.graph.get_nodes() if 'arrayvals' in node.get_comment()]
        for node in nodes_to_proc:
            # Now we have arrayvals, let's turn it into same stuff
            # Rewrite this node to a read
            attrs = node.get_attributes()
            og_label = attrs['label']
            del attrs['label']
            rd_scan = pydot.Node(f"rd_scan_{self.get_next_seq()}",
                                 **attrs, label=f"{og_label}_rd_scan", hwnode=f"{HWNodeType.ReadScanner}",
                                 fa_color=self.fa_color)
            wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}",
                                 **attrs, label=f"{og_label}_wr_scan", hwnode=f"{HWNodeType.WriteScanner}",
                                 fa_color=self.fa_color)
            buffet = pydot.Node(f"buffet_{self.get_next_seq()}",
                                **attrs, label=f"{og_label}_buffet", hwnode=f"{HWNodeType.Buffet}",
                                fa_color=self.fa_color)

            self.fa_color += 1

            glb_write = pydot.Node(f"glb_write_{self.get_next_seq()}",
                                   **attrs, label=f"{og_label}_glb_write", hwnode=f"{HWNodeType.GLB}")
            if self.local_mems is False:
                memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs,
                                    label=f"{og_label}_SRAM", hwnode=f"{HWNodeType.Memory}")
            val_out_edge = [edge for edge in self.graph.get_edges() if edge.get_source() == node.get_name() and
                            "val" in edge.get_label()][0]
            # Then we have ref in edge...
            ref_in_edge = [edge for edge in self.graph.get_edges() if edge.get_destination() == node.get_name() and
                           "ref" in edge.get_label()][0]
            # Now add the nodes and move the edges...
            self.graph.add_node(rd_scan)
            self.graph.add_node(wr_scan)
            self.graph.add_node(buffet)
            self.graph.add_node(glb_write)
            if self.local_mems is False:
                self.graph.add_node(memory)
            # Glb to WR
            glb_to_wr = pydot.Edge(src=glb_write, dst=wr_scan, label=f"glb_to_wr_{self.get_next_seq()}", style="bold")
            self.graph.add_edge(glb_to_wr)
            # write + read to buffet
            wr_to_buff = pydot.Edge(src=wr_scan, dst=buffet, label=f'wr_to_buff_{self.get_next_seq()}')
            self.graph.add_edge(wr_to_buff)
            rd_to_buff = pydot.Edge(src=rd_scan, dst=buffet, label=f'rd_to_buff_{self.get_next_seq()}')
            self.graph.add_edge(rd_to_buff)
            if self.local_mems is False:
                # Mem to buffet
                mem_to_buff = pydot.Edge(src=buffet, dst=memory, label=f'mem_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(mem_to_buff)
            # Now inject the read scanner to other nodes...
            # rd_to_down_crd = pydot.Edge(src=rd_scan, dst=crd_out_edge.get_destination(), **crd_out_edge.get_attributes())
            rd_to_down_val = pydot.Edge(src=rd_scan, dst=val_out_edge.get_destination(),
                                        **val_out_edge.get_attributes())
            self.graph.add_edge(rd_to_down_val)
            up_to_ref = pydot.Edge(src=ref_in_edge.get_source(), dst=rd_scan, **ref_in_edge.get_attributes())
            self.graph.add_edge(up_to_ref)

            # Delte old stuff...
            ret = self.graph.del_node(node)
            ret = self.graph.del_edge(val_out_edge.get_source(), val_out_edge.get_destination())
            self.graph.del_edge(ref_in_edge.get_source(), ref_in_edge.get_destination())

    def get_graph(self):
        return self.graph


def parse_graph(graph):
    type_cnt = {}
    hwnode_cnt = {}
    for node in graph.get_nodes():
        attr = node.get_attributes()
        # print(node)
        prim_type = attr['type']
        if prim_type not in type_cnt:
            type_cnt[prim_type] = 1
        else:
            type_cnt[prim_type] += 1

        prim_hwnode = attr['hwnode']
        if prim_type not in prim_hwnode:
            hwnode_cnt[prim_type] = 1
        else:
            hwnode_cnt[prim_type] += 1

    print("type", type_cnt)
    print("hwnode", hwnode_cnt)
    return type_cnt, hwnode_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM DOT Parser')
    parser.add_argument('--sam_graph',
                        type=str,
                        default="/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/mat_identity.gv")
    parser.add_argument('--output_png',
                        type=str,
                        default="output.png")
    parser.add_argument('--output_graph',
                        type=str,
                        default="/home/max/Documents/SPARSE/sam/mek.gv")
    args = parser.parse_args()

    sam_graph = args.sam_graph
    output_png = args.output_png
    output_graph = args.output_graph
    sdg = SAMDotGraph(filename=sam_graph, use_fork=True)
    graph = sdg.get_graph()
    print(graph)
    # parse_graph(graph)
    graph.write_png(output_png)
    output_graphviz = graph.create_dot()
    graph.write_dot(output_graph)
