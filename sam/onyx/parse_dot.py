import argparse
from numpy import broadcast
import pydot
from sam.onyx.hw_nodes.hw_node import HWNodeType


class SAMDotGraphLoweringError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SAMDotGraph():

    def __init__(self, filename=None, local_mems=True, use_fork=False) -> None:
        assert filename is not None, "filename is None"
        self.graphs = pydot.graph_from_dot_file(filename)
        self.graph = self.graphs[0]
        self.mode_map = {}
        self.get_mode_map()
        self.mapped_graph = {}
        self.seq = 0
        self.local_mems = local_mems
        self.use_fork = use_fork

        # Rewrite each 3-input joiners to 3 2-input joiners
        self.rewrite_tri_to_binary()

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
                                     **attrs, label=f"{og_label}_rd_scan", hwnode=f"{HWNodeType.ReadScanner}")
                wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}",
                                     **attrs, label=f"{og_label}_wr_scan", hwnode=f"{HWNodeType.WriteScanner}")
                buffet = pydot.Node(f"buffet_{self.get_next_seq()}",
                                    **attrs, label=f"{og_label}_buffet", hwnode=f"{HWNodeType.Buffet}")
                glb_write = pydot.Node(f"glb_write_{self.get_next_seq()}",
                                       **attrs, label=f"{og_label}_glb_write", hwnode=f"{HWNodeType.GLB}")
                if self.local_mems is False:
                    memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs,
                                        label=f"{og_label}_SRAM", hwnode=f"{HWNodeType.Memory}")
                crd_out_edge = [edge for edge in self.graph.get_edges() if "crd" in edge.get_label() and
                                edge.get_source() == node.get_name()][0]
                ref_out_edge = [edge for edge in self.graph.get_edges() if "ref" in edge.get_label() and
                                edge.get_source() == node.get_name()][0]
                ref_in_edge = None
                if not root:
                    # Then we have ref in edge...
                    ref_in_edge = [edge for edge in self.graph.get_edges() if "ref" in edge.get_label() and
                                   edge.get_destination() == node.get_name()][0]
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
                                     label=f"{og_label}_rd_scan", hwnode=f"{HWNodeType.ReadScanner}")
                wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}", **attrs,
                                     label=f"{og_label}_wr_scan", hwnode=f"{HWNodeType.WriteScanner}")
                buffet = pydot.Node(f"buffet_{self.get_next_seq()}", **attrs,
                                    label=f"{og_label}_buffet", hwnode=f"{HWNodeType.Buffet}")
                glb_read = pydot.Node(f"glb_read_{self.get_next_seq()}", **attrs,
                                      label=f"{og_label}_glb_read", hwnode=f"{HWNodeType.GLB}")
                if self.local_mems is False:
                    memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs,
                                        label=f"{og_label}_SRAM", hwnode=f"{HWNodeType.Memory}")
                vals = 'vals' in node.get_mode()
                in_edge = None
                if vals:
                    in_edge = [edge for edge in self.graph.get_edges() if "val" in edge.get_label() and
                               edge.get_destination() == node.get_name()][0]
                else:
                    in_edge = [edge for edge in self.graph.get_edges() if "crd" in edge.get_label() and
                               edge.get_destination() == node.get_name()][0]

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
                                 **attrs, label=f"{og_label}_rd_scan", hwnode=f"{HWNodeType.ReadScanner}")
            wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}",
                                 **attrs, label=f"{og_label}_wr_scan", hwnode=f"{HWNodeType.WriteScanner}")
            buffet = pydot.Node(f"buffet_{self.get_next_seq()}",
                                **attrs, label=f"{og_label}_buffet", hwnode=f"{HWNodeType.Buffet}")
            glb_write = pydot.Node(f"glb_write_{self.get_next_seq()}",
                                   **attrs, label=f"{og_label}_glb_write", hwnode=f"{HWNodeType.GLB}")
            if self.local_mems is False:
                memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs,
                                    label=f"{og_label}_SRAM", hwnode=f"{HWNodeType.Memory}")
            val_out_edge = [edge for edge in self.graph.get_edges() if "val" in edge.get_label() and
                            edge.get_source() == node.get_name()][0]
            # Then we have ref in edge...
            ref_in_edge = [edge for edge in self.graph.get_edges() if "ref" in edge.get_label() and
                           edge.get_destination() == node.get_name()][0]
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
