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
        self.mapped_graph = {}
        self.seq = 0
        self.local_mems = local_mems
        self.use_fork = use_fork
        # Passes to lower to CGRA
        self.rewrite_lookup()
        self.rewrite_arrays()
        # If using real fork, we don't rewrite the rsg broadcast in the same way
        if self.use_fork:
            self.rewrite_broadcast()
        else:
            self.rewrite_rsg_broadcast()
        self.map_nodes()

    def map_nodes(self):
        '''
        Iterate through the nodes and map them to the proper HWNodes
        '''

        for node in self.graph.get_nodes():
            # Simple write the HWNodeType attribute
            if 'hwnode' not in node.get_attributes():
                node.create_attribute_methods(node.get_attributes())
                n_type = node.get_type().strip('"')
                # print(n_type)
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
                elif n_type == "mul":
                    hw_nt = f"HWNodeType.Compute"
                elif n_type == "reduce":
                    hw_nt = f"HWNodeType.Reduce"
                elif n_type == "intersect":
                    hw_nt = f"HWNodeType.Intersect"
                elif n_type == "crddrop":
                    hw_nt = f"HWNodeType.Merge"
                else:
                    raise SAMDotGraphLoweringError(f"Node is of type {n_type}")

                node.get_attributes()['hwnode'] = hw_nt

    def get_next_seq(self):
        ret = self.seq
        self.seq += 1
        return ret

    def rewrite_broadcast(self):
        '''
        Rewrites the broadcast going into an RSG and makes it n or more separate connections
        '''
        nodes_to_proc = []
        for node in self.graph.get_nodes():
            # print(node)
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
            # print(node)
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
                # print(edge)
                source_node = edge.get_source()
                source_node = self.graph.get_node(source_node)[0]
                # print(source_node)
                if "broadcast" in source_node.get_attributes()['type'].strip('"') and \
                        edge.get_destination() == rsg_node.get_name():
                    broadcast_nodes.append(source_node)

            # Leave early.
            if len(broadcast_nodes) == 0:
                # print(f"NO BROADCAST NODES?")
                return
            bc_node = broadcast_nodes[0]
            # Now that we have the broadcast node, get the incoming edge and other outgoing edge
            incoming_edge = [edge for edge in self.graph.get_edges() if edge.get_destination() == bc_node.get_name()][0]
            outgoing_edge = [edge for edge in self.graph.get_edges() if edge.get_source() == bc_node.get_name() and
                             edge.get_destination() == rsg_node.get_name()][0]
            other_outgoing_edge = [edge for edge in self.graph.get_edges() if edge.get_source() == bc_node.get_name() and
                                   edge.get_destination() != rsg_node.get_name()][0]
            # Now, connect the original source to the rsg and the rsg to the original other branch
            og_to_rsg = pydot.Edge(src=incoming_edge.get_source(), dst=rsg_node, **incoming_edge.get_attributes())
            rsg_to_branch = pydot.Edge(src=rsg_node, dst=other_outgoing_edge.get_destination(),
                                       **other_outgoing_edge.get_attributes())

            # Now delete the broadcast and all original edge
            ret = self.graph.del_edge(incoming_edge.get_source(), incoming_edge.get_destination())
            # print(f"DELETED EDGE0? : {ret}")
            ret = self.graph.del_edge(outgoing_edge.get_source(), outgoing_edge.get_destination())
            # print(f"DELETED EDGE1? : {ret}")
            ret = self.graph.del_edge(other_outgoing_edge.get_source(), other_outgoing_edge.get_destination())
            # print(f"DELETED EDGE2? : {ret}")
            ret = self.graph.del_node(bc_node)

            # print(f"DELETED NODE? : {ret}")

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
                rd_to_down_crd = pydot.Edge(src=rd_scan, dst=crd_out_edge.get_destination(), **crd_out_edge.get_attributes())
                # print(rd_to_down_crd)
                rd_to_down_ref = pydot.Edge(src=rd_scan, dst=ref_out_edge.get_destination(), **ref_out_edge.get_attributes())
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
                rd_to_glb = pydot.Edge(src=rd_scan, dst=glb_read, label=f"glb_to_wr_{self.get_next_seq()}", style="bold")
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
            # print(rd_to_down_crd)
            rd_to_down_val = pydot.Edge(src=rd_scan, dst=val_out_edge.get_destination(), **val_out_edge.get_attributes())
            self.graph.add_edge(rd_to_down_val)
            up_to_ref = pydot.Edge(src=ref_in_edge.get_source(), dst=rd_scan, **ref_in_edge.get_attributes())
            self.graph.add_edge(up_to_ref)

            # Delte old stuff...
            ret = self.graph.del_node(node)
            ret = self.graph.del_edge(val_out_edge.get_source(), val_out_edge.get_destination())
            self.graph.del_edge(ref_in_edge.get_source(), ref_in_edge.get_destination())

    def get_graph(self):
        return self.graph


if __name__ == "__main__":
    matmul_dot = "/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/" + "matmul_ijk.gv"
    # matmul_dot = "/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/" + "mat_identity.gv"
    # matmul_dot = "/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/" + "mat_elemmul.gv"
    temp_out = "/home/max/Documents/SPARSE/sam/mek.gv"
    sdg = SAMDotGraph(filename=matmul_dot, use_fork=True)
    graph = sdg.get_graph()
    print(graph)
    graph.write_png('output.png')
    output_graphviz = graph.create_dot()
    graph.write_dot(temp_out)
