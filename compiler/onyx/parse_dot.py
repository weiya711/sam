import pydot


class SAMDotGraph():

    def __init__(self, filename=None) -> None:
        assert filename is not None, "filename is None"
        self.graphs = pydot.graph_from_dot_file(filename)
        self.graph = self.graphs[0]
        self.mapped_graph = {}
        self.seq = 0
        # Passes to lower to CGRA
        self.rewrite_lookup()
        self.rewrite_arrays()

    def get_next_seq(self):
        ret = self.seq
        self.seq += 1
        return ret

    def rewrite_lookup(self):
        '''
        Rewrites the lookup nodes to become (wr_scan, rd_scan, buffet) triples
        '''
        nodes_to_proc = [node for node in self.graph.get_nodes() if 'fiberlookup' in node.get_comment() or 'fiberwrite' in node.get_comment()]

        # for node in self.graph.get_nodes():
        for node in nodes_to_proc:
            if 'fiberlookup' in node.get_comment():
                # Rewrite this node to a read
                root = bool(node.get_root())
                print("HELLO")
                root = False
                if 'true' in node.get_root():
                    root = True
                print(node.get_root())
                print(root)
                attrs = node.get_attributes()
                og_label = attrs['label']
                del attrs['label']
                rd_scan = pydot.Node(f"rd_scan_{self.get_next_seq()}", **attrs, label=f"{og_label}_rd_scan")
                wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}", **attrs, label=f"{og_label}_wr_scan")
                buffet = pydot.Node(f"buffet_{self.get_next_seq()}", **attrs, label=f"{og_label}_buffet")
                glb_write = pydot.Node(f"glb_write_{self.get_next_seq()}", **attrs, label=f"{og_label}_glb_write")
                memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs, label=f"{og_label}_SRAM")
                # for edge in self.graph.get_edges():
                #     print(edge.get_destination())
                crd_out_edge = [edge for edge in self.graph.get_edges() if "crd" in edge.get_label() and edge.get_source() == node.get_name()][0]
                # print(crd_out_edge)
                # print(crd_out_edge)
                ref_out_edge = [edge for edge in self.graph.get_edges() if "ref" in edge.get_label() and edge.get_source() == node.get_name()][0]
                # print(ref_out_edge)
                ref_in_edge = None
                if not root:
                    # Then we have ref in edge...
                    # ref_in_edge = [edge for edge in self.graph.get_edges() if "ref" in edge.get_label() and edge.get_destination() == node.get_name()][0]
                    # print("HEREE")
                    ref_in_edge = [edge for edge in self.graph.get_edges() if "ref" in edge.get_label() and edge.get_destination() == node.get_name()][0]
                    # print(ref_in_edge)
                # Now add the nodes and move the edges...
                self.graph.add_node(rd_scan)
                self.graph.add_node(wr_scan)
                self.graph.add_node(buffet)
                self.graph.add_node(glb_write)
                self.graph.add_node(memory)
                # Glb to WR
                glb_to_wr = pydot.Edge(src=glb_write, dst=wr_scan, label=f"glb_to_wr_{self.get_next_seq()}", style="bold")
                self.graph.add_edge(glb_to_wr)
                # write + read to buffet
                wr_to_buff = pydot.Edge(src=wr_scan, dst=buffet, label=f'wr_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(wr_to_buff)
                rd_to_buff = pydot.Edge(src=rd_scan, dst=buffet, label=f'rd_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(rd_to_buff)
                # Mem to buffet
                mem_to_buff = pydot.Edge(src=buffet, dst=memory, label=f'mem_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(mem_to_buff)
                # Now inject the read scanner to other nodes...
                rd_to_down_crd = pydot.Edge(src=rd_scan, dst=crd_out_edge.get_destination(), **crd_out_edge.get_attributes())
                print(rd_to_down_crd)
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
                rd_scan = pydot.Node(f"rd_scan_{self.get_next_seq()}", **attrs, label=f"{og_label}_rd_scan")
                wr_scan = pydot.Node(f"wr_scan_{self.get_next_seq()}", **attrs, label=f"{og_label}_wr_scan")
                buffet = pydot.Node(f"buffet_{self.get_next_seq()}", **attrs, label=f"{og_label}_buffet")
                glb_read = pydot.Node(f"glb_read_{self.get_next_seq()}", **attrs, label=f"{og_label}_glb_read")
                memory = pydot.Node(f"memory_{self.get_next_seq()}", **attrs, label=f"{og_label}_SRAM")
                vals = 'vals' in node.get_mode()
                in_edge = None
                if vals:
                    in_edge = [edge for edge in self.graph.get_edges() if "val" in edge.get_label() and edge.get_destination() == node.get_name()][0]
                else:
                    in_edge = [edge for edge in self.graph.get_edges() if "crd" in edge.get_label() and edge.get_destination() == node.get_name()][0]

                # Now add the nodes and move the edges...
                self.graph.add_node(rd_scan)
                self.graph.add_node(wr_scan)
                self.graph.add_node(buffet)
                self.graph.add_node(glb_read)
                self.graph.add_node(memory)
                # RD to GLB
                rd_to_glb = pydot.Edge(src=rd_scan, dst=glb_read, label=f"glb_to_wr_{self.get_next_seq()}", style="bold")
                self.graph.add_edge(rd_to_glb)
                # write + read to buffet
                wr_to_buff = pydot.Edge(src=wr_scan, dst=buffet, label=f'wr_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(wr_to_buff)
                rd_to_buff = pydot.Edge(src=rd_scan, dst=buffet, label=f'rd_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(rd_to_buff)
                # Mem to buffet
                mem_to_buff = pydot.Edge(src=buffet, dst=memory, label=f'mem_to_buff_{self.get_next_seq()}')
                self.graph.add_edge(mem_to_buff)
                # Now inject the read scanner to other nodes...
                up_to_wr = pydot.Edge(src=in_edge.get_source(), dst=wr_scan, **in_edge.get_attributes())
                self.graph.add_edge(up_to_wr)

                # Delte old stuff...
                self.graph.del_node(node)
                self.graph.del_edge(in_edge.get_source(), in_edge.get_destination())

    def rewrite_arrays(self):
        '''
        Rewrites the array nodes to become (lookup, buffet) triples
        '''
        for node in self.graph.get_nodes():
            pass

    def get_graph(self):
        return self.graph


if __name__ == "__main__":
    matmul_dot = "/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/" + "matmul_ijk.gv"
    temp_out = "/home/max/Documents/SPARSE/sam/mek.gv"
    sdg = SAMDotGraph(filename=matmul_dot)
    graph = sdg.get_graph()
    # print(graph)
    graph.write_png('output.png')
    output_graphviz = graph.create_dot()
    graph.write_dot(temp_out)
