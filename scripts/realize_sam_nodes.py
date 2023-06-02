

def realize_sam_node(node_info):
    if node_info["type"] == "fiberlookup":
        if node_info["format"] == "dense":
            block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                             node_info["index"] + "_" + str(u) + "_" + self.mem_lvl + " = UncompressCrdRdScan(dim=" + node_info["tensor"] +
                             "_shape[" + node_info["mode"] + "]" + ", debug=debug_sim, statistics=report_stats)\n")]
            object_name = [node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl]
        if node_info["format"] == "compressed":
            block_realize = (tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] +
                            "_" + str(u) + "_" + self.mem_lvl + " = CompressedCrdRdScan(crd_arr=" + node_info["tensor"] +
                            "_crd" + node_info["mode"] + ", seg_arr=" + node_info["tensor"] +
                            "_seg" + node_info["mode"] + ", debug=debug_sim, statistics=report_stats)\n")
            object_name = [node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "arrayvals":
        block_realize = [tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] + "_" + str(u) + "_" + self.mem_lvl + " = Array(init_arr=" +
                         node_info["tensor"] + "_vals, " + "debug=debug_sim, statistics=report_stats)\n"]
        object_name = [node_info["type"] + "_" + node_info["tensor"] + "_" + str(u) + "_" + self.mem_lvl]
    elif "broadcast" in self.networkx_graph.nodes[u]['comment']:
        return [], []
    elif node_info["type"] == "repsiggen":
        block_realize = [tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                        " = RepeatSigGen(debug=debug_sim, statistics=report_stats)\n")]
        object_name = [node_info["type"] + "_" + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "repeat":
        block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                        " = Repeat(debug=debug_sim, statistics=report_stats)\n")]
        object_name = [node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "intersect":
        block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl + " = Intersect2(debug=debug_sim, " +
                         "statistics=report_stats)\n")]
        object_name = [node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "union":
        block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl + " = Union2(debug=debug_sim, " +
                         "statistics=report_stats)\n")]
        object_name = [node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "spaccumulator" and node_info["order"] == "1":
        block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                         u) + "_" + self.mem_lvl +\
                         " = SparseAccumulator" + node_info["order"] + 1 + "(debug=debug_sim, statistics=report_stats)\n")]
        object_name = node_info["type"] + node_info["order"] + "_" + str(u) + "_" + self.mem_lvl
        block_realize.append((tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_crd_inner" + "_" + self.mem_lvl +\
                              " = StknDrop(debug=debug_sim, statistics=report_stats)\n"))
        block_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_crd_outer" + "_" + self.mem_lvl +\
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
        block_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_val" + "_" + self.mem_lvl + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
        block_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_val" + "_" + self.mem_lvl + " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
    elif node_info["type"] == "crddrop":
        block_realize = [(tab(self.scope_lvl + 1) +
                         node_info["type"] + "_" + str(u) + "_" + self.mem_lvl + " = CrdDrop(debug=debug_sim, statistics=report_stats)\n")
        object_name = [(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl)]
    elif node_info["type"] == "crdhold":
        block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) +
                         "_" + self.mem_lvl + " = CrdHold(debug=debug_sim, statistics=report_stats)\n")
        object_name = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
    elif node_info["type"] == "mul":
        block_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) +
                         "_" + self.mem_lvl + " = Multiply2(debug=debug_sim, statistics=report_stats)\n")
        object_name = [node_info["type"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "add":
        blcok_realize = [(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl + " = Add2(debug=debug_sim, statistics=report_stats)\n")]
        self.d[u]["object"] = [node_info["type"] + "_" + str(u) + "_" + self.mem_lvl]
    elif node_info["type"] == "reduce":
        self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl + " = Reduce(debug=debug_sim, statistics=report_stats)\n")
        self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
    elif node_info["type"] == "fiberwrite":
        if node_info["mode"] == "vals":
            block_realize = self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl +
                                                " = ValsWrScan(size=" + array_size_computation(node_info["size"]) +
                                                ", fill=fill, debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl
                elif node_info["format"] == "compressed":
                    self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl +
                                                " = CompressWrScan(seg_size=" + array_size_computation(node_info["segsize"]) + ", size=" +
                                                array_size_computation(node_info["crdsize"]) + ", fill=fill," + " debug=debug_sim, " +
                                                "statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl
                else:
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl
                    continue
                if node_info["sink"] == "true":
                    output_nodes[self.d[u]["object"]] = node_info["mode"]

