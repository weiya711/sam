import pydot
import os
import argparse
import networkx as nx
import sys
from collections import defaultdict
# import realize_sam_node


frostt_list = ["tensor3_elemmul", "tensor3_identity", "tensor3_ttm", "tensor3_elemadd",
               "tensor3_innerprod", "tensor3_mttkrp", "tensor3_ttv", "tensor3_identity_dense"]
suitesparse_list = ["mat_elemmul", "mat_identity", "matmul_ijk", "matmul_ikj", "matmul_jki",
                    "matmul_jik", "matmul_kij", "matmul_jki", "mat_vecmul_ij", "mat_vecmul_ji",
                    "matmul_kji", "mat_elemadd3", "mat_sddmm.gv", "mat_elemadd", "mat_mattransmul",
                    "mat_residual", "mat_sddmm", "mat_identity_dense", "mat_spacc_simple"]
vec_list = ["vec_elemadd", "vec_elemmul", "vec_scalar_mul", "vec_identity",
            "vec_scalar_mul", "vecmul", "vecmul_ij", "vecmul_ki", "vec_spacc_simple",
            "vec_sd_compression", "vec_ds_compression"]
other_list = ["mat_mattransmul", "mat_residual", "tensor3_ttm", "tensor3_mttkrp", "tensor3_ttv", "mat_vecmul_ij",
              "mat_vecmul_ji"]
MEM_LEVELS = 2


class TensorFormat:
    def __init__(self):
        self.tensors = {}

    def add_tensor_and_format(self, ten, form):
        self.tensors[ten] = form

    def set_format(self, ten, form):
        self.tensors[ten] = form

    def get_format(self, ten):
        return self.tensors[ten]

    def set_all_tensors(self, path):
        graphs1 = pydot.graph_from_dot_file(path)
        graph1 = graphs1[0]
        networkx_graph = nx.nx_pydot.from_pydot(graph1)
        tensor_list = graph1.get_comment().strip('"').split(",")
        for tensor_info in tensor_list:
            node = tensor_info.split("=")
            self.add_tensor_and_format(node[0], node[1])
        return

    def return_size(self):
        return len(self.tensors)

    def return_all_tensors(self):
        return self.tensors.keys()

    def get_location(self, ten):
        if ten == "X" or ten == "x":
            return 0
        else:
            return 1


class CodeGenerationdatasets:
    def __init__(self, graph=None):
        # Rememebers [parents of a node
        self.stream_join_elements = {}
        # ALl edges into a node
        self.edge_data = {}
        # Not used required since intersection has special structure need to know which reference is which coordinate
        self.intersect_dataset = defaultdict(dict)
        self.done_all = {}
        self.graph = graph

    def build_datasets(self, networkx_graph):
        for u, v, a in networkx_graph.edges(data=True):
            if v not in self.done_all:
                self.done_all[v] = 0
            if v not in self.stream_join_elements:
                self.stream_join_elements[v] = [u]
                self.edge_data[v] = [str((a["label"]).strip('"'))]
            else:
                self.stream_join_elements[v].append(u)
                self.edge_data[v].append(str((a["label"]).strip('"')))

    def get_edge_data(self):
        return self.edge_data

    def get_edge_info(self, v, i):
        if self.edge_data[v][i] is None and len(self.edge_data[v][i]) == 0:
            return ""
        if "-i" in self.edge_data[v][i] or "-j" in self.edge_data[v][i] or "-k" in self.edge_data[v][i]:
            return str(self.edge_data[v][i])[:-2]
        return str(self.edge_data[v][i])

    def get_parents(self):
        return self.stream_join_elements

    def get_if_done(self):
        return self.done_all

    def add_done(self, a):
        self.done_all[a] = 1

    def set_edge_data(self, v, i, string):
        self.edge_data[v][i] = string

    def get_if_node_done(self, v):
        return self.done_all[v]

    def if_all_graph_realized(self):
        if self.graph is not None:
            print("checking if done : ", self.done_all)
            for a in self.done_all.keys():
                if self.done_all[a] == 0:
                    print("                 ", a, " ", self.graph.nodes()[a])
        for nodes in self.done_all.keys():
            if self.done_all[nodes] == 0:
                return False
        return True


def generate_tiling_header(f, app_name, loop_order_and_sizes=None):
    f.write(tab(1) + "with open(os.path.join(sam_home, \"tiles/" + app_name + "/tensor_sizes\"), \"rb\") as ff:\n")
    f.write(tab(2) + "sizes_dict_level_full = pickle.load(ff)\n")
    for i in range(MEM_LEVELS):
        f.write(tab(1) + "with open(os.path.join(sam_home, \"tiles/" +
                app_name + "/hw_level_" + str(i) + "_sizes\"), \"rb\") as ff:\n")
        f.write(tab(2) + "sizes_dict_level" + str(i) + " = pickle.load(ff)\n")
    f.write(tab(1) + "with open(os.path.join(sam_home, \"./sam/sim/src/tiling/\" + yaml_name), \"r\") as stream:\n")
    f.write(tab(2) + "memory_config = yaml.safe_load(stream)\n")
    # Get the arrays for seg and crd arrays for the higher levels
    f.write(tab(1) + "struct = {")
    if MEM_LEVELS == 2:
        if loop_order_and_sizes is None:
            index = "B"
            val = 100
            f.write(index + "00 : 1 + int(" + str(val) + ") // (")
            f.write("(loop_config[\"Glb_tile_size\"] * loop_config[\"Mem_tile_size\"])), ")
            f.write(index + "0 :  int(loop_config[\"Glb_tile_size\"])")
        else:
            for index in loop_order_and_sizes.keys():
                f.write(index + "00 : 1 + int(" + loop_order_and_sizes[key] + ") // (")
                f.write("(loop_config[\"Glb_tile_size\"] * loop_config[\"Mem_tile_size\"]))\n")

            for index in loop_order_and_sizes.keys():
                f.write(index + "0 :  int(loop_config[\"Glb_tile_size\"])")
    f.write("}")
    f.write("\n")


def generate_tiling_graph(graph):
    g = graph.copy()
    # Need 2 passes over the graph
    rem_list = []
    last_a = None
    print("generating tilig graph")
    print(g)
    memory_nodes = defaultdict(dict)
    # node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) + "_"
    while (last_a is None) or (last_a not in rem_list):
        for a in list(nx.topological_sort(g)):
            node_i = breakup_node_info(g.nodes[a])
            print("type of the block ", node_i["type"])
            if node_i["type"] == "arrayvals" or node_i["type"] == "fiberlookup":
                print(g.nodes[a]["comment"])
                print(node_i)
                if not node_i["type"] == "arrayvals":
                    memory_nodes[node_i["type"] + "_" +
                                 node_i["tensor"] + node_i["index"] +
                                 "_" + str(u) + "_"] = {"node_type": node_i["type"],
                                                        "tensor": node_i["tensor"],
                                                        "index": node_i["index"],
                                                        "mode": node_i["mode"]}
                else:
                    memory_nodes[node_i["type"] + "_" + node_i["tensor"] + "_" + str(u) + "_"] =\
                        {"node_type": node_i["type"], "tensor": node_i["tensor"]}
            if node_i["type"] == "arrayvals":
                if a not in rem_list:
                    rem_list.append(a)
            elif node_i["type"] != "arrayvals":
                for preds in g.predecessors(a):
                    if preds in rem_list:
                        if a not in rem_list:
                            rem_list.append(a)
        last_a = a
    for a in rem_list:
        node_i = breakup_node_info(g.nodes[a])
        if node_i["type"] == "arrayvals":
            g.nodes[a].update({'comment': '"type=memory_block,tensor=' + node_i["tensor"] + '"',
                               'label': '"memory_block"',
                               'type': '"memory_block"'})
        else:
            g.remove_node(a)
    any_nodes_exist = True
    while any_nodes_exist:
        any_nodes_exist = False
        for a in list(nx.topological_sort(g)):
            node_i = breakup_node_info(g.nodes[a])
            if sum(1 for _ in g.successors(a)) == 0 and node_i["type"] != "memory_block":
                any_nodes_exist = True
                print("node is :: ", a, g.nodes[a])
                g.remove_node(a)
        print(any_nodes_exist)
    return g, memory_nodes


def generate_tiling_end(f, mem_blks):
    f.write(tab(2) + "done = ")
    temp = False
    for blk in mem_blks:
        if not temp:
            temp = True
            f.write("tiled_done and " + blk + ".done()")
        else:
            f.write(" and " + blk + ".done()")
    f.write("\n")
    f.write(tab(2) + "time_cnt += 1\n\n")


def tab(a):
    ans = ""
    for i in range(a):
        ans += "    "
    return ans


def sort_output_nodes(output_nodes, flag):
    node_with_vals = {}
    node_with_modes = {}
    for nodes in output_nodes.keys():
        if "vals" in output_nodes[nodes]:
            node_with_vals[nodes] = output_nodes[nodes]
        else:
            node_with_modes[nodes] = int(output_nodes[nodes])
    output = node_with_modes
    output = {}
    output = dict(sorted(node_with_modes.items(), key=lambda item: item[1]))
    flag = flag.replace("s", "")
    flag = flag.replace("d", "")
    flag = flag.replace("none", "")
    output2 = {}
    for i in flag:
        i = int(i)
        for key in output.keys():
            if output[key] == i:
                output2[key] = str(output[key])
                break

    for nodes in node_with_vals.keys():
        output2[nodes] = node_with_vals[nodes]
    return output2


def generate_header(f, out_name):
    f.write("import pytest\n")
    f.write("import time\n")
    f.write("import scipy.sparse\n")
    f.write("from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan\n")
    f.write("from sam.sim.src.wr_scanner import ValsWrScan\n")
    f.write("from sam.sim.src.joiner import Intersect2, Union2\n")
    f.write("from sam.sim.src.compute import Multiply2, Add2\n")
    f.write("from sam.sim.src.crd_manager import CrdDrop, CrdHold\n")
    f.write("from sam.sim.src.repeater import Repeat, RepeatSigGen\n")
    f.write("from sam.sim.src.accumulator import Reduce\n")
    f.write("from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2\n")
    f.write("from sam.sim.src.token import *\n")
    f.write("from sam.sim.test.test import *\n")
    f.write("from sam.sim.test.gold import *\n")
    f.write("import os\n")
    f.write("import csv\n")
    f.write("cwd = os.getcwd()\n")
    if out_name in suitesparse_list:
        f.write("formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n")
    elif out_name in frostt_list:
        f.write("formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n")
    if out_name in other_list:
        f.write("other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n")
    f.write("\n\n# FIXME: Figureout formats\n")
    f.write("@pytest.mark.skipif(\n")
    f.write(tab(1) + "os.getenv('CI', 'false') == 'true',\n")
    f.write(tab(1) + "reason='CI lacks datasets',\n")
    f.write(")\n")
    if out_name in suitesparse_list:
        f.write("@pytest.mark.suitesparse\n")
    if out_name in frostt_list:
        f.write("@pytest.mark.frostt\n")
    if out_name in vec_list:
        f.write("@pytest.mark.vec\n")
    # f.write("def test_" + out_name + "(samBench, " + get_dataset_name(out_name) + ", cast, check_gold, debug_sim, "
    f.write("def test_" + out_name + "(samBench, " + get_dataset_name(out_name) + ", check_gold, debug_sim, "
            "report_stats, fill=0):\n")

def get_dataset_name(test_name):
    if test_name in frostt_list:
        return "frosttname"
    elif test_name in suitesparse_list:
        return "ssname"
    elif test_name in vec_list:
        return "vecname"
    else:
        return ""

def get_common_test_name(test_name):
    if "matmul" in test_name:
        return test_name[:-4]
    else:
        return test_name


def generate_tiling_output_crds(f, scope_lvl, parents=[], tensors=[]):
    if MEM_LEVELS == 2:
        for i in range(len(parents)):
            tensor = tensors[i]
            par = parents[i]
            f.write(tab(scope_lvl) + tensor + "_glb = " + par + ".token() // 1000000\n")
        for i in range(len(parents)):
            tensor = tensors[i]
            par = parents[i]
            f.write(tab(scope_lvl) + tensor + "_mem = " + par + ".token() % 1000000\n")
        for i in range(len(tensors)):
            f.write(tab(scope_lvl) + tensors[i] + "_dir = \"tensor_" + tensors[i] +
                    "_tile_\" + str(" + tensors[i] + "_glb) + str(" + tensors[i] + "_mem)\n")
    else:
        for lvl in range(MEM_LEVELS, 0, -1):
            mem_lvl = "0" * lvl
            for i in range(len(parents)):
                tensor = tensors[i]
                par = parents[i]
                if lvl > 1:
                    f.write(tab(scope_lvl) + tensor + "_" + mem_lvl +
                            " = (" + par + ".token() // (10 ** " +
                            str(6 * (lvl - 1)) + ") % " + str(6 * (lvl)) + ") \n")
                elif lvl == 1:
                    f.write(tab(scope_lvl) + tensor + "_" + mem_lvl + " = " + par + ".token() % (10 ** " +
                            str(6 * (lvl)) + ") \n")
        for i in range(len(tensors)):
            f.write(tab(scope_lvl) + tensors[i] + "_dir = \"tensor_" + tensors[i] + "_tile_\"")
        for lvl in range(MEM_LEVELS, 0, -1):
            mem_lvl = "0" * lvl
            f.write(" + str(" + tensors[i] + "_" + mem_lvl + ")")
        f.write("\n")
    return


def generate_datasets_code(f, tensor_formats, scope_lvl, tensor_info, tensor_format_parse,
                           test_name, tiling=False, parents=[], tensors=[], selected_tensor=None):
    # Assuming the format is csr and csc:
    f.write("\n")
    scope_lvl += 1
    if tiling:
        assert len(parents) > 0
        print(parents, tensors)
        assert len(parents) == len(tensors)
        if MEM_LEVELS == 2:
            for i in range(len(parents)):
                tensor = tensors[i]
                if selected_tensor is not None and tensor != selected_tensor:
                    continue
                par = parents[i]
                f.write(tab(scope_lvl) + tensor + "_glb = " + par + ".token() // 1000000\n")
            for i in range(len(parents)):
                tensor = tensors[i]
                if selected_tensor is not None and tensor != selected_tensor:
                    continue
                par = parents[i]
                f.write(tab(scope_lvl) + tensor + "_mem = " + par + ".token() % 1000000\n")
            for i in range(len(tensors)):
                tensor = tensors[i]
                if selected_tensor is not None and tensor != selected_tensor:
                    continue
                f.write(tab(scope_lvl) + tensors[i] + "_dir = \"tensor_" + tensors[i] + "_tile_\" + str(" +
                        tensors[i] + "_glb) + str(" + tensors[i] + "_mem)\n")

    for ten in tensor_format_parse.return_all_tensors():
        if selected_tensor is not None and ten != selected_tensor:
            continue

        if tensor_format_parse.get_location(ten) == 0:
            continue
        if not tiling:
            f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, " + get_dataset_name(test_name) +
                    ", \"" + test_name + "\")\n")
        else:
            f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, " + ten + "_dir)\n")
        f.write(
            tab(scope_lvl) + ten + "_shape_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
            "_mode_shape\")\n")
        f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
        if tensor_format_parse.get_format(ten) == "ds01":
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname" + ",  \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_dirname" + "tensor_" + ten + "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
        elif tensor_format_parse.get_format(ten) == "ds10":
            f.write(
                tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            # f.write(tab(scope_lvl) + ten + "_seg1" +  " = read_inputs(" + ten + "1_crd_filename)")
        elif tensor_format_parse.get_format(ten) == "ss01":
            f.write(
                tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname,  \"tensor_" + ten +
                "_mode_0_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
        elif tensor_format_parse.get_format(ten) == "ss10":
            f.write(
                tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname,  \"tensor_" + ten +
                "_mode_0_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_0_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")

        elif tensor_format_parse.get_format(ten) == "dss012":
            f.write(
                tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_1_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
            f.write(
                tab(scope_lvl) + ten + "2_seg_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_2_seg\" )\n")
            f.write(tab(scope_lvl) + ten + "_seg2" + " = read_inputs(" + ten + "2_seg_filename)\n")
            f.write(
                tab(scope_lvl) + ten + "2_crd_filename = os.path.join(" + ten + "_dirname, \"tensor_" + ten +
                "_mode_2_crd\" )\n")
            f.write(tab(scope_lvl) + ten + "_crd2" + " = read_inputs(" + ten + "2_crd_filename)\n\n")
        else:
            ct = 0
            for i in range(len(tensor_format_parse.get_format(ten))):
                if tensor_format_parse.get_format(ten)[i] == "s":
                    f.write(tab(scope_lvl) + ten + str(i) + "_seg_filename = os.path.join(" +
                            ten + "_dirname, \"tensor_" + ten +
                            "_mode_" + str(i) + "_seg\")\n")
                    f.write(
                        tab(scope_lvl) + ten + "_seg" + str(i) + " = read_inputs(" + ten + str(i) + "_seg_filename)\n")
                    f.write(tab(scope_lvl) + ten + str(i) + "_crd_filename = os.path.join(" +
                            ten + "_dirname, \"tensor_" + ten + "_mode_" + str(i) + "_crd\")\n")
                    f.write(tab(scope_lvl) + ten + "_crd" + str(i) + " = read_inputs(" + ten + str(
                        i) + "_crd_filename)\n\n")
        f.write(tab(scope_lvl) + ten + "_vals_filename = os.path.join(" + ten +
                "_dirname, \"tensor_" + ten + "_mode_vals\")\n")
        f.write(tab(scope_lvl) + ten + "_vals" + " = read_inputs(" + ten + "_vals_filename, float)\n\n")


def gen_data_formats(size, app_name, path):
    ans_list = []
    if size - 1 == 1:
        ans_list = ["orig"]
        return ans_list
    if size - 1 == 2:
        if "matmul" in app_name:
            ans_list = ["orig", "shift-trans"]
            return ans_list
        if "elemmul" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list
        if "elemadd" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list
        if "innerprod" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list

        ans_list = ["orig", "other"]

        return ans_list
    else:
        ans_list = ["orig"]
        for i in range(size - 2):
            ans_list.append("other")
        return ans_list


def output_check_nodes(f, root_nodes):
    for r in root_nodes:
        f.write(tab(1) + "in_ref_" + str(r) + " = [0, 'D']\n")
    f.write(tab(1) + "done = False\n")
    f.write(tab(1) + "time_cnt = 0\n")


def finish_outputs(f, elements, nodes_completed):
    for i in nodes_completed:
        f.write(i)
    f.write("\n")
    output_list = ""
    # Write done statement
    f.write(tab(2) + "done = ")
    elements2 = []
    for elem in elements.keys():
        elements2.append(elem)
    for elem in elements.keys():
        f.write(elem + ".out_done()")
        if elem != elements2[-1]:
            f.write(" and ")
        else:
            f.write("\n")
    # TIme counter update
    f.write(tab(2) + "time_cnt += 1\n\n")
    # Autosize all blocks
    for elem in elements.keys():
        f.write(tab(1) + elem + ".autosize()\n")
    f.write("\n")

    if len(elements.keys()) > 1:
        f.write(tab(1) + "out_crds = [")
    else:
        f.write(tab(1) + "out_crds = []\n")

    output_list += " out_crds = ["
    count = 0
    for elem in elements.keys():
        if elements[elem] != "vals":
            f.write(elem + ".get_arr()")
            output_list += elem + ".get_arr()"
            count += 1
            if count < len(elements2) - 1:
                f.write(", ")
                output_list += ", "
            else:
                f.write("]\n")
                output_list += "]"
    count = 0
    if len(elements.keys()) > 1:
        f.write(tab(1) + "out_segs = [")
    else:
        f.write(tab(1) + "out_segs = []\n")

    output_list += ", out_segs = ["
    for elem in elements.keys():
        if elements[elem] != "vals":
            f.write(elem + ".get_seg_arr()")
            output_list += elem + ".get_seg_arr()"
            count += 1
            if count < len(elements2) - 1:
                f.write(", ")
                output_list += ", "
            else:
                f.write("]\n")
                output_list += "]"
    f.write(tab(1) + "out_vals = ")
    output_string = ", out_vals = "
    for elem in elements.keys():
        if elements[elem] == "vals":
            f.write(elem + ".get_arr()\n")
            output_string += elem + ".get_arr()"
    output_list += output_string
    return output_list


def generate_benchmarking_code(f, tensor_format_parse, test_name):
    f.write("\n" + tab(1) + "def bench():\n")
    f.write(tab(2) + "time.sleep(0.01)\n\n")
    f.write(tab(1) + "extra_info = dict()\n")
    f.write(tab(1) + "extra_info[\"dataset\"] = " + get_dataset_name(test_name) + "\n")
    f.write(tab(1) + "extra_info[\"cycles\"] = time_cnt\n")
    ct = 0
    output_tensor = ""
    for k in tensor_format_parse.return_all_tensors():
        if ct == 0:
            output_tensor = k
        if ct != 0:
            f.write(tab(1) + "extra_info[\"tensor_" + k + "_shape\"] = " + k + "_shape\n")
        ct += 1
    statistic_available = ["fiberlookup", "reduce", "spaccumulator", "crddrop",
                           "repeat", "repeatsiggen", "intersect", "fiberwrite",
                           "arrayvals"]
    for u in networkx_graph.nodes():
        if d[u]["type"] in statistic_available:
            f.write(tab(1) + "sample_dict = " + d[u]["object"] + ".return_statistics()\n")
            f.write(tab(1) + "for k in sample_dict.keys():\n")
            f.write(tab(2) + "extra_info[\"" + d[u]["object"] + "\" + \"_\" + k] = sample_dict[k]\n\n")


def generate_check_against_gold_code(f, tensor_format_parse, test_name):
    f.write(tab(1) + "if check_gold:\n")
    f.write(tab(2) + "print(\"Checking gold...\")\n")
    f.write(tab(2) + "check_gold_")
    check = out_name[num]
    check = get_common_test_name(check)
    f.write(check + "(" + get_dataset_name(test_name) + ", debug_sim, cast, out_crds, out_segs, out_vals, \"" +
            tensor_format_parse.get_format(output_tensor) + "\")\n")
    f.write(tab(1) + "samBench(bench, extra_info)\n")


def size_computation_write(a):
    ans = " 1 "
    a = int(a)
    if a == 1:
        ans = " B_dim0 "
    return ans


def size_comp_write(a):
    ans = ""
    a = int(a)
    for i in range(a - 1):
        ans = " B_dim0 *"
    ans += "B_dim0"
    return ans


def breakup_node_info(node_obj_name):
    node_name = node_obj_name["comment"]
    d2 = {}
    for k in node_obj_name.keys():
        if k != "comment":
            d2[k] = str(node_obj_name[k]).replace('"', '')
    d = dict(x.split("=") for x in node_name[1: -1].split(","))
    # print("d ", d)
    # print("d2 ", d2)
    # print("_______")
    return d


def remove_broadcast_nodes(G):
    g = G.copy()
    for a in g:
        g0 = g.copy()
        node_i = breakup_node_info(g.nodes[a])
        if node_i["type"] == "broadcast":
            for preds in g0.predecessors(a):
                for succs in g0.neighbors(a):
                    g0.add_edge(preds, succs, **(g0.get_edge_data(a, succs)[0]))
            g0.remove_node(a)
        g = g0
    return g


def parents_done(G, done, u):
    g = G.copy()
    ans = True
    for pred in G.predecessors(u):
        if pred not in done:
            ans = ans and True
        elif done[pred] == 1:
            ans = ans and True
        else:
            ans = False
    return ans


def array_size_computation(size_array):
    if "+" in size_array:
        size_arr = size_array.split("+")
    else:
        size_arr = [size_array]
    final_output_with_adds = ""
    output_array_with_adds = []
    for j in range(len(size_arr)):
        size_array = size_arr[j]
        if "*" in size_array:
            size_array = size_array.split("*")
        else:
            size_array = [size_array]
        output = []
        for strs in size_array:
            if len(strs) > 1:
                ans = strs[0]
                dim = strs[1]
                output.append(ans + "_shape[" + dim + "]")
            else:
                output.append(strs)
        output_str = ""
        for i in range(len(output) - 1):
            output_str += output[i]
            output_str += " * "
        output_str += output[-1]
        output_array_with_adds.append(output_str)
    for i in range(len(output_array_with_adds) - 1):
        final_output_with_adds += output_array_with_adds[i]
        final_output_with_adds += " + "
    final_output_with_adds += output_array_with_adds[-1]
    return final_output_with_adds


def get_all_files(directory_path):
    file_paths = []
    out_name = []

    for filename in os.listdir(directory_path):
        f = os.path.join(directory_path, filename)
        if filename[0] == ".":
            continue
        out_name.append(filename[0:-3])
        # checking if it is a file
        print(out_name[-1])
        if os.path.isfile(f):
            file_paths.append(f)
    return file_paths, out_name


class ParentChildGraph:
    def __init__(self):
        self.parent = defaultdict(dict)
        self.operation = defaultdict(dict)

    def add_parent_child(self, parent, child, child2):
        if parent in self.parent.keys():
            self.parent[parent].append(child)
            self.operation[parent].append(child2)
        else:
            self.parent[parent] = [child]
            self.operation[parent] = [child2]

    def get_parent(self, child):
        for parent in self.parent:
            # f.write("# "+ parent + " ##  " + self.parent[parent] + "  @@  " + child)
            if self.parent[parent][0] == child:
                return parent
        return parent

    def get_child(self, parent):
        # f.write("# ")
        # for a in self.parent[parent]:
        #     f.write(", "+ a)
        # f.write("\n")
        return self.parent[parent][0] + self.operation[parent][0]


class GraphRealization:
    def __init__(self, graph, mem_lvl, scope_lvl, f, parent=None, mem_blks_connect=None,
                 mem_blks=[], pipelined_tiles=False, pipelined_memory_nodes=defaultdict(dict)):
        self.mem_lvl = mem_lvl
        self.d = {}
        self.f = f
        self.networkx_graph = graph
        self.scope_lvl = scope_lvl
        self.intersect_dataset = defaultdict(dict)
        self.root_nodes = []
        self.memory_blks = mem_blks
        self.mem_blks_connect = ParentChildGraph()  # mem_blks_connect

        self.blks_to_realize = []
        self.parent_block = parent
        self.tensor_list = []
        # self.memory_blocks
        self.nxt_parent = None
        self.pipelined_tiles = pipelined_tiles
        self.pipelined_memory_nodes = pipelined_memory_nodes

    def get_tensor_list(self):
        return self.tensor_list

    def loop_start(self):
        self.blks_to_realize.append(tab(1) + "while not done and time_cnt < TIMEOUT:\n")

    def output_check_nodes(self):
        for r in self.root_nodes:
            self.blks_to_realize.append(tab(self.scope_lvl + 1) + "in_ref_" + str(r) + self.mem_lvl + " = [0, 'D']\n")

    def node_instantiations_mem(self, output_nodes, tens_fmt={}, tensor_information={},
                                tensor_format_parse=None, out_name=None, pipelined_tiles=False,
                                parent=[], whether_pipelined=False):
        invalid_flag = 0
        temp_string = ""
        temp_string += tab(self.scope_lvl)
        temp_flag = False
        nxt_parents = []
        for par in parent:
            if not temp_flag:
                temp_flag = True
                temp_string += tab(self.scope_lvl) + "if " + par + ".valid_tile()"
            else:
                temp_string += " and " + par + ".valid_tile()"
        if len(parent) != 0:
            self.scope_lvl += 1
            temp_string += ":\n"
            self.blks_to_realize.append(temp_string)
        self.blks_to_realize.append(tab(self.scope_lvl + 1) + "check_flag" + self.mem_lvl + " = True\n")
        for u in list(nx.topological_sort(self.networkx_graph)):
            node_info = breakup_node_info(self.networkx_graph.nodes[u])
            self.d[u] = node_info
            u_val = u
            if (node_info["type"] == "fiberlookup" or node_info["type"] == "repeat") and node_info["root"] == "true":
                self.root_nodes.append(node_info["tensor"])
            # realize_sam_node(node_info)
            if node_info["type"] == "fiberlookup":
                if node_info["format"] == "dense":
                    self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                                node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                                                " = UncompressCrdRdScan(dim=" + node_info["tensor"] +
                                                "_shape[" + node_info["mode"] +
                                                "]" + ", debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" +\
                        str(u) + "_" + self.mem_lvl
                if node_info["format"] == "compressed":
                    self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] +
                                                "_" + node_info["tensor"] + node_info["index"] +
                                                "_" + str(u) + "_" + self.mem_lvl +
                                                " = CompressedCrdRdScan(crd_arr=" + node_info["tensor"] +
                                                "_crd" + node_info["mode"] + ", seg_arr=" + node_info["tensor"] +
                                                "_seg" + node_info["mode"] +
                                                ", debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] +\
                        node_info["index"] + "_" + str(u) + "_" + self.mem_lvl

            elif node_info["type"] == "arrayvals":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] +
                                            "_" + node_info["tensor"] + "_" + str(u) +
                                            "_" + self.mem_lvl + " = Array(init_arr=" +
                                            node_info["tensor"] + "_vals, " +
                                            "debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + "_" + str(u) + "_" + self.mem_lvl

            elif "broadcast" in self.networkx_graph.nodes[u]['comment']:
                continue

            elif node_info["type"] == "repsiggen":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["index"] +
                                            "_" + str(u) + "_" + self.mem_lvl +
                                            " = RepeatSigGen(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl

            elif node_info["type"] == "repeat":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                            node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                                            " = Repeat(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] +\
                    node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "intersect":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["index"] + "_" + str(u) +
                                            "_" + self.mem_lvl + " = Intersect2(debug=debug_sim, " +
                                            "statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "union":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["index"] + "_" + str(u) +
                                            "_" + self.mem_lvl + " = Union2(debug=debug_sim, " +
                                            "statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
                # invalid_flag = 1
            elif node_info["type"] == "spaccumulator" and node_info["order"] == "1":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_" + self.mem_lvl +
                                            " = SparseAccumulator1(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["order"] + "_" + str(u) + "_" + self.mem_lvl
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_drop_crd_inner" + "_" + self.mem_lvl +
                                            " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_drop_crd_outer" + "_" + self.mem_lvl +
                                            " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_drop_val" + "_" + self.mem_lvl +
                                            " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            elif node_info["type"] == "spaccumulator" and node_info["order"] == "2":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_" + self.mem_lvl +
                                            " = SparseAccumulator2(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["order"] + "_" + str(u) + "_" + self.mem_lvl
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_drop_crd_inner" + "_" + self.mem_lvl +
                                            " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                                            u) + "_drop_crd_outer" + "_" + self.mem_lvl +
                                            " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                # f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                #    u) + "_drop_crd_in_2" + " = StknDrop(debug=debug_sim)\n")
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] +
                                            node_info["order"] + "_" + str(
                                            u) + "_drop_val" + "_" + self.mem_lvl +
                                            " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            elif node_info["type"] == "crddrop":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) +
                                            node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                            " = CrdDrop(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "crdhold":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) +
                                            node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                            " = CrdHold(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "mul":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) +
                                            node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                            " = Multiply2(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "add":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                            " = Add2(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "reduce":
                self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                            " = Reduce(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "fiberwrite":
                if node_info["mode"] == "vals":
                    self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                                node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl +
                                                " = ValsWrScan(size=" + array_size_computation(node_info["size"]) +
                                                ", fill=fill, debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] +\
                        node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl
                elif node_info["format"] == "compressed":
                    self.blks_to_realize.append(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                                node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl +
                                                " = CompressWrScan(seg_size=" +
                                                array_size_computation(node_info["segsize"]) + ", size=" +
                                                array_size_computation(node_info["crdsize"]) +
                                                ", fill=fill," + " debug=debug_sim, " +
                                                "statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] +\
                        node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl
                else:
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] +\
                        node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl
                    continue
                if node_info["sink"] == "true":
                    output_nodes[self.d[u]["object"]] = node_info["mode"]
            elif node_info["type"] == "memory_block":
                if True:  # not pipelined_tiles:
                    self.memory_blks.append(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl)
                    if self.mem_lvl == "00":
                        self.f.write(tab(1) + node_info["type"] + "_" + str(u) + "_" +
                                     self.mem_lvl + " = memory_block(" +
                                     "name= \"" + self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"Glb_memory\"]," +
                                     "bandwidth=memory_config[\"Glb_tile_bandwidth\"] // " +
                                     "memory_config[\"Glb_tiles\"]," +
                                     " latency=memory_config[\"Global_Glb_latency\"], debug=debug_sim)\n")
                    elif self.mem_lvl == "0":
                        self.f.write(tab(1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                     " = memory_block(" + "name= \"" +
                                     self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"Mem_memory\"]," +
                                     " bandwidth=memory_config[\"Mem_tile_bandwidth\"] // " +
                                     "memory_config[\"Mem_tiles\"]," +
                                     " latency=memory_config[\"Glb_Mem_latency\"], debug=debug_sim)\n")
                    else:
                        self.f.write(tab(1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                     " = memory_block(" + "name= \"" + self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"" + self.mem_lvl + "_memory\"], " +
                                     " bandwidth=memory_config[\"" + self.mem_lvl + "_tile_bandwidth\"] // " +
                                     "memory_config[\"" + self.mem_lvl + "_tiles\"]," +
                                     " latency=memory_config[\"" + self.mem_lvl + "_latency\"], debug=debug_sim)\n")
                else:
                    pass
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
                nxt_parents.append(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl)
                self.tensor_list.append(node_info["tensor"])
                if len(self.mem_lvl) == 2:
                    self.mem_blks_connect.add_parent_child(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl,
                                                           node_info["type"] + "_" + str(u) + "_" + self.mem_lvl[:-1],
                                                           ".out_done_in()")
                else:
                    self.mem_blks_connect.add_parent_child(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl,
                                                           "tiled_done", "")  # and not tile_signalled", "")
            else:
                invalid_flag = 1
        for par in parent:
            self.blks_to_realize.append(tab(self.scope_lvl + 1) + par + ".valid_tile_received()\n")
        if pipelined_tiles:
            for mem_nodes in self.pipelined_memory_nodes.keys():
                if self.mem_lvl == "00":
                    self.f.write(tab(1) + "memory_block_" + mem_nodes + self.mem_lvl[:-1] +
                                 "= memory_block(name = \"\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                 " element_size=memory_config[\"Bytes_per_element\"]," +
                                 "size=memory_config[\"Glb_memory\"]," +
                                 " bandwidth=memory_config[\"Glb_tile_bandwidth\"]," +
                                 " latency=memory_config[\"Global_Glb_latency\"], debug=debug_sim)\n")
                elif self.mem_lvl == "0":
                    self.f.write(tab(1) + "memory_block_" + mem_nodes + self.mem_lvl[:-1] +
                                 "= memory_block(name = \"\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                 " element_size=memory_config[\"Bytes_per_element\"]," +
                                 "size=memory_config[\"Mem_memory\"], bandwidth=memory_config[\"Mem_tile_bandwidth\"]," +
                                 " latency=memory_config[\"Glb_Mem_latency\"], debug=debug_sim)\n")
        self.output_check_nodes()
        self.f.write("\n")
        self.nxt_parent = nxt_parents
        return self.d, invalid_flag, output_nodes, nxt_parents, pipelined_tiles

    def write_nodes(self):
        for node in self.blks_to_realize:
            f.write(node)

    def node_instantiations(self, output_nodes, tens_fmt={}, tensor_information={},
                            tensor_format_parse=None, out_name=None, pipelined_tiles=False,
                            parent=[], tensor_list=[], whether_pipelined=False):
        invalid_flag = 0
        f.write(tab(self.scope_lvl))
        temp_flag = False
        nxt_parents = []
        self.f.write("\n")
        if whether_pipelined:
            for node in self.pipelined_memory_nodes.keys():
                self.f.write(tab(self.scope_lvl + 1) + "if check_flag" + self.mem_lvl + " and " + node + ".out_done():\n")
                self.f.write(tab(self.scope_lvl + 2) + "memory_node_" + node + ".check_if_done(True)\n")
                self.f.write(tab(self.scope_lvl + 1) + "elif check_flag" + self.mem_lvl + " and " + node +
                             ".out_done() and memory_node_" + node + self.mem_lvl + ".valid_tile():\n")
                generate_datasets_code(self.f, tens_fmt, self.scope_lvl + 1,
                                       tensor_information, tensor_format_parse,
                                       out_name, tiling=True, parents=parent, tensors=tensor_list,
                                       selected_tensor=self.pipelined_memory_nodes[node]["tensor"])
                if self.pipelined_memory_nodes[node]["node_type"] == "arrayvals":
                    self.f.write(tab(self.scope_lvl + 2) + node + " = Array(init_array=" +
                                 self.pipelined_memory_nodes[node]["tensor"] +
                                 "_vals, debug=debug_sim, statistics=report_stats, " +
                                 "fifo=in_fifo, back_en=backpressure, depth=depth)\n")
                elif self.pipelined_memory_nodes[node]["node_type"] == "fiberlookup":
                    self.f.write(tab(self.scope_lvl + 2) + node + " = CompressedCrdRdScan(crd_arr=" +
                                 self.pipelined_memory_nodes[node]["tensor"] + "_crd" +
                                 self.pipelined_memory_nodes[node]["mode"] + ", seg_arr=" +
                                 self.pipelined_memory_nodes[node]["tensor"] + "_seg" +
                                 self.pipelined_memory_nodes[node]["mode"] +
                                 ", debug=debug_sim, statsitics=report_stats," +
                                 " fifo=in_fifo, back_en=backpressure, depth=depth)\n")
                self.f.write(tab(self.scope_lvl + 2) + "memory_node_" + node + ".valid_tile_received()\n")

        for par in parent:
            if not temp_flag:
                temp_flag = True
                self.f.write(tab(self.scope_lvl + 1) + "if " + par + ".valid_tile()")
            else:
                self.f.write(" and " + par + ".valid_tile()")
        if len(parent) != 0:
            self.scope_lvl += 1
            self.f.write(":\n")
        if MEM_LEVELS > 0 and len(self.mem_lvl) == 0:
            assert(len(parent) == len(tensor_list))
            generate_datasets_code(self.f, tens_fmt, self.scope_lvl, tensor_information, tensor_format_parse,
                                   out_name, tiling=True, parents=parent, tensors=tensor_list)
        else:
            generate_datasets_code(self.f, tens_fmt, self.scope_lvl, tensor_information, tensor_format_parse,
                                   out_name, tiling=False, parents=parent, tensors=tensor_list)
        self.f.write(tab(self.scope_lvl + 1) + "check_flag" + self.mem_lvl + " = True\n")
        for u in list(nx.topological_sort(self.networkx_graph)):
            node_info = breakup_node_info(self.networkx_graph.nodes[u])
            self.d[u] = node_info
            u_val = u
            if (node_info["type"] == "fiberlookup" or node_info["type"] == "repeat") and node_info["root"] == "true":
                self.root_nodes.append(node_info["tensor"])
            if node_info["type"] == "fiberlookup":
                if node_info["format"] == "dense":
                    self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                 node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                                 " = UncompressCrdRdScan(dim=" + node_info["tensor"] +
                                 "_shape[" + node_info["mode"] + "]" + ", debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] +\
                        "_" + str(u) + "_" + self.mem_lvl

                if node_info["format"] == "compressed":
                    self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] +
                                 "_" + str(u) + "_" + self.mem_lvl + " = CompressedCrdRdScan(crd_arr=" + node_info["tensor"] +
                                 "_crd" + node_info["mode"] + ", seg_arr=" + node_info["tensor"] +
                                 "_seg" + node_info["mode"] + ", debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] +\
                        "_" + str(u) + "_" + self.mem_lvl

            elif node_info["type"] == "arrayvals":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                             "_" + str(u) + "_" + self.mem_lvl + " = Array(init_arr=" +
                             node_info["tensor"] + "_vals, " + "debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + "_" + str(u) + "_" + self.mem_lvl
            elif "broadcast" in self.networkx_graph.nodes[u]['comment']:
                continue
            elif node_info["type"] == "repsiggen":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" +
                             node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                             " = RepeatSigGen(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "repeat":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                             node_info["index"] + "_" + str(u) + "_" + self.mem_lvl +
                             " = Repeat(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] +\
                    node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "intersect":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["index"] + "_" +
                             str(u) + "_" + self.mem_lvl + " = Intersect2(debug=debug_sim, " +
                             "statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "union":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["index"] + "_" + str(u) +
                             "_" + self.mem_lvl + " = Union2(debug=debug_sim, " +
                             "statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u) + "_" + self.mem_lvl
                # invalid_flag = 1
            elif node_info["type"] == "spaccumulator" and node_info["order"] == "1":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_" + self.mem_lvl +
                             " = SparseAccumulator1(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["order"] + "_" + str(u) + "_" + self.mem_lvl
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_crd_inner" + "_" + self.mem_lvl +
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_crd_outer" + "_" + self.mem_lvl +
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_val" + "_" + self.mem_lvl +
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            elif node_info["type"] == "spaccumulator" and node_info["order"] == "2":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_" + self.mem_lvl +
                             " = SparseAccumulator2(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + node_info["order"] + "_" + str(u) + "_" + self.mem_lvl
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_crd_inner" + "_" + self.mem_lvl +
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_crd_outer" + "_" + self.mem_lvl +
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
                # f.write(tab(1) + node_info["type"] + node_info["order"] + "_" + str(
                #    u) + "_drop_crd_in_2" + " = StknDrop(debug=debug_sim)\n")
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + node_info["order"] + "_" + str(
                             u) + "_drop_val" + "_" + self.mem_lvl +
                             " = StknDrop(debug=debug_sim, statistics=report_stats)\n")
            elif node_info["type"] == "crddrop":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" +
                             self.mem_lvl + " = CrdDrop(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "crdhold":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" +
                             self.mem_lvl + " = CrdHold(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "mul":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" +
                             self.mem_lvl + " = Multiply2(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "add":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" +
                             self.mem_lvl + " = Add2(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "reduce":
                self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" +
                             self.mem_lvl + " = Reduce(debug=debug_sim, statistics=report_stats)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
            elif node_info["type"] == "fiberwrite":
                if node_info["mode"] == "vals":
                    self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                 node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl +
                                 " = ValsWrScan(size=" + array_size_computation(node_info["size"]) +
                                 ", fill=fill, debug=debug_sim, statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] +\
                        "_" + str(u) + "_" + self.mem_lvl
                elif node_info["format"] == "compressed":
                    self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + node_info["tensor"] +
                                 node_info["mode"] + "_" + str(u) + "_" + self.mem_lvl +
                                 " = CompressWrScan(seg_size=" + array_size_computation(node_info["segsize"]) + ", size=" +
                                 array_size_computation(node_info["crdsize"]) + ", fill=fill," + " debug=debug_sim, " +
                                 "statistics=report_stats)\n")
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] +\
                        "_" + str(u) + "_" + self.mem_lvl
                else:
                    self.d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] +\
                        "_" + str(u) + "_" + self.mem_lvl
                    continue
                if node_info["sink"] == "true":
                    output_nodes[self.d[u]["object"]] = node_info["mode"]
            elif node_info["type"] == "memory_block":
                if not pipelined_tiles:
                    self.memory_blks.append(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl)
                    if self.mem_lvl == "00":
                        self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) +
                                     "_" + self.mem_lvl + " = memory_block(" + "name= \"" +
                                     self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"Glb_memory\"]," +
                                     "bandwidth=memory_config[\"Glb_tile_bandwidth\"] // " +
                                     "memory_config[\"Glb_tiles\"]," +
                                     "latency=memory_config[\"Global_Glb_latency\"], debug=debug_sim)\n")
                    elif self.mem_lvl == "0":
                        self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) +
                                     "_" + self.mem_lvl + " = memory_block(" + "name= \"" +
                                     self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"Mem_memory\"]," +
                                     " bandwidth=memory_config[\"Mem_tile_bandwidth\"] // " +
                                     "memory_config[\"Mem_tiles\"]," +
                                     "latency=memory_config[\"Glb_Mem_latency\"], debug=debug_sim)\n")
                    else:
                        self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" + self.mem_lvl +
                                     " = memory_block(" + "name= \"" + self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"" + self.mem_lvl + "_memory\"], bandwidth=memory_config[\"" +
                                     self.mem_lvl + "_tile_bandwidth\"] // " +
                                     "memory_config[\"" + self.mem_lvl + "_tiles\"], latency=memory_config[\"" +
                                     self.mem_lvl + "_latency\"], debug=debug_sim)\n")
                else:
                    assert self.memory_channels is not None
                    tensor_indexes = self.memory_channels[node_info["tensor"]]
                    for ind in tensor_indexes:
                        self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_" + str(ind) +
                                     self.mem_lvl + " = memory_block(" + "name= \"" + self.mem_lvl +
                                     node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     " element_size=memory_config[\"Bytes_per_element\"]," +
                                     "size=memory_config[\"" + self.mem_lvl + "_memory\"], bandwidth=memory_config[\"" +
                                     self.mem_lvl + "_tile_bandwidth\"] // " +
                                     "memory_config[\"" + self.mem_lvl + "_tiles\"], latency=memory_config[\"" +
                                     self.mem_lvl + "_latency\"], debug=debug_sim)\n")
                    if len(self.mem_lvl) == 1:
                        self.f.write(tab(self.scope_lvl + 1) + node_info["type"] + "_" + str(u) + "_val" +
                                     self.mem_lvl + " = memory_block(" + "name= \"" + self.mem_lvl + node_info["tensor"] +
                                     "\", skip_blocks=skip_empty, nbuffer=nbuffer," +
                                     "element_size=memory_config[\"Bytes_per_element\"]," +
                                     " size=memory_config[\"" + self.mem_lvl + "_memory\"], bandwidth=memory_config[\"" +
                                     self.mem_lvl + "_tile_bandwidth\"] // " +
                                     "memory_config[\"" + self.mem_lvl + "_tiles\"], latency=memory_config[\"" +
                                     self.mem_lvl + "_latency\"], debug=debug_sim)\n")
                self.d[u]["object"] = node_info["type"] + "_" + str(u) + "_" + self.mem_lvl
                nxt_parents.append(node_info["type"] + "_" + str(u) + "_" + self.mem_lvl)
                if len(self.mem_lvl) == 2:
                    self.mem_blks_connect.add_parent_child(node_info["type"] + "_" + str(u) + "_" +
                                                           self.mem_lvl, node_info["type"] + "_" +
                                                           str(u) + "_" + self.mem_lvl[:-1] + ".out_done_in()")
                else:
                    self.mem_blks_connect.add_parent_child(node_info["type"] + "_" + str(u) + "_" +
                                                           self.mem_lvl, "tiled_done")  # and not tile_signalled")
            else:
                print(node_info)
                invalid_flag = 1
                print("Error invalid node detected", node_info["type"], "\n")
        for par in parent:
            self.f.write(tab(self.scope_lvl + 1) + par + ".valid_tile_received()\n")
        # if len(self.mem_lvl) == 0:
        #    self.f.write(tab(self.scope_lvl + 1) + "tile_signalled = False\n")
        self.output_check_nodes()
        self.f.write("\n")
        return self.d, invalid_flag, output_nodes, nxt_parents

    def fetch_block_parent(self, node):
        if self.parent_block is None:
            return ""
        par = self.parent_block.get_memory_structure()
        return ", " + par.get_parent(node)

    def get_memory_structure(self):
        return self.mem_blks_connect

    def get_memory_blocks(self):
        return self.memory_blks

    def connect_nodes(self, nodes_updating_list, data):
        if len(self.mem_lvl) == MEM_LEVELS:
            self.scope_lvl += 1
        else:
            self.scope_lvl -= 1
        self.f.write(tab(self.scope_lvl + 1) + "if check_flag" + self.mem_lvl + ":\n")
        nodes_updating_list.append(tab(self.scope_lvl + 1) + "if check_flag" + self.mem_lvl + ":\n")
        # self.scope_lvl -= 1
        for u in self.networkx_graph.nodes():
            if self.d[u]["type"] == "fiberlookup" and u not in data.get_if_done():
                if self.d[u]["root"] == "true":
                    self.f.write(tab(2 + self.scope_lvl) + "if len(in_ref_" + self.d[u]["tensor"] + self.mem_lvl + ") > 0:\n")
                    self.f.write(tab(3 + self.scope_lvl) + self.d[u]["object"] +
                                 ".set_in_ref(in_ref_" + self.d[u]["tensor"] + self.mem_lvl + ".pop(0))\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[u]["object"] + ".update()\n\n")
                    data.add_done(u)
        # print("FLAG ", apath, data.if_all_graph_realized(), " ", data.get_if_done())
        while not data.if_all_graph_realized():
            # print("FLAG ", apath, data.if_all_graph_realized(), " ", data.get_if_done())
            for u, v, _ in list(nx.edge_bfs(self.networkx_graph)):  # .edges(data=True), networkx_graph.nodes())):
                a = self.networkx_graph.get_edge_data(u, v)[0]
                if self.d[v]["type"] == "fiberlookup" and data.get_if_node_done(v) == 0 and\
                    parents_done(self.networkx_graph,
                                 data.get_if_done(), v):
                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_in_ref(" + self.d[u_]["object"] + ".out_ref" +
                                         str(self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()]) +
                                         "())\n")
                        else:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_in_ref(" +
                                         self.d[u_]["object"] + ".out_" +
                                         str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                        nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                        # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                        data.add_done(v)

                if self.d[v]["type"] == "repsiggen" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for u_ in data.get_parents()[v]:
                        self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                     ".set_istream(" + str(self.d[u_]["object"]).strip('"') +
                                     ".out_" + str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "repeat" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    if self.d[v]["root"] == "true":
                        self.f.write(tab(2 + self.scope_lvl) + "if len(in_ref_" +
                                     self.d[v]["tensor"] + self.mem_lvl + ") > 0:\n")
                        self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] +
                                     ".set_in_ref(in_ref_" + self.d[v]["tensor"] +
                                     self.mem_lvl + ".pop(0))\n")
                    for u_ in data.get_parents()[v]:
                        if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_in_ref(" +
                                         self.d[u_]["object"] + ".out_ref" +
                                         str(self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()]) +
                                         "())\n")
                        else:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_in_" +
                                         str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) +
                                         "(" + self.d[u_]["object"] + ".out_" +
                                         str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "arrayvals" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for u_ in data.get_parents()[v]:
                        if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                            print(self.scope_lvl, self.d[v], self.d[u_],
                                  self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()])
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_load(" + self.d[u_]["object"] + ".out_ref" +
                                         str(self.intersect_dataset[d[u_]["object"]][d[v]["tensor"].upper()]) + "())\n")
                        else:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_load(" + self.d[u_]["object"] + ".out_ref" + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)
                if self.d[v]["type"] == "memory_block" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    # print(intersect_dataset, d[u_]["object"], d[v]["tensor"])
                    if not self.pipelined_tiles:
                        for u_ in data.get_parents()[v]:
                            self.f.write(tab(2 + self.scope_lvl) +
                                         "if isinstance(" + self.d[u_]["object"] +
                                         ".out_ref(), int):\n")
                            if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                                print(self.scope_lvl, self.d[v], self.d[u_],
                                      self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()])
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] + ".add_tile(" +
                                             self.d[u_]["object"] + ".out_ref" +
                                             str(self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()]) +
                                             "(), sizes_dict_level" +
                                             self.mem_lvl + "[\"" + self.d[v]["tensor"] + "\"][" + self.d[u]["object"] +
                                             ".out_ref()]" + self.fetch_block_parent(self.d[v]["object"]) + ".token())\n")
                            else:
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] +
                                             ".add_tile(" + self.d[u_]["object"] +
                                             ".out_ref" + "(), sizes_dict_level" +
                                             self.mem_lvl + "[\"" + self.d[v]["tensor"] + "\"][" + self.d[u]["object"] +
                                             ".out_ref()]" + self.fetch_block_parent(self.d[v]["object"]) + ".token())\n")
                            self.f.write(tab(2 + self.scope_lvl) + "else:\n")
                            if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] +
                                             ".add_tile(" + self.d[u_]["object"] + ".out_ref" +
                                             str(self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()]) +
                                             "(), 8)\n")
                            else:
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] + ".add_tile(" +
                                             self.d[u_]["object"] + ".out_ref" + "(), 8)\n")
                    else:
                        ## Need to make this run for all possible memory nodes
                        # Also need to make initialization of each blocks
                        for u_ in data.get_parents()[v]:
                            self.f.write(tab(2 + self.scope_lvl) + "if isinstance(" +
                                         self.d[u_]["object"] + ".out_ref(), int):\n")
                            if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                                print(self.scope_lvl, self.d[v], self.d[u_],
                                      self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()], )
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] + ".add_tile(" +
                                             self.d[u_]["object"] + ".out_ref" +
                                             str(self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()]) +
                                             "(), sizes_dict_level" +
                                             self.mem_lvl + "[\"" + self.d[v]["tensor"] + "\"][" + self.d[u]["object"] +
                                             ".out_ref()]" + self.fetch_block_parent(self.d[v]["object"]) + ".token())\n")
                            else:
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] + ".add_tile(" +
                                             self.d[u_]["object"] + ".out_ref" + "(), sizes_dict_level" +
                                             self.mem_lvl + "[\"" + self.d[v]["tensor"] + "\"][" + self.d[u]["object"] +
                                             ".out_ref()]" + self.fetch_block_parent(self.d[v]["object"]) + ".token())\n")
                            self.f.write(tab(2 + self.scope_lvl) + "else:\n")
                            if "intersect" in self.d[u_]["object"] or "union" in self.d[u_]["object"]:
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] +
                                             ".add_tile(" + self.d[u_]["object"] + ".out_ref" +
                                             str(self.intersect_dataset[self.d[u_]["object"]][self.d[v]["tensor"].upper()]) +
                                             "(), 8)\n")
                            else:
                                self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] +
                                             ".add_tile(" + self.d[u_]["object"] + ".out_ref" + "(), 8)\n")
                    self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".check_if_done(" +
                                 self.mem_blks_connect.get_child(self.d[v]["object"]) + ")\n")
                    # if len(self.mem_lvl) < MEM_LEVELS:
                    #     if len(self.mem_lvl)
                    #     par = self.mem_blks_connect.get_parent(self.d[v]["object"])
                    #     self.f.write(tab(3 + self.scope_lvl) + self.d[v]["object"] +
                    #                  ".check_done(" + par + ".get_done_in())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update(time_cnt)\n")
                    data.add_done(v)

                if self.d[v]["type"] == "intersect" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        if i % 2 == 1:
                            continue
                        u_ = data.get_parents()[v][i]
                        self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_in" + str((i) // 2 + 1) + "(" +
                                     self.d[u_]["object"] + ".out_ref(), " + self.d[u_]["object"] + ".out_crd())\n")
                        self.intersect_dataset[self.d[v]["object"]][self.d[u_]["tensor"].upper()] = i // 2 + 1
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "union" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        if i % 2 == 1:
                            continue
                        u_ = data.get_parents()[v][i]
                        self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_in" + str((i) // 2 + 1) + "(" +
                                     self.d[u_]["object"] + ".out_ref(), " + self.d[u_]["object"] + ".out_crd())\n")
                        self.intersect_dataset[self.d[v]["object"]][self.d[u_]["tensor"].upper()] = i // 2 + 1
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "crddrop" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for u_ in data.get_parents()[v]:
                        index_value = data.get_edge_data()[v][data.get_parents()[v].index(u_)][-1]
                        if index_value == self.d[v]["inner"]:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_inner_crd" +
                                         "(" + self.d[u_]["object"] + ".out_crd())\n")
                        if index_value == self.d[v]["outer"]:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_outer_crd" +
                                         "(" + self.d[u_]["object"] + ".out_crd())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "crdhold" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        index_value = data.get_edge_data()[v][i][-1]
                        local_edge = data.get_edge_data()[v][i][:-2]
                        if index_value == self.d[v]["inner"]:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_inner_crd" + "(" + self.d[u_]["object"] +
                                         ".out_" + local_edge + "())\n")
                        if index_value == self.d[v]["outer"]:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_outer_crd" + "(" + self.d[u_]["object"] +
                                         ".out_" + local_edge + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n")
                    data.add_done(v)

                if self.d[v]["type"] == "spaccumulator" and self.d[v]["order"] == "1" and parents_done(self.networkx_graph,
                                                                                                       data.get_if_done(), v) \
                        and data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        local_edge = ""
                        if "crd" in data.get_edge_data()[v][i]:
                            local_edge = data.get_edge_data()[v][i][:-2]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         "_drop_" + local_edge + ".set_in_stream(" +
                                         self.d[u_]["object"] + ".out_" + local_edge + "())\n")
                        else:
                            local_edge = data.get_edge_data()[v][i]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         "_drop_" + local_edge + ".set_in_stream(" +
                                         self.d[u_]["object"] + ".out_val())\n")
                        nodes_updating_list.append(tab(2 + self.scope_lvl) +
                                                   self.d[v]["object"] + "_drop_" + local_edge + ".update()\n")
                        # f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".update()\n")

                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        local_edge = ""
                        if "crd" in data.get_edge_data()[v][i]:
                            local_edge = data.get_edge_data()[v][i][:-2]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_" + local_edge + "(" +
                                         self.d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                        else:
                            local_edge = data.get_edge_data()[v][i]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_" + local_edge + "(" +
                                         self.d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "spaccumulator" and self.d[v]["order"] == "2" and parents_done(self.networkx_graph,
                                                                                                       data.get_if_done(), v) \
                        and data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        local_edge = ""
                        if "crd" in data.get_edge_data()[v][i]:
                            local_edge = data.get_edge_data()[v][i][:-2]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         "_drop_" + local_edge + ".set_in_stream(" +
                                         self.d[u_]["object"] + ".out_" + local_edge + "())\n")
                        else:
                            local_edge = data.get_edge_data()[v][i]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         "_drop_" + local_edge + ".set_in_stream(" +
                                         self.d[u_]["object"] + ".out_val())\n")
                        nodes_updating_list.append(tab(2 + self.scope_lvl) +
                                                   self.d[v]["object"] + "_drop_" +
                                                   local_edge + ".update()\n")
                        # f.write(tab(2) + d[v]["object"] + "_drop_" + local_edge + ".update()\n")

                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        local_edge = ""
                        if "crd" in data.get_edge_data()[v][i]:
                            local_edge = data.get_edge_data()[v][i][:-2]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_" + local_edge + "(" +
                                         self.d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                        else:
                            local_edge = data.get_edge_data()[v][i]
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_" + local_edge + "(" +
                                         self.d[v]["object"] + "_drop_" + local_edge + ".out_val())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "mul" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                     ".set_in" + str(data.get_parents()[v].index(u_) + 1) + "(" +
                                     self.d[u_]["object"] + ".out_" + str(data.get_edge_info(v, i)) + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "add" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for u_ in data.get_parents()[v]:
                        self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                     ".set_in" + str(data.get_parents()[v].index(u_) + 1) + "(" +
                                     self.d[u_]["object"] + ".out_" + str(data.get_edge_info(v, i)) + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "reduce" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for u_ in data.get_parents()[v]:
                        self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_in_" +
                                     str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) +
                                     "(" + self.d[u_]["object"] +
                                     ".out_" + str(data.get_edge_data()[v][data.get_parents()[v].index(u_)]) + "())\n")
                    nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    # f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    data.add_done(v)

                if self.d[v]["type"] == "fiberwrite" and parents_done(self.networkx_graph, data.get_if_done(), v) and \
                        data.get_if_node_done(v) == 0:
                    for i in range(len(data.get_parents()[v])):
                        u_ = data.get_parents()[v][i]
                        if "val" not in data.get_edge_data()[v][i] and "spaccumulator" \
                                in self.d[u_]["object"]:
                            local_index = data.get_edge_data()[v][i][-1]
                            print(self.d[u_], " ", local_index, " ", apath)
                            if self.d[u_]["in0"] == local_index:
                                local_cord = "_inner"
                            else:
                                local_cord = "_outer"
                            data.set_edge_data(v, i, "crd" + local_cord)
                        if self.d[v]["mode"] == "vals":
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".set_input(" +
                                         self.d[u_]["object"] + ".out_" +
                                         str(data.get_edge_info(v, i)) + "())\n")
                            nodes_updating_list.append(tab(2 + self.scope_lvl) +
                                                       self.d[v]["object"] + ".update()\n")
                        else:
                            self.f.write(tab(2 + self.scope_lvl) + self.d[v]["object"] +
                                         ".set_input(" + self.d[u_]["object"] + ".out_" +
                                         str(data.get_edge_info(v, i)) + "())\n")
                            nodes_updating_list.append(tab(2 + self.scope_lvl) + self.d[v]["object"] + ".update()\n")
                    data.add_done(v)

    def finish_outputs_tiled(self, elements, nodes_completed):
        self.scope_lvl += 2
        for i in nodes_completed:
            self.f.write(i)
        self.f.write("\n")
        output_list = ""
        # Write done statement
        self.f.write(tab(self.scope_lvl) + "tiled_done = ")
        elements2 = []
        for elem in elements.keys():
            elements2.append(elem)
        for elem in elements.keys():
            self.f.write(elem + ".out_done()")
            if elem != elements2[-1]:
                self.f.write(" and ")
            else:
                self.f.write("\n")
        # TIme counter update
        # self.f.write(tab(2) + "time_cnt += 1\n\n")
        # Autosize all blocks
        self.f.write(tab(self.scope_lvl) + "if tiled_done:\n")  # and not tile_signalled:\n")
        # self.f.write(tab(self.scope_lvl + 1) + "tile_signalled = True\n")
        for elem in elements.keys():
            self.f.write(tab(self.scope_lvl + 1) + elem + ".autosize()\n")
        self.f.write("\n")

        if len(elements.keys()) > 1:
            self.f.write(tab(self.scope_lvl + 1) + "out_crds = [")
        else:
            self.f.write(tab(self.scope_lvl + 1) + "out_crds = []\n")

        output_list += " out_crds = ["
        count = 0
        for elem in elements.keys():
            if elements[elem] != "vals":
                self.f.write(elem + ".get_arr()")
                output_list += elem + ".get_arr()"
                count += 1
                if count < len(elements2) - 1:
                    self.f.write(", ")
                    output_list += ", "
                else:
                    self.f.write("]\n")
                    output_list += "]"
        count = 0
        if len(elements.keys()) > 1:
            self.f.write(tab(self.scope_lvl + 1) + "out_segs = [")
        else:
            self.f.write(tab(self.scope_lvl + 1) + "out_segs = []\n")

        output_list += ", out_segs = ["
        for elem in elements.keys():
            if elements[elem] != "vals":
                self.f.write(elem + ".get_seg_arr()")
                output_list += elem + ".get_seg_arr()"
                count += 1
                if count < len(elements2) - 1:
                    self.f.write(", ")
                    output_list += ", "
                else:
                    self.f.write("]\n")
                    output_list += "]"

        self.f.write(tab(self.scope_lvl + 1) + "out_vals = ")
        output_string = ", out_vals = "
        for elem in elements.keys():
            if elements[elem] == "vals":
                self.f.write(elem + ".get_arr()\n")
                output_string += elem + ".get_arr()"
        output_list += output_string
        return output_list

    def return_next_parent(self):
        return self.nxt_parent

    def generate_check_against_gold_code(self, tensor_format_parse, test_name, parents=[], tensors=[]):
        f.write(tab(self.scope_lvl + 1) + "if check_gold:\n")
        generate_tiling_output_crds(f, self.scope_lvl + 2, parents, tensors)
        f.write(tab(self.scope_lvl + 2) + "print(\"Checking gold...\")\n")
        f.write(tab(self.scope_lvl + 2) + "check_gold_")
        check = out_name[num]
        check = get_common_test_name(check)
        f.write(check + "(" + get_dataset_name(test_name) + ", debug_sim, cast, out_crds, out_segs, out_vals, \"" +
                tensor_format_parse.get_format(output_tensor) + "\")\n")
        f.write(tab(self.scope_lvl + 2) + "samBench(bench, extra_info)\n")


### START OF THE CODE GENERATION FOR THE TESTS
parser = argparse.ArgumentParser("Generate sam apps/")
parser.add_argument("--input_dir", type=str, default="./compiler/sam-outputs/dot")
parser.add_argument("--memory_level", type=str, default=2)
args = parser.parse_args()
MEM_LEVELS = int(args.memory_level)
file_paths, out_name = get_all_files(args.input_dir)
num = 0
for apath in file_paths:
    print("______________________________________________")
    print(apath)
    graphs = pydot.graph_from_dot_file(apath)
    graph = graphs[0]
    networkx_graph = nx.nx_pydot.from_pydot(graph)
    tensor_format_parse = TensorFormat()
    tensor_format_parse.set_all_tensors(apath)
    f = open(out_name[num] + ".py", "w")
    generate_header(f, out_name[num])
    networkx_graph = remove_broadcast_nodes(networkx_graph)
    print(" remove broadcast niodes  ")
    # Node Dataset present
    d = {}
    invalid_flag = 0
    root_nodes = []
    output_nodes = {}
    tensor_information = {}
    nodes_updating_list = []
    print(" breakip and get tensor info ")
    for u in list(nx.topological_sort(networkx_graph)):
        node_info = breakup_node_info(networkx_graph.nodes[u])
        if node_info["type"] == "fiberlookup":
            if node_info["tensor"] not in tensor_information:
                tensor_information[node_info["tensor"]] = 0
            if node_info["format"] == "compressed":
                tensor_information[node_info["tensor"]] += 1 * (2 ** int(node_info["mode"]))
    print("mem lvlv starting")
    # generate_tiling_header()
    if MEM_LEVELS == 2:
        generate_tiling_header(f, out_name[num])
        tiling_graph, pipelined_memory_nodes = generate_tiling_graph(networkx_graph)
        d = {}
        mem_blks = []
        glb_lvl = GraphRealization(tiling_graph, mem_lvl="00", scope_lvl=0, f=f,
                                   mem_blks=mem_blks, pipelined_memory_nodes=pipelined_memory_nodes)
        d, invalid_flag, output_nodes, parents, whether_pipelined = glb_lvl.node_instantiations_mem(output_nodes)
        data = CodeGenerationdatasets(tiling_graph)
        data.build_datasets(tiling_graph)
        output_check_nodes(f, root_nodes)
        f.write("\n\n")
        d = {}
        mem_lvl = GraphRealization(tiling_graph, mem_lvl="0", scope_lvl=1, f=f,
                                   parent=glb_lvl, mem_blks=glb_lvl.get_memory_blocks(),
                                   pipelined_memory_nodes=pipelined_memory_nodes)
        d, invalid_flag, output_nodes, parents, whether_pipelined =\
            mem_lvl.node_instantiations_mem(output_nodes,
                                            pipelined_tiles=True,
                                            parent=parents,
                                            whether_pipelined=whether_pipelined)
        glb_lvl.loop_start()
        glb_lvl.write_nodes()
        glb_lvl.connect_nodes(nodes_updating_list, data)
        data = CodeGenerationdatasets(tiling_graph)
        data.build_datasets(tiling_graph)
        mem_lvl.write_nodes()
        mem_lvl.connect_nodes(nodes_updating_list, data)
        tens_fmt = {}
        count = 0
        comp_lvl = GraphRealization(networkx_graph, mem_lvl="",
                                    scope_lvl=1, f=f,
                                    pipelined_memory_nodes=pipelined_memory_nodes)
        data_formats = gen_data_formats(len(tensor_format_parse.return_all_tensors()), out_name[num], apath)
        ct = 0
        for k in tensor_format_parse.return_all_tensors():
            if ct != 0:
                tens_fmt[k] = {}
                tens_fmt[k]["information"] = data_formats[ct - 1]
            ct += 1
        f.write("\n")
        d, invalid_flag, output_nodes, parents =\
            comp_lvl.node_instantiations(output_nodes, tens_fmt, tensor_information,
                                         tensor_format_parse, out_name[num], parent=parents,
                                         tensor_list=mem_lvl.get_tensor_list(),
                                         whether_pipelined=whether_pipelined)
        if invalid_flag == 1:
            os.system("rm " + out_name[num] + ".py")
            print(out_name[num] + " failed\n")
            num += 1
            continue
        f.write("\n")
        data = CodeGenerationdatasets(networkx_graph)
        data.build_datasets(networkx_graph)
        comp_lvl.connect_nodes(nodes_updating_list, data)
        output_tensor = ""
        ct = 0
        for k in tensor_format_parse.return_all_tensors():
            if ct == 0:
                output_tensor = k
            ct += 1
        sorted_nodes = sort_output_nodes(output_nodes, tensor_format_parse.get_format(output_tensor))
        output_list = comp_lvl.finish_outputs_tiled(sorted_nodes, nodes_updating_list)
        comp_lvl.generate_check_against_gold_code(tensor_format_parse, out_name[num],
                                                  parents=mem_lvl.return_next_parent(),
                                                  tensors=mem_lvl.get_tensor_list())
        mem_blks = mem_lvl.get_memory_blocks()
        generate_tiling_end(f, mem_blks)

    elif MEM_LEVELS != 0:
        # NOT FULLY IMPLEMENTED YET
        generate_tiling_header(f, out_name[num])
        tiling_graph = generate_tiling_graph(networkx_graph)
        mem_blks = []
        parents = []
        for mem_loop in reverse(range(MEM_LEVELS)):
            mem_string = "0" * mem_loop
            d = {}
            mem_lvl = GraphRealization(tiling_graph, mem_lvl=mem_string, scope_lvl=MEM_LEVELS - mem_loop,
                                       f=f, mem_blks=mem_blks)
            d, invalid_flag, output_nodes, parents, whether_pipelined =\
                mem_lvl.node_instantiations_mem(output_nodes,
                                                parent=parents,
                                                whether_pipelined=whether_pipelined)
        data = CodeGenerationdatasets(tiling_graph)
        data.build_datasets(tiling_graph)
        output_check_nodes(f, root_nodes)
        f.write("\n")
        d = {}
        glb_lvl.loop_start()
        glb_lvl.write_nodes()
        glb_lvl.connect_nodes(nodes_updating_list, data)
        data = CodeGenerationdatasets(tiling_graph)
        data.build_datasets(tiling_graph)
        mem_lvl.write_nodes()
        mem_lvl.connect_nodes(nodes_updating_list, data)
        tens_fmt = {}
        count = 0
        comp_lvl = GraphRealization(networkx_graph, mem_lvl="", scope_lvl=1, f=f)
        data_formats = gen_data_formats(len(tensor_format_parse.return_all_tensors()), out_name[num], apath)
        ct = 0
        for k in tensor_format_parse.return_all_tensors():
            if ct != 0:
                tens_fmt[k] = {}
                tens_fmt[k]["information"] = data_formats[ct - 1]
            ct += 1
        f.write("\n")
        print(parents)
        print(mem_lvl.get_tensor_list())
        assert len(parents) == len(mem_lvl.get_tensor_list())
        d, invalid_flag, output_nodes, parents = comp_lvl.node_instantiations(output_nodes, tens_fmt,
                                                                              tensor_information, tensor_format_parse,
                                                                              out_name[num], parent=parents,
                                                                              tensor_list=mem_lvl.get_tensor_list(),
                                                                              whether_pipelined=whether_pipelined)
        if invalid_flag == 1:
            os.system("rm " + out_name[num] + ".py")
            print(out_name[num] + " failed\n")
            num += 1
            continue
        f.write("\n")
        data = CodeGenerationdatasets(networkx_graph)
        data.build_datasets(networkx_graph)
        comp_lvl.connect_nodes(nodes_updating_list, data)
        output_tensor = ""
        ct = 0
        for k in tensor_format_parse.return_all_tensors():
            if ct == 0:
                output_tensor = k
            ct += 1
        sorted_nodes = sort_output_nodes(output_nodes, tensor_format_parse.get_format(output_tensor))
        output_list = comp_lvl.finish_outputs_tiled(sorted_nodes, nodes_updating_list)
        comp_lvl.generate_check_against_gold_code(tensor_format_parse, out_name[num],
                                                  parents=mem_lvl.return_next_parent(),
                                                  tensors=mem_lvl.get_tensor_list())
        mem_blks = mem_lvl.get_memory_blocks()
        generate_tiling_end(f, mem_blks)
    else:
        tens_fmt = {}
        count = 0
        comp_lvl = GraphRealization(networkx_graph, mem_lvl="", scope_lvl=0, f=f)
        data_formats = gen_data_formats(len(tensor_format_parse.return_all_tensors()),
                                        out_name[num], apath)
        ct = 0
        for k in tensor_format_parse.return_all_tensors():
            if ct != 0:
                tens_fmt[k] = {}
                tens_fmt[k]["information"] = data_formats[ct - 1]
            ct += 1
        # generate_datasets_code(f, tens_fmt, 1, tensor_information, tensor_format_parse, out_name[num])
        f.write("\n")
        d, invalid_flag, output_nodes, parents = comp_lvl.node_instantiations(output_nodes, tens_fmt,
                                                                              tensor_information,
                                                                              tensor_format_parse,
                                                                              out_name[num],
                                                                              parent=[])
        if invalid_flag == 1:
            os.system("rm " + out_name[num] + ".py")
            print(out_name[num] + " failed\n")
            num += 1
            continue
        output_check_nodes(f, root_nodes)
        f.write("\n")
        f.write(tab(1) + "while not done and time_cnt < TIMEOUT:\n")
        intersect_dataset = defaultdict(dict)
        data = CodeGenerationdatasets()
        data.build_datasets(networkx_graph)
        comp_lvl.connect_nodes(nodes_updating_list, data)
        output_tensor = ""
        ct = 0
        for k in tensor_format_parse.return_all_tensors():
            if ct == 0:
                output_tensor = k
            ct += 1

        sorted_nodes = sort_output_nodes(output_nodes, tensor_format_parse.get_format(output_tensor))
        output_list = finish_outputs(f, sorted_nodes, nodes_updating_list)

        generate_benchmarking_code(f, tensor_format_parse, out_name[num])
        generate_check_against_gold_code(f, tensor_format_parse, out_name[num])
    f.close()
    os.system("cp " + out_name[num] + ".py " + os.getcwd() + "/sam/sim/test/apps/test_" + out_name[num] + ".py")
    os.system("rm " + out_name[num] + ".py")
    num += 1
