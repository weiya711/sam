import pydot
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


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
        read_f = open(apath, "r")
        comment_line = read_f.readlines()
        comment = comment_line[1]
        comment = comment[comment.index("\"") + 1: -1]
        comment = comment[0: comment.index("\"")]
        comment = comment.split(",")
        for tensor_info in comment:
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


def tab(a):
    ans = ""
    for i in range(a):
        ans += "    "
    return ans


def generate_header(f, out_name):
    f.write("import scipy.sparse\n")
    f.write("from sam.sim.src.rd_scanner import UncompressRdScan, CompressedRdScan\n")
    f.write("from sam.sim.src.wr_scanner import ValsWrScan\n")
    f.write("from sam.sim.src.joiner import Intersect2\n")
    f.write("from sam.sim.src.compute import Multiply2\n")
    f.write("from sam.sim.src.crd_manager import CrdDrop\n")
    f.write("from sam.sim.src.repeater import Repeat, RepeatSigGen\n")
    f.write("from sam.sim.src.accumulator import Reduce\n")
    f.write("from sam.sim.test.test import *\n")
    f.write("import os\n")
    f.write("cwd = os.getcwd()\n")
    f.write("formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))\n\n\n")
    # f.write("formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default = './mode-formats')\n\n")
    f.write("# FIXME: Figureout formats\n")
    f.write("@pytest.mark.skipif(\n")
    f.write(tab(1) + "os.getenv('CI', 'false') == 'true',\n")
    f.write(tab(1) + "reason='CI lacks datasets',\n")
    f.write(")\n")
    f.write("def test_" + out_name + "(ssname, debug_sim, fill=0):\n")


def generate_datasets_code(f, tensor_formats, scope_lvl, tensor_info, tensor_format_parse):
    # Assuming tje format is csr and csc:
    for ten in tensor_format_parse.return_all_tensors():
        if tensor_format_parse.get_location(ten) == 0:
            continue
        f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname, \"" +
                tensor_formats[ten]["information"] + "\", \"" + tensor_format_parse.get_format(ten) + "\")\n")
        f.write(tab(scope_lvl) + ten + "_shape_filename = os.path.join(" + ten + "_dirname, \"" + ten + "_shape.txt\")\n")
        f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
        # if tensor_formats.get_format(ten) == "dense":
        #    f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname + \"" +
        #       tensor_formats[ten]["information"]  + "\" \"" + tensor_formats[ten]["format"] + "\")\n")
        #    f.write(tab(scope_lvl) + ten + "_shape_filename = os.name.join(" + ten + "_dirname, \"" +
        #       ten + "_shape.txt\")\n")
        #    f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
        if tensor_format_parse.get_format(ten) == "ds01":
            # f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname + \"" +
            #   tensor_formats[ten]["information"]  + "\" \"" + tensor_formats[ten]["format"] + "\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape_filename = os.name.join(" + ten + "_dirname, \"" +
            #   ten + "_shape.txt\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
            # f.write(tab(scope_lvl) + ten + "_seg1" +  " = read_inputs(" + ten + "1_crd_filename)")
        elif tensor_format_parse.get_format(ten) == "ds10":
            # f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname + \"" +
            #   tensor_formats[ten]["information"]
            #   + "\" \"" + tensor_formats[ten]["format"] + "\" +  )\n")
            # f.write(tab(scope_lvl) + ten + "_shape_filename = os.name.join(" + ten + "_dirname, \"" +
            # ten + "_shape.txt\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "0_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "0_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            # f.write(tab(scope_lvl) + ten + "_seg1" +  " = read_inputs(" + ten + "1_crd_filename)")
        elif tensor_format_parse.get_format(ten) == "ss01":
            # f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname + \""+
            #   tensor_formats[ten]["information"] + "\" \"" +  tensor_formats[ten]["format"] + "\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape_filename = os.name.join(" + ten + "_dirname, \"" + ten
            #   + "_shape.txt\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n"
            f.write(tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "0_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "0_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
        elif tensor_format_parse.get_format(ten) == "ss10":
            # f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname + "+
            #   tensor_formats[ten]["information"] + "\" \"" + tensor_formats[ten]["format"] + "\" +  )\n")
            # f.write(tab(scope_lvl) + ten + "_shape_filename = os.name.join(" + ten + "_dirname, \"" + ten + "_shape.txt\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "0_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "0_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg0" + " = read_inputs(" + ten + "0_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "0_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "0_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd0" + " = read_inputs(" + ten + "0_crd_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")

        elif tensor_format_parse.get_format(ten) == "dss012":
            # f.write(tab(scope_lvl) + ten + "_dirname = os.path.join(formatted_dir, ssname + "+
            #   tensor_formats[ten]["information"] + "\" \"" + tensor_formats[ten]["format"] + "\" +  )\n")
            # f.write(tab(scope_lvl) + ten + "_shape_filename = os.name.join(" + ten + "_dirname, \"" + ten +
            #   "_shape.txt\")\n")
            # f.write(tab(scope_lvl) + ten + "_shape = read_inputs(" + ten + "_shape_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "1_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg1" + " = read_inputs(" + ten + "1_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "1_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "1_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd1" + " = read_inputs(" + ten + "1_crd_filename)\n\n")
            f.write(tab(scope_lvl) + ten + "2_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + "2_seg.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_seg2" + " = read_inputs(" + ten + "2_seg_filename)\n")
            f.write(tab(scope_lvl) + ten + "2_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + "2_crd.txt\")\n")
            f.write(tab(scope_lvl) + ten + "_crd2" + " = read_inputs(" + ten + "2_crd_filename)\n\n")
        else:
            ct = 0
            for i in range(len(tensor_format_parse.get_format(ten))):
                if tensor_format_parse.get_format(ten)[i] == "s":
                    f.write(tab(scope_lvl) + ten + str(i) + "_seg_filename = os.path.join(" + ten + "_dirname, \"" + ten + str(i) + "_seg.txt\")\n")
                    f.write(tab(scope_lvl) + ten + "_seg" + str(i) + " = read_inputs(" + ten + str(i) + "_seg_filename)\n")
                    f.write(tab(scope_lvl) + ten + str(i) + "_crd_filename = os.path.join(" + ten + "_dirname, \"" + ten + str(i) + "_crd.txt\")\n")
                    f.write(tab(scope_lvl) + ten + "_crd" + str(i) + " = read_inputs(" + ten + str(i) + "_crd_filename)\n\n")
        f.write(tab(scope_lvl) + ten + "_vals_filename = os.path.join(" + ten + "_dirname, \"" + ten + "_vals.txt\")\n")
        f.write(tab(scope_lvl) + ten + "_vals" + " = read_inputs(" + ten + "_vals_filename, float)\n\n")


def gen_data_formats(size, app_name, path):
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
        if "vecmul" in app_name:
            ans_list = ["orig", "shift"]
            return ans_list
        else:
            ans_list = ["dummy", "dummy"]

            return ans_list
    else:
        for i in range(size - 1):
            ans_list.append("dummy")
        return ans_list


def output_check_nodes(f, root_nodes):
    for r in root_nodes:
        f.write(tab(1) + "in_ref_" + str(r) + " = [0, 'D']\n")
    f.write(tab(1) + "done = False\n")
    f.write(tab(1) + "time = 0\n")


def finish_outputs(f, elements):
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
    f.write(tab(2) + "time += 1\n\n")
    for elem in elements.keys():
        f.write(tab(1) + elem + ".autosize()\n")
    f.write("\n")
    f.write(tab(1) + "out_crds = [")
    count = 0
    for elem in elements.keys():
        if elements[elem] != "vals":
            f.write(elem + ".get_arr()")
            count += 1
            if count < len(elements2) - 1:
                f.write(", ")
            else:
                f.write("]\n")
    count = 0
    f.write(tab(1) + "out_segs = [")
    for elem in elements.keys():
        if elements[elem] != "vals":
            f.write(elem + ".get_seg_arr()")
            count += 1
            if count < len(elements2) - 1:
                f.write(", ")
            else:
                f.write("]\n")
    f.write(tab(1) + "out_vals = ")
    for elem in elements.keys():
        if elements[elem] == "vals":
            f.write(elem + ".get_arr()\n")


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


def breakup_node_info(node_name):
    d = dict(x.split("=") for x in node_name[1: -1].split(","))
    return d


def remove_broadcast_nodes(G):
    g = G.copy()
    for a in g:
        g0 = g.copy()
        node_i = breakup_node_info(g.nodes[a]["comment"])
        if node_i["type"] == "broadcast":
            for preds in g0.predecessors(a):
                for succs in g0.neighbors(a):
                    g0.add_edge(preds, succs, **(g0.get_edge_data(preds, a)[0]))
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


directory = './compiler/sam-outputs/dot'
file_paths = []
out_name = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if filename[0] == ".":
        continue
    out_name.append(filename[0:-3])
    # checking if it is a file
    print(out_name[-1])
    if os.path.isfile(f):
        file_paths.append(f)
print(file_paths)
print(len(file_paths))
num = 0
for apath in file_paths:
    # apath = "../compiler/sam-outputs/dot/matmul_ijk.gv"
    # out_name[0] = "matmul_ijk"
    # apath = os.path.join(directory, ".gv")
    graphs = pydot.graph_from_dot_file(apath)
    graph = graphs[0]
    networkx_graph = nx.nx_pydot.from_pydot(graph)

    tensor_format_parse = TensorFormat()
    tensor_format_parse.set_all_tensors(apath)
    f = open(out_name[num] + ".py", "w")
    generate_header(f, out_name[num])
    # nx.draw(networkx_graph, with_labels=True)
    # plt.savefig("Graph.png", format="PNG")
    # plt.show()
    #
    networkx_graph = remove_broadcast_nodes(networkx_graph)
    # nx.draw(networkx_graph, with_labels=True)
    # plt.savefig("Graph_pruned.png", format="PNG")
    # plt.show()
    d = {}
    invalid_flag = 0
    root_nodes = []
    output_nodes = {}
    tensor_information = {}

    for u in list(nx.topological_sort(networkx_graph)):
        node_info = breakup_node_info(networkx_graph.nodes[u]["comment"])
        if node_info["type"] == "fiberlookup":
            if node_info["tensor"] not in tensor_information:
                tensor_information[node_info["tensor"]] = 0
            if node_info["format"] == "compressed":
                tensor_information[node_info["tensor"]] += 1 * (2 ** int(node_info["mode"]))
    tens_fmt = {}
    count = 0
    # for k in tensor_information.keys():
    #    count += 1
    #    temp = ""
    #    if tensor_information[k] == 0:
    #        tens_fmt[k] = {}
    #        tens_fmt[k]["format"] = "dense"
    #        temp = "dd00"
    #    elif tensor_information[k] == 3:
    #        tens_fmt[k] = {}
    #        tens_fmt[k]["format"] = "dcsr"
    #        temp = "ss01"
    #    elif tensor_information[k] == 1:
    #        tens_fmt[k] = {}
    #        tens_fmt[k]["format"] = "csc"
    #        temp = "ds10"
    #    elif tensor_information[k] == 2:
    #        tens_fmt[k] = {}
    #        tens_fmt[k]["format"] = "csr"
    #        temp = "ds10"
    #    else:
    #        tens_fmt[k] = {}
    #        tens_fmt[k]["format"] = "cccc"
    data_formats = gen_data_formats(len(tensor_format_parse.return_all_tensors()), out_name[num], apath)
    ct = 0
    for k in tensor_format_parse.return_all_tensors():
        if ct != 0:
            tens_fmt[k] = {}
            tens_fmt[k]["information"] = data_formats[ct - 1]
        ct += 1
    # tens_fmt["B"]["information"] = data_formats[0]
    # tens_fmt["C"]["information"] = data_formats[1]
    generate_datasets_code(f, tens_fmt, 1, tensor_information, tensor_format_parse)
    node_number = []
    for u in list(nx.topological_sort(networkx_graph)):
        node_info = breakup_node_info(networkx_graph.nodes[u]["comment"])
        d[u] = node_info
        if node_info["type"] == "fiberlookup" and node_info["tensor"] not in node_number:
            node_number.append(node_info["tensor"])
        if (node_info["type"] == "fiberlookup" or node_info["type"] == "repeat") and node_info["root"] == "true":
            root_nodes.append(node_info["tensor"])
        if node_info["type"] == "fiberlookup":
            # print(u, " fiber lookup in :: ", networkx_graph.nodes[u]['comment'])
            if node_info["format"] == "dense":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] +
                        node_info["index"] + "_" + str(u) + " = UncompressRdScan(dim=" + node_info["tensor"] +
                        "_shape[" + node_info["mode"] + "]" + ", debug=debug_sim)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u)

            if node_info["format"] == "compressed":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] +
                        "_" + str(u) + " = CompressedRdScan(crd_arr=" + node_info["tensor"] +
                        "_crd" + node_info["mode"] + ", seg_arr=" + node_info["tensor"] + "_seg" + node_info["mode"] + ", debug=debug_sim)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u)

        elif node_info["type"] == "arrayvals":
            print(u, " arrayvals in ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + "_" + str(u) + " = Array(init_arr=" +
                    node_info["tensor"] + "_vals, " + "debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + "_" + str(u)

        elif "broadcast" in networkx_graph.nodes[u]['comment']:
            print(u, "broadcast in :: ", networkx_graph.nodes[u]['comment'])

        elif node_info["type"] == "repsiggen":
            print(u, " repeatsiggen in :: ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) + node_info["type"] + "_" + node_info["index"] + "_" + str(u) + " = RepeatSigGen(debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + "_" + node_info["index"] + "_" + str(u)

        elif node_info["type"] == "repeat":
            print(u, " repeat in :: ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u) +
                    " = Repeat(debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["index"] + "_" + str(u)
        elif node_info["type"] == "intersect":
            f.write(tab(1) + node_info["type"] + node_info["index"] + "_" + str(u) + " = Intersect2(debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + node_info["index"] + "_" + str(u)
        elif node_info["type"] == "crddrop":
            f.write(tab(1) + node_info["type"] + "_" + str(u) + " = CrdDrop(debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "mul":
            f.write(tab(1) + node_info["type"] + "_" + str(u) + " = Multiply2(debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "reduce":
            f.write(tab(1) + node_info["type"] + "_" + str(u) + " = Reduce(debug=debug_sim)\n")
            d[u]["object"] = node_info["type"] + "_" + str(u)
        elif node_info["type"] == "fiberwrite":
            print(node_info)
            if node_info["mode"] == "vals":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) +
                        " = ValsWrScan(size=" + array_size_computation(node_info["size"]) +
                        ", fill=fill, debug=debug_sim)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u)

            elif node_info["format"] == "compressed":
                f.write(tab(1) + node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u) +
                        " = CompressWrScan(seg_size=" + array_size_computation(node_info["segsize"]) + ", size=" +
                        array_size_computation(node_info["crdsize"]) + ", fill=fill, debug=debug_sim)\n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u)
            else:
                print("uncompressed_node write" + apath + "  \n")
                d[u]["object"] = node_info["type"] + "_" + node_info["tensor"] + node_info["mode"] + "_" + str(u)
                continue
            if node_info["sink"] == "true":
                output_nodes[d[u]["object"]] = node_info["mode"]
        else:
            invalid_flag = 1
            print("Error invalid node detected", node_info["type"], "\n")
    if invalid_flag == 1:
        num += 1
        continue
    output_check_nodes(f, root_nodes)
# nx.topological_sort(networkx_graph)
    f.write("\n")
    f.write(tab(1) + "while not done and time < TIMEOUT:\n")
    stream_join_elements = {}
    ready_dataset = {}
    edge_data = {}
    mul_dataset = {}
    intersect_dataset = defaultdict(dict)
    done_all = {}
    for u, v, a in networkx_graph.edges(data=True):
        if v not in done_all:
            done_all[v] = 0
        if v not in stream_join_elements:
            stream_join_elements[v] = [u]
            ready_dataset[v] = [0]
            edge_data[v] = [str((a["label"]).strip('"'))]
        else:
            if u not in stream_join_elements[v]:
                stream_join_elements[v].append(u)
                ready_dataset[v].append(0)
                edge_data[v].append(str((a["label"]).strip('"')))
    # for u, v, a in networkx_graph.edges():
    #    if u in ready_dataset:
    #        ready_dataset[v][stream_join_elements[v].index(d[u]["object"])] = 1
    #        if ready_dataset[v].sum() == len(ready_dataset[v]):
    #            stack.push([u, v])
    #        else:
    #            print("")
    for u in networkx_graph.nodes():
        if d[u]["type"] == "fiberlookup" and u not in done_all:
            if d[u]["root"] == "true":
                f.write(tab(2) + "if len(in_ref_" + d[u]["tensor"] + ") > 0:\n")
                f.write(tab(3) + d[u]["object"] + ".set_in_ref(in_ref_" + d[u]["tensor"] + ".pop(0))\n")
                f.write(tab(2) + d[u]["object"] + ".update()\n\n")
                done_all[u] = 1

    for i in range(3):
        for u, v, a in list(nx.edge_bfs(networkx_graph)):  # .edges(data=True), networkx_graph.nodes())):
            a = networkx_graph.get_edge_data(u, v)[0]
            ready_dataset[v][stream_join_elements[v].index(u)] = done_all[u]
            if d[v]["type"] == "fiberlookup" and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        if "intersect" in d[u_]["object"]:
                            f.write(tab(2) + d[v]["object"] + ".set_in_ref(" +
                                    d[u_]["object"] + ".out_ref" + str(intersect_dataset[d[u_]["object"]][d[v]["tensor"]]) + "())\n")
                        else:
                            f.write(tab(2) + d[v]["object"] + ".set_in_ref(" + d[u_]["object"] + ".out_" +
                                    str(edge_data[v][stream_join_elements[v].index(u_)]) + "())\n")
                        # f.write(tab(3) + d[v]["object"] + ".set_in_ref(0)\n")
                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1
            if d[v]["type"] == "repsiggen" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        f.write(tab(2) + d[v]["object"] + ".set_istream(" + str(d[u_]["object"]).strip('"') +
                                ".out_" + str(edge_data[v][stream_join_elements[v].index(u_)]) + "())\n")
                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1
            if d[v]["type"] == "repeat" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    if d[v]["root"] == "true":
                        f.write(tab(2) + "if len(in_ref_" + d[v]["tensor"] + ") > 0:\n")
                        f.write(tab(3) + d[v]["object"] + ".set_in_ref(in_ref_" + d[v]["tensor"] + ".pop(0))\n")
                    for u_ in stream_join_elements[v]:
                        f.write(tab(2) + d[v]["object"] + ".set_in_" + str(edge_data[v][stream_join_elements[v].index(u_)]) +
                                "(" + d[u_]["object"] + ".out_" + str(edge_data[v][stream_join_elements[v].index(u_)]) + "())\n")
                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1

            if d[v]["type"] == "arrayvals" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        if "intersect" in d[u_]["object"]:
                            f.write(tab(2) + d[v]["object"] + ".set_load(" + d[u_]["object"] + ".out_ref" +
                                    str(intersect_dataset[d[u_]["object"]][d[v]["tensor"]]) + "())\n")
                        else:
                            f.write(tab(2) + d[v]["object"] + ".set_load(" + d[u_]["object"] + ".out_ref" + "())\n")
                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1

            if d[v]["type"] == "intersect" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        f.write(tab(2) + d[v]["object"] + ".set_in" + str(stream_join_elements[v].index(u_) + 1) + "(" +
                                d[u_]["object"] + ".out_ref(), " + d[u_]["object"] + ".out_crd())\n")
                        intersect_dataset[d[v]["object"]][d[u_]["tensor"]] = stream_join_elements[v].index(u_) + 1
                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1

            if d[v]["type"] == "crddrop" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        index_value = edge_data[v][stream_join_elements[v].index(u_)][-1]
                        print(d[v])
                        if index_value == d[v]["inner"]:
                            f.write(tab(2) + d[v]["object"] + ".set_inner_crd" + "(" + d[u_]["object"] + ".out_crd())\n")
                        if index_value == d[v]["outer"]:
                            f.write(tab(2) + d[v]["object"] + ".set_outer_crd" + "(" + d[u_]["object"] + ".out_crd())\n")
                    done_all[v] = 1
#            if d[v]["type"] == "intersect" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
#                if sum(ready_dataset[v]) == len(ready_dataset[v]):
#                    for u_ in stream_join_elements[v]:
#                        f.write(tab(2) + d[v]["object"] + ".set_in" + str(stream_join_elements[v].index(u_)+ 1) + "(" +
#                                d[u_]["object"] + ".out_ref(), " + d[u_]["object"]+ ".out_crd())\n")
#                        intersect_dataset[d[v]["object"]][d[u_]["tensor"]] = stream_join_elements[v].index(u_) + 1
#                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
#                    done_all[v] = 1

            # if d[v]["object"] not in intersect_dataset:
            #    intersect_dataset[d[v]["object"]] = [d[u]["object"]]
            #    f.write(tab(2) + d[v]["object"] + ".set_in" + "1" + "(" + d[u]["object"] + ".out_ref(), " +
            #            d[u]["object"]+ ".out_crd()))\n")
            # else:
            #    if d[u]["object"] not in intersect_dataset[d[v]["object"]]:
            #        intersect_dataset[d[v]["object"]].append(d[u]["object"])
            #        f.write(tab(2) + d[v]["object"] + ".set_in" + str(len(intersect_dataset[d[v]["object"]])) +
            #                "(" + d[u]["object"] + ".out_ref(), " +  d[u]["object"]+ ".out_crd()))\n\n")
            #        f.write(tab(2) +  d[v]["object"]  +  ".update()\n\n")

            if d[v]["type"] == "mul" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        f.write(tab(2) + d[v]["object"] + ".set_in" + str(stream_join_elements[v].index(u_) + 1) + "(" +
                                d[u_]["object"] + ".out_load())\n")
                        # f.write(tab(2) + d[v]["object"] + ".set_in" + stream_join_elements[v].index(u_) +
                        # "(" + d[u_]["object"] + ".out_ref(), " +  d[u_]["object"]+ ".out_crd()))\n")

                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1
            # if d[v]["object"] not in mul_dataset:
            #    mul_dataset[d[v]["object"]] = [d[u]["object"]]
            # else:
            #    mul_dataset[d[v]["object"]].append(d[u]["object"])
            #    #if len(d[v]["object"]) == 2:
            #    for i in range(mul_dataset(d[v]["object"])):
            #        f.write(tab(2) + d[v]["object"] + ".set_in" + str(i) + "(" +
            #                mul_dataset[d[v]["object"]][i] + ".out_load())\n")
            #    f.write(tab(2) +  d[v]["object"] + ".update()\n\n")

            if d[v]["type"] == "reduce" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        f.write(tab(2) + d[v]["object"] + ".set_in_" +
                                str(edge_data[v][stream_join_elements[v].index(u_)]) + "(" + d[u_]["object"] +
                                ".out_" + str(edge_data[v][stream_join_elements[v].index(u_)]) + "())\n")
                    f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                    done_all[v] = 1

            if d[v]["type"] == "fiberwrite" and parents_done(networkx_graph, done_all, v) and done_all[v] == 0:
                if sum(ready_dataset[v]) == len(ready_dataset[v]):
                    for u_ in stream_join_elements[v]:
                        if "inner-" in edge_data[v][stream_join_elements[v].index(u_)] or "outer-" in edge_data[v][stream_join_elements[v].index(u_)]:
                            edge_data[v][stream_join_elements[v].index(u_)] = edge_data[v][stream_join_elements[v].index(u_)][:-2]
                        if d[v]["mode"] == "vals":
                            f.write(tab(2) + d[v]["object"] + ".set_input(" + d[u_]["object"] + ".out_" +
                                    str(edge_data[v][stream_join_elements[v].index(u_)]) + "())\n")
                            f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                            done_all[v] = 1
                        else:
                            f.write(tab(2) + d[v]["object"] + ".set_input(" + d[u]["object"] + ".out_" +
                                    str(edge_data[v][stream_join_elements[v].index(u_)]) + "())\n")
                            f.write(tab(2) + d[v]["object"] + ".update()\n\n")
                            done_all[v] = 1

    # f.write(tab(1) + "\n\n")
    finish_outputs(f, output_nodes)
    for u in networkx_graph.nodes():
        if "fiberlookup" not in d[u]["object"] and "fiberwrite" not in d[u]["object"]:
            f.write(tab(1) + d[u]["object"] + ".print_fifos()\n")
    for u in networkx_graph.nodes():
        if "intersect" in d[u]["object"]:
            f.write(tab(1) + d[u]["object"] + ".return_intersection_rate()\n")
    f.close()
    if "matmul_ijk" in out_name[num] or "mat_elemmul" in out_name[num] or "mat_identity" in out_name[num]:
        os.system("cp " + out_name[num] + ".py ./sam/sim/test/apps/test_" + out_name[num] + ".py")
        os.system("rm " + out_name[num] + ".py")
    else:
        os.system("rm " + out_name[num] + ".py")
    num += 1
