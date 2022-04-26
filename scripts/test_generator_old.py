import pydot
import os
import networkx as nx
directory = '../compiler/sam-outputs/dot'


def generate_header(f, out_name):
    f.write("from sim.src.rd_scanner import UncompressRdScan, COmpressedRdScan")
    f.write("from sim.src.wr_scanner import ValsWrScan")
    f.write("from sim.src.joiner import Intersect2")
    f.write("from sim.src.compute import Multiply2")
    f.write("from sim.src.crd_manager import CrdDrop")
    f.write("from sim.src.base import remove_emptystr")
    f.write("from sim.test.test import *")
    f.write("def test_"+ out_name+ "():")





# iterate over files in
# that directory
file_paths = []
out_names = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    out_names.append(filename.strip(".gv"))
    # checking if it is a file
    if os.path.isfile(f):
        file_paths.append(f)
print(file_paths)
        
i =0
for apath in file_paths:
    print(apath, " ", out_names[i])
    graphs = pydot.graph_from_dot_file(apath)
    graph = graphs[0]
    networkx_graph = nx.nx_pydot.from_pydot(graph)

    #for u in networkx_graph.nodes(data=True):
       # print(u)
    print("--------------------------------")
    f = open(out_names[i] + ".txt", "w")
    generate_header(f, out_names[i])
    for u in list(nx.topological_sort(networkx_graph)):
        if "fiber_lookup" in networkx_graph.nodes[u]['comment']:
            print(u, " fiber lookup in :: ", networkx_graph.nodes[u]['comment'])
        if "intersect" in networkx_graph.nodes[u]['comment']:
            print(u, " intersect :: ", networkx_graph.nodes[u]['comment'])
        if "broadcast" in networkx_graph.nodes[u]['comment']:
            print(u, "broadscast in :: ", networkx_graph.nodes[u]['comment'])
        if "repeat" in networkx_graph.nodes[u]['comment']:
            print(u, " repeat in :: ", networkx_graph.nodes[u]['comment'])
        if "mul" in networkx_graph.nodes[u]['comment']:
            print(u, " mul in :: ", networkx_graph.nodes[u]['comment'])
        if "spaccumulator" in networkx_graph.nodes[u]['comment']:
            print(u, " fiber spaccumulator in :: ", networkx_graph.nodes[u]['comment'])
        if "fiber_write" in networkx_graph.nodes[u]['comment']:
            print(u, " fiber write in :: ", networkx_graph.nodes[u]['comment'])

    f.close()
    i += 1
    assert(0)
