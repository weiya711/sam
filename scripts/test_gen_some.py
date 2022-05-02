import pydot
import os
import networkx as nx
import matplotlib.pyplot as plt
directory = '../compiler/sam-outputs/dot'

def tab(a):
    ans = ""
    for i in range(a):
        ans += "\t"
    return ans


def generate_header(f, out_name):
    f.write("from sim.src.rd_scanner import UncompressRdScan, COmpressedRdScan\n")
    f.write("from sim.src.wr_scanner import ValsWrScan\n")
    f.write("from sim.src.joiner import Intersect2\n")
    f.write("from sim.src.compute import Multiply2\n")
    f.write("from sim.src.crd_manager import CrdDrop\n")
    f.write("from sim.src.base import remove_emptystr\n")
    f.write("from sim.test.test import *\n")
    f.write("def test_"+ out_name+ "():\n")

def size_computation_write(a):
    ans = ""
    for i in range(a-1):
        ans += "dim *"
    ans += "dim"
    return ans

def breakup_node_info(node_name):
    d  = dict(x.split("=") for x in node_name[1:-1].split(","))
    #for a, b in d.items():
    #   b.strip()
    return d


def remove_broadcast_nodes(G):
    g = G.copy()
    for a in g:
        g0 = g.copy()
        node_i = breakup_node_info(g.nodes[a]["comment"])
        if node_i["name"] == "broadcast":
            for preds in g0.predecessors(a):
                for succs in g0.neighbors(a):
                    print("----")
                    print( preds)
                    print( succs)
                    print(a)
                    print(g0.get_edge_data(preds, a))
                    g0.add_edge(preds, succs,**(g0.get_edge_data(preds, a)[0]))
                    print(g0.get_edge_data(preds, succs))
            g0.remove_node(a)
        g = g0
    return g



# iterate over files in
# that directory
file_paths = []
out_name = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    out_name.append(filename.strip(".gv"))

directory = '../compiler/sam-outputs/dot'
 
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
        
i =0
for apath in file_paths[0:1]:
    print(apath)
    apath = os.path.join(directory, "matmul_ijk.gv")
    
    graphs = pydot.graph_from_dot_file(apath)
    graph = graphs[0]
    networkx_graph = nx.nx_pydot.from_pydot(graph)

    #for u in list(nx.topological_sort(networkx_graph)):#    networkx_graph.nodes(data=True):
    #    print(str(u) + " " + networkx_graph.nodes[u]["comment"])
    #print("--------------------------------")
     
    f = open(out_name[i]+ ".py", "w")
    generate_header(f, out_name[i])
    
    nx.draw(networkx_graph, with_labels=True)
    plt.savefig("Graph.png", format="PNG")
    plt.show()
    #for u in networkx_graph.nodes():
    #    print(str(u) + " " + networkx_graph.nodes[u]["comment"])
    #for u,v,a in networkx_graph.edges(data=True):
    #    print(u, " ", v, " ", a)



    networkx_graph = remove_broadcast_nodes(networkx_graph)
    for u in list(nx.topological_sort(networkx_graph)):#    networkx_graph.nodes(data=True):
        print(str(u) + " " + networkx_graph.nodes[u]["comment"])
 
    for u,v,a in networkx_graph.edges(data=True):
        print(u, " ", v, " ", a)


    #for u in list(nx.topological_sort(networkx_graph)):#    networkx_graph.nodes(data=True):
    #    print(u)
 



    nx.draw(networkx_graph, with_labels=True)
    plt.savefig("Graph_pruned.png", format="PNG")
    plt.show()

    d = {}


    for u in list(nx.topological_sort(networkx_graph)):
        node_info = breakup_node_info(networkx_graph.nodes[u]["comment"])
        d[u] = node_info
        if node_info["name"] == "fiberlookup":
            print(u, " fiber lookup in :: ", networkx_graph.nodes[u]['comment'])
            if node_info["format"] == "dense":
                f.write(tab(1) + node_info["name"] + "_" + node_info["tensor"] + node_info["index"]  +" = UncompressRdScan( dim = " + node_info["mode"] + ", debug = debug_sim) \n")
                d[u]["object"] =  node_info["name"] + "_" + node_info["tensor"] + node_info["index"] ;

            if node_info["format"] == "compressed":
                f.write(tab(1) +  node_info["name"] + "_" + node_info["tensor"] + node_info["index"] +" = CompressedRdScan(crd_arr=in_mat_crds1[" + node_info["mode"] + "], seg_arr=in_mat_segs1[" +node_info["mode"] + "], debug=debug_sim)\n")
                d[u]["object"] =  node_info["name"] + "_" + node_info["tensor"] + node_info["index"]

        if node_info["name"] == "arrayvals":
            print(u, " arrayvals in ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) +  node_info["name"] + "_"+ node_info["tensor"] +  " = Array(init_arr=in_mat_vals2, debug = debug_sim)\n")
            d[u]["object"] =  node_info["name"] + "_" + node_info["tensor"]

        if "broadcast" in networkx_graph.nodes[u]['comment']:
            print(u, "broadcast in :: ", networkx_graph.nodes[u]['comment'])



        if node_info["name"] == "repsiggen":
            print(u, " repeatsiggen in :: ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) +  node_info["name"] + "_" + node_info["index"] + " = RepeatSigGen(debug=debug_sim)\n")
            d[u]["object"] = node_info["name"] + "_" + node_info["index"];

        if node_info["name"] == "repeat":
            print(u, " repeat in :: ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) +  node_info["name"] + "_" + node_info["tensor"] + node_info["index"] + " = Repeat(debu=debug_sim)\n")
            d[u]["object"] =  node_info["name"] + "_" + node_info["tensor"] + node_info["index"];

        if node_info["name"] == "intersect":
            print(u, " repeat in :: ", networkx_graph.nodes[u]['comment'])
            f.write(tab(1) +  node_info["name"]  + " = Intersect2(debug = debug_sim)\n")
            d[u]["object"] =  node_info["name"]



        if node_info["name"] == "mul":
            f.write(tab(1) +  node_info["name"]  + " = Multiply2(debug=debug_sim)\n")
            d[u]["object"] =  node_info["name"]

        if node_info["name"] == "reduce":
            f.write(tab(1) + node_info["name"] +" = Reduce(debug=debug_sum)\n")
            d[u]["object"] =  node_info["name"]

        if node_info["name"] == "fiberwrite":
            if node_info["mode"] == "vals":
                f.write(tab(1) +  node_info["name"] + "_" + node_info["tensor"]  +  node_info["mode"] + " = ValsWrScan(size=dim*dim, fill=fill, debug=debug_sim)\n")
                d[u]["object"] =  node_info["name"] + "_" + node_info["tensor"]  +  node_info["mode"]
            else:
                f.write(tab(1) +  node_info["name"] + "_" + node_info["tensor"] + node_info["mode"] + " = CompressWrScan(seg_size = " + size_computation_write(int(node_info["mode"]) + 1) +  ", size=dim, fill = fill)\n")

                d[u]["object"] =  node_info["name"] + "_" + node_info["tensor"]  +  node_info["mode"]

# nx.topological_sort(networkx_graph)
    f.write("\n\n")
    f.write(tab(1) + "while not done and time < TIMEOUT:\n")
    stream_join_elements = {}
    ready_dataset = {}
    edge_data = {}
    for u, v, a in networkx_graph.edges(data = True):
        if v not in stream_join_elements:
            stream_join_elements[v] = [u]
            ready_dataset[v] = [0]
            edge_dataset = str(a["label"]).strip('"')
        else:
            stream_join_elements[v].append(u)
            ready_dataset[v].append(0)
            edge_dataset = str(a["label"]).strip('"')
    intersect_dataset = {}
    mul_dataset = {}

    for u,v,a in list(nx.edge_bfs(networkx_graph)): #.edges(data=True), networkx_graph.nodes())):

        print(u, " ", v, " ", a , " ", d[v]["object"], " ", d[u]["object"])
        a = networkx_graph.get_edge_data(u, v)[0] 
        #print(a)
        #ready_dataset[v][stream_join_elements[v].index(d[u]["object"])] = 1

        if d[v]["name"] == "fiberlookup":            
            f.write(tab(2) + d[v]["object"] + ".set_in_ref(" + u + ".in_ref_" + d[u]["tensor"] +".pop())\n")
            f.write(tab(2) + d[v]["object"] + ".update()\n\n")

        if d[v]["name"] == "repsiggen":
            f.write(tab(2) + d[v]["object"] + ".set_istream(" + str(d[u]["object"]).strip('"') + ".out_" + str(a["label"]).strip('"')+")\n")
            f.write(tab(2) + d[v]["object"] + ".update()\n\n")

        if d[v]["name"] == "repeat":
            f.write(tab(2) + d[v]["object"] + ".set_in_" + str(a["label"]).strip('"') + "(" + d[u]["object"]+ ".out_repeat())\n\n")
            f.write(tab(2) + d[v]["object"] + ".update()\n\n")

        if d[v]["name"] == "arrayvals":
            f.write(tab(2) + d[v]["object"] + ".set_load(" + d[u]["object"]+ ".out_ref())\n")
            f.write(tab(2) + d[v]["object"] + ".update()\n\n")
            
        if d[v]["name"] == "intersect":
            if d[v]["object"] not in intersect_dataset:
                intersect_dataset[d[v]["object"]] = [d[u]["object"]]  
                f.write(tab(2) + d[v]["object"] + ".set_in" + "1" + "(" + d[u]["object"] + ".out_ref(), " +  d[u]["object"]+ ".out_crd()))\n")
            else:
                if d[u]["object"] not in intersect_dataset[d[v]["object"]]:
                    intersect_dataset[d[v]["object"]].append(d[u]["object"])
                    f.write(tab(2) + d[v]["object"] + ".set_in" + str(len(intersect_dataset[d[v]["object"]])) + "(" + d[u]["object"] + ".out_ref(), " +  d[u]["object"]+ ".out_crd()))\n\n")
                    f.write(tab(2) +  d[v]["object"]  +  ".update()\n\n")

        if d[v]["name"] == "mul":
            if d[v]["object"] not in mul_dataset:
                mul_dataset[d[v]["object"]] = [d[u]["object"]]  
            else:
                mul_dataset[d[v]["object"]].append(d[u]["object"])
                #if len(d[v]["object"]) == 2:
                for i in range(len(mul_dataset[d[v]["object"]])):
                    f.write(tab(2) + d[v]["object"] + ".set_in" + str(i) + "(" + mul_dataset[d[v]["object"]][i]+ ".out_load())\n")
                f.write(tab(2) +  d[v]["object"]  +  ".update()\n\n")


        if d[v]["name"] == "reduce":
            f.write(tab(2) + d[v]["object"] + ".set_in_"+ str(a["label"]).strip('"')  +"(" + d[u]["object"] + ".out_" + str(a["label"]).strip('"') + "())\n")
            f.write(tab(2) +  d[v]["object"]  +  ".update()\n\n")


        if d[v]["name"] == "fiberwrite":
            if d[v]["mode"] == "vals":
                f.write("write  ------ \n")
            else:
                print(u, " ", v)
                f.write(tab(2) + d[v]["object"] + ".set_input(" + d[u]["object"] + "out_" + str(a["label"]).strip('"') + "())\n")
                f.write(tab(2) + d[v]["object"] + ".update()\n\n")



    f.close()
    assert(0)
    i += 1

