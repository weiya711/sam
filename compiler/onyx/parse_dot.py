import pydot

class SAMDotGraph():

    def __init__(self, filename=None) -> None:
        assert filename is not None, "filename is None"
        self.graphs = pydot.graph_from_dot_file(filename)
        self.graph = self.graphs[0]
        print(self.graph)
        nodes = self.graph.get_nodes()
        print(nodes)
        for node in nodes:
            attrs = node.get_attributes()

            print(node.get_name())
            print(node.get_comment())

if __name__ == "__main__":
    matmul_dot = "/home/max/Documents/SPARSE/sam/compiler/sam-outputs/dot/" + "matmul_ijk.gv"
    SAMDotGraph(filename=matmul_dot)