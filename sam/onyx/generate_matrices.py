from ast import dump
import numpy
import random
import scipy.sparse as ss
import tempfile
from sam.onyx.fiber_tree import *
import argparse


class MatrixGenerator():

    def __init__(self, name='B', shape=None, sparsity=0.6, format='CSF', dump_dir=None) -> None:

        # assert dimension is not None
        assert shape is not None
        # self.dimension = dimension
        self.shape = shape
        self.array = None
        self.sparsity = sparsity
        self.format = format
        self.name = name

        self.fiber_tree = None

        if dump_dir is not None:
            self.dump_dir = dump_dir
        else:
            self.dump_dir = tempfile.gettempdir()
            print(f"Using temporary directory - {self.dump_dir}")

        self._create_matrix()
        self._create_fiber_tree()

    def _create_matrix(self):
        '''
        Routine to create the actual matrix from the dimension/shape
        '''
        self.array = numpy.random.randint(low=0, high=1000, size=self.shape)
        for idx, x in numpy.ndenumerate(self.array):
            if random.random() < self.sparsity:
                self.array[idx] = 0

    def _create_fiber_tree(self):
        self.fiber_tree = FiberTree(tensor=self.array)

    def dump_outputs(self):
        '''
        Dump the matrix into many files depending on matrix format
        '''
        dim = 0

        if self.format == 'CSF':
            # In CSF format, need to iteratively create seg/coord arrays
            tmp_lvl_list = []
            tmp_lvl_list.append(self.fiber_tree.get_root())

            seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
            self.write_array(seg_arr, name=f"tensor_{self.name}_mode_0_seg")
            self.write_array(coord_arr, name=f"tensor_{self.name}_mode_0_crd")
            print("SEG/CRD DEPTH 0")
            print(seg_arr)
            print(coord_arr)

            at_vals = False
            i = 1
            while at_vals is False:
                # Make the next level of fibers - basically BFS but segmented across depth of tree
                next_tmp_lvl_list = []
                for fib in tmp_lvl_list:
                    crd_payloads_tmp = fib.get_coord_payloads()
                    if type(crd_payloads_tmp[0][1]) is not FiberTreeFiber:
                        at_vals = True
                        for crd, pld in crd_payloads_tmp:
                            next_tmp_lvl_list.append(pld)
                    else:
                        for crd, pld in crd_payloads_tmp:
                            next_tmp_lvl_list.append(pld)
                tmp_lvl_list = next_tmp_lvl_list
                if at_vals:
                    # If at vals, we don't need to dump csf, we have the level
                    print(f"VALS DEPTH {i}")
                    print(tmp_lvl_list)
                    self.write_array(tmp_lvl_list, name=f"tensor_{self.name}_mode_vals")
                else:
                    seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
                    self.write_array(seg_arr, name=f"tensor_{self.name}_mode_{i}_seg")
                    self.write_array(coord_arr, name=f"tensor_{self.name}_mode_{i}_crd")
                    print(f"SEG/CRD DEPTH {i}")
                    print(seg_arr)
                    print(coord_arr)
                i = i + 1

    def _dump_csf(self, level_list):
        """ Dumps the csf-based seg/coord array for each level, unless it is a vals list

        Args:
            level_list (list): list of fibers at this level
        """

        seg_arr = []
        seg_running = 0
        coord_arr = []
        seg_arr.append(0)
        for level_fiber in level_list:
            # For each fiber in the level, the seg is just adding the running length, coord array is simple
            lf_crd_pld = level_fiber.get_coord_payloads()
            seg_running += len(lf_crd_pld)
            seg_arr.append(seg_running)
            for crd, pld in lf_crd_pld:
                coord_arr.append(crd)

        return seg_arr, coord_arr

    def write_array(self, str_list, name):
        """Write an array/list to a file

        Args:
            list (list): array/list of values
            name (str): name of file
        """
        full_path = self.dump_dir + "/" + name
        with open(full_path, "w+") as wr_file:
            for item in str_list:
                wr_file.write(f"{item}\n")

    def get_matrix(self):
        return self.array

    def __str__(self):
        return str(self.array)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--dump_dir", type=str, default='/Users/maxwellstrange/Documents/SPARSE/sam/tmp_dump')
    parser.add_argument("--name", type=str, default='B')
    args = parser.parse_args()

    seed = args.seed
    sparsity = args.sparsity
    name = args.name
    dump_dir = args.dump_dir

    random.seed(seed)
    numpy.random.seed(seed)
    mg = MatrixGenerator(name=name, shape=[10, 10], dump_dir=dump_dir, sparsity=sparsity)
    print(mg)
    mg.dump_outputs()
