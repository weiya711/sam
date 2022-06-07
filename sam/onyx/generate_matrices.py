import numpy
import random
import scipy.sparse as ss
import tempfile
from fiber_tree import *


class MatrixGenerator():

    def __init__(self, name='B', shape=None, sparsity=0.6, format='CSF', dump_dir=None) -> None:

        # assert dimension is not None
        assert shape is not None
        # self.dimension = dimension
        self.shape = shape
        self.array = None
        self.sparsity = sparsity
        self.format = format

        self.fiber_tree = None

        if dump_dir is not None:
            self.dump_dir = None
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
            seg_arr = []
            coord_arr = []

    def get_matrix(self):
        return self.array

    def __str__(self):
        return str(self.array)


if __name__ == "__main__":

    mg = MatrixGenerator(name='B', shape=[10, 10, 10], dump_dir='/home/max/Documents/SPARSE/sam/OUTPUTS_DUMP')
    print(mg)
