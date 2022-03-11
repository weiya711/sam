import scipy.sparse
import scipy.io
import sparse
import os
import glob
import numpy

SUITESPARSE_PATH = os.environ['SUITESPARSE_PATH']

# TnsFileLoader loads a tensor stored in .tns format.
class TnsFileLoader:
    def __init__(self):
        pass

    def load(self, path):
        coordinates = []
        values = []
        dims = []
        first = True
        with open(path, 'r') as f:
            for line in f:
                data = line.split(' ')
                if first:
                    first = False
                    dims = [0] * (len(data) - 1)
                    for i in range(len(data) - 1):
                        coordinates.append([])

                for i in range(len(data) - 1):
                    coordinates[i].append(int(data[i]) - 1)
                    dims[i] = max(dims[i], coordinates[i][-1] + 1)
                # TODO (rohany): What if we want this to be an integer?
                values.append(float(data[-1]))
        return dims, coordinates, values

# TnsFileDumper dumps a dictionary of coordinates to values
# into a coordinate list tensor file.
class TnsFileDumper:
    def __init__(self):
        pass

    def dump_dict_to_file(self, shape, data, path, write_shape = False):
        # Sort the data so that the output is deterministic.
        sorted_data = sorted([list(coords) + [value] for coords, value in data.items()])
        with open(path, 'w+') as f:
            for line in sorted_data:
                coords = [str(elem + 1) for elem in line[:len(line) - 1]]
                strings = coords + [str(line[-1])]
                f.write(" ".join(strings))
                f.write("\n")
            if write_shape:
                shape_strings = [str(elem) for elem in shape] + ['0']
                f.write(" ".join(shape_strings))
                f.write("\n")

# ScipySparseTensorLoader loads a sparse tensor from a file into a
# scipy.sparse CSR matrix.
class ScipySparseTensorLoader:
    def __init__(self, format):
        self.loader = TnsFileLoader()
        self.format = format

    def load(self, path):
        dims, coords, values = self.loader.load(path)
        if self.format == "csr":
            return scipy.sparse.csr_matrix((values, (coords[0], coords[1])), shape=tuple(dims))
        elif self.format == "csc":
            return scipy.sparse.csc_matrix((values, (coords[0], coords[1])), shape=tuple(dims))
        elif self.format == "coo":
            return scipy.sparse.coo_matrix(values, (coords[0], coords[1]), shape=tuple(dims))
        else:
            assert(False)

# PydataSparseTensorLoader loads a sparse tensor from a file into
# a pydata.sparse tensor.
class PydataSparseTensorLoader:
    def __init__(self):
        self.loader = TnsFileLoader()
    
    def load(self, path):
        dims, coords, values = self.loader.load(path)
        return sparse.COO(coords, values, tuple(dims))

# PydataSparseTensorDumper dumps a sparse tensor to a the desired file.
class PydataSparseTensorDumper:
    def __init__(self):
        self.dumper = TnsFileDumper()

    def dump(self, tensor, path):
        self.dumper.dump_dict_to_file(tensor.shape, sparse.DOK(tensor).data, path)



# PydataTensorShifter shifts all elements in the last mode
# of the input pydata/sparse tensor by one.
class PydataTensorShifter:
    def __init__(self):
        pass

    def shiftLastMode(self, tensor):
        coords = tensor.coords
        data = tensor.data
        resultCoords = []
        for j in range(len(tensor.shape)):
            resultCoords.append([0] * len(data))
        resultValues = [0] * len(data)
        for i in range(len(data)):
            for j in range(len(tensor.shape)):
                resultCoords[j][i] = coords[j][i]
            # resultValues[i] = data[i]
            # TODO (rohany): Temporarily use a constant as the value.
            resultValues[i] = 2
            # For order 2 tensors, always shift the last coordinate. Otherwise, shift only coordinates
            # that have even last coordinates. This ensures that there is at least some overlap
            # between the original tensor and its shifted counter part.
            if len(tensor.shape) <= 2 or resultCoords[-1][i] % 2 == 0:
                resultCoords[-1][i] = (resultCoords[-1][i] + 1) % tensor.shape[-1]
        return sparse.COO(resultCoords, resultValues, tensor.shape)

# ScipyTensorShifter shifts all elements in the last mode
# of the input scipy/sparse tensor by one.
class ScipyTensorShifter:
    def __init__(self, format):
        self.format = format

    def shiftLastMode(self, tensor):
        dok = scipy.sparse.dok_matrix(tensor)
        result = scipy.sparse.dok_matrix(tensor.shape)
        for coord, val in dok.items():
            newCoord = list(coord[:])
            newCoord[-1] = (newCoord[-1] + 1) % tensor.shape[-1]
            # result[tuple(newCoord)] = val
            # TODO (rohany): Temporarily use a constant as the value.
            result[tuple(newCoord)] = 2
        if self.format == "csr":
            return scipy.sparse.csr_matrix(result)
        elif self.format == "csc":
            return scipy.sparse.csc_matrix(result)
        else:
            assert(False)

# ScipyMatrixMarketTensorLoader loads tensors in the matrix market format
# into scipy.sparse matrices.
class ScipyMatrixMarketTensorLoader:
    def __init__(self, format):
        self.format = format 

    def load(self, path):
        coo = scipy.io.mmread(path)
        if self.format == "csr":
            return scipy.sparse.csr_matrix(coo)
        elif self.format == "csc":
            return scipy.sparse.csc_matrix(coo)
        elif self.format == "coo":
            return coo
        else:
            assert(False)

# PydataMatrixMarketTensorLoader loads tensors in the matrix market format
# into pydata.sparse matrices.
class PydataMatrixMarketTensorLoader:
    def __init__(self):
        pass

    def load(self, path):
        coo = scipy.io.mmread(path)
        return sparse.COO.from_scipy_sparse(coo)

# SuiteSparseTensor represents a tensor in the suitesparse collection.
class SuiteSparseTensor:
    def __init__(self, path):
        self.path = path
        self.__name__ = self.__str__()

    def __str__(self):
        f = os.path.split(self.path)[1]
        return f.replace(".mtx", "")

    def load(self, loader):
        return loader.load(self.path)

# TensorCollectionSuiteSparse represents the set of all downloaded
# SuiteSparse tensors.
class TensorCollectionSuiteSparse:
    def __init__(self):
        data = SUITESPARSE_PATH 
        sstensors = glob.glob(os.path.join(data, "*.mtx"))
        self.tensors = [SuiteSparseTensor(t) for t in sstensors]

    def getTensors(self):
        return self.tensors
    def getTensorNames(self):
        return [str(tensor) for tensor in self.getTensors()]
    def getTensorsAndNames(self):
        return [(str(tensor), tensor) for tensor in self.getTensors()]

# safeCastPydataTensorToInts casts a floating point tensor to integers
# in a way that preserves the sparsity pattern.
def safeCastPydataTensorToInts(tensor):
    data = numpy.zeros(len(tensor.data), dtype='int64')
    for i in range(len(data)):
        # If the cast would turn a value into 0, instead write a 1. This preserves
        # the sparsity pattern of the data.
        if int(tensor.data[i]) == 0:
            data[i] = 1
        else:
            data[i] = int(tensor.data[i])
    return sparse.COO(tensor.coords, data, tensor.shape)


