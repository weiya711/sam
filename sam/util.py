import scipy.sparse
import os
import math
import numpy

# All environment variables for SAM should live here or in make file
cwd = os.getcwd()
SAM_HOME = os.getenv('HOSTNAME', default=cwd)
HOSTNAME = os.getenv('HOSTNAME', default="local")
SUITESPARSE_PATH = os.getenv('SUITESPARSE_PATH', default=os.path.join(SAM_HOME, "data", "suitesparse"))
SUITESPARSE_FORMATTED_PATH = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(SAM_HOME, "data",
                                                                                          "suitesparse-formatted"))
FROSTT_PATH = os.getenv('FROSTT_PATH', default=os.path.join(SAM_HOME, "data", "frostt"))
VALIDATION_OUTPUT_PATH = os.getenv('VALIDATION_OUTPUT_PATH', default=os.path.join(SAM_HOME, "data", "gold"))


def safeCastScipyTensorToInts(tensor):
    data = numpy.zeros(len(tensor.data), dtype='int64')
    for i in range(len(data)):
        # If the cast would turn a value into 0, instead write a 1. This preserves
        # the sparsity pattern of the data.
        # if int(tensor.data[i]) == 0:
        #     data[i] = 1
        # else:
        #     data[i] = int(tensor.data[i])
        data[i] = round_sparse(tensor.data[i])
    return scipy.sparse.coo_matrix(tensor.coords, data, tensor.shape)


# ScipyTensorShifter shifts all elements in the last mode
# of the input scipy/sparse tensor by one.
class ScipyTensorShifter:
    def __init__(self):
        pass

    def shiftLastMode(self, tensor):
        dok = scipy.sparse.dok_matrix(tensor)
        result = scipy.sparse.dok_matrix(tensor.shape)
        for coord, val in dok.items():
            newCoord = list(coord[:])
            newCoord[-1] = (newCoord[-1] + 1) % tensor.shape[-1]
            # result[tuple(newCoord)] = val
            # TODO (rohany): Temporarily use a constant as the value.
            result[tuple(newCoord)] = 2
        return scipy.sparse.coo_matrix(result)


def round_sparse(x):
    if 0.0 <= x < 1:
        return 1
    elif 0.0 > x > -1:
        return -1
    elif x >= 0.0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)


# TnsFileLoader loads a tensor stored in .tns format.
class TnsFileLoader:
    def __init__(self, cast_int=False):
        self.cast = cast_int

    def load(self, path):
        coordinates = []
        values = []
        dims = []
        first = True
        with open(path, 'r') as f:
            for line in f:
                data = line[:-1].split(' ')
                if first:
                    first = False
                    dims = [0] * (len(data) - 1)
                    for i in range(len(data) - 1):
                        coordinates.append([])
                data = [elem for elem in data if elem != '']

                for i in range(len(data) - 1):
                    coordinates[i].append(int(data[i]) - 1)
                    dims[i] = max(dims[i], coordinates[i][-1] + 1)
                # TODO (rohany): What if we want this to be an integer?
                if self.cast:
                    val = round_sparse(float(data[-1]))
                    values.append(val)
                else:
                    values.append(float(data[-1]))
        return dims, coordinates, values
