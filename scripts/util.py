import scipy.sparse
import scipy.io
import sparse
import os
import glob
import numpy
import itertools
import shutil
import numpy as np
import math
import functools

from pathlib import Path
from dataclasses import dataclass


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
    def __init__(self, cast_int=True):
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


# TnsFileDumper dumps a dictionary of coordinates to values
# into a coordinate list tensor file.
class TnsFileDumper:
    def __init__(self):
        pass

    def dump_dict_to_file(self, shape, data, path, write_shape=False):
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
            assert (False)


# PydataSparseTensorLoader loads a sparse tensor from a file into
# a pydata.sparse tensor.
# class PydataSparseTensorLoader:
#     def __init__(self):
#         self.loader = TnsFileLoader()
#
#     def load(self, path):
#         dims, coords, values = self.loader.load(path)
#         return sparse.COO(coords, values, tuple(dims))
#
# # PydataSparseTensorDumper dumps a sparse tensor to a the desired file.
# class PydataSparseTensorDumper:
#     def __init__(self):
#         self.dumper = TnsFileDumper()
#
#     def dump(self, tensor, path):
#         self.dumper.dump_dict_to_file(tensor.shape, sparse.DOK(tensor).data, path)
#
#
#
# # PydataTensorShifter shifts all elements in the last mode
# # of the input pydata/sparse tensor by one.
# class PydataTensorShifter:
#     def __init__(self):
#         pass
#
#     def shiftLastMode(self, tensor):
#         coords = tensor.coords
#         data = tensor.data
#         resultCoords = []
#         for j in range(len(tensor.shape)):
#             resultCoords.append([0] * len(data))
#         resultValues = [0] * len(data)
#         for i in range(len(data)):
#             for j in range(len(tensor.shape)):
#                 resultCoords[j][i] = coords[j][i]
#             # resultValues[i] = data[i]
#             # TODO (rohany): Temporarily use a constant as the value.
#             resultValues[i] = 2
#             # For order 2 tensors, always shift the last coordinate. Otherwise, shift only coordinates
#             # that have even last coordinates. This ensures that there is at least some overlap
#             # between the original tensor and its shifted counter part.
#             if len(tensor.shape) <= 2 or resultCoords[-1][i] % 2 == 0:
#                 resultCoords[-1][i] = (resultCoords[-1][i] + 1) % tensor.shape[-1]
#         return sparse.COO(resultCoords, resultValues, tensor.shape)

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


@dataclass
class DoublyCompressedMatrix:
    shape: (int)
    seg0: [int]
    crd0: [int]
    seg1: [int]
    crd1: [int]
    data: [float]


# ScipyMatrixMarketTensorLoader loads tensors in the matrix market format
# into scipy.sparse matrices.
class ScipyMatrixMarketTensorLoader:
    def __init__(self):
        pass

    def load(self, path):
        coo = scipy.io.mmread(path)
        return coo


def shape_str(shape):
    return str(shape[0]) + " " + str(shape[1])


# FIXME: This fixed point number of decimals may not be enough
def array_str(array):
    if isinstance(array[0], float):
        return ' '.join(['{:5.5f}'.format(item) for item in array])

    return ' '.join([str(item) for item in array])


def array_newline_str(array):
    if isinstance(array[0], float):
        return '\n'.join(['{:5.5f}'.format(item) for item in array])

    return '\n'.join([str(item) for item in array])


# InputCacheSuiteSparse attempts to avoid reading the same tensor from disk multiple
# times in a benchmark run.
class InputCacheSuiteSparse:
    def __init__(self):
        self.lastLoaded = None
        self.lastName = None
        self.tensor = None

    def load(self, tensor, cast):
        if self.lastName == str(tensor):
            return self.tensor
        else:
            self.lastLoaded = tensor.load(ScipyMatrixMarketTensorLoader())
            self.lastName = str(tensor)
            if cast:
                self.tensor = self.lastLoaded
                # self.tensor = safeCastPydataTensorToInts(self.lastLoaded)
            else:
                self.tensor = self.lastLoaded
            return self.tensor


class FormatWriter:
    def __init__(self, cast_int=True):
        self.cast = cast_int

    def convert_format(self, coo, format_str):
        if self.cast:
            cast_data = np.array([round_sparse(elem) for elem in coo.data])
            coo = scipy.sparse.coo_matrix((cast_data, (coo.row, coo.col)))

        if format_str == "csr":
            return scipy.sparse.csr_matrix(coo)
        elif format_str == "csc":
            return scipy.sparse.csc_matrix(coo)
        elif format_str == "coo" or format_str == "cooT":
            # This matrix will get transposed on writeout for cooT
            return coo
        else:
            if format_str == "dense":
                return coo.todense()
            elif format_str == "denseT":
                return coo.todense().getT()
            if format_str == "dcsr":
                csr = scipy.sparse.csr_matrix(coo)
                has_row = [rc > 0 for rc in csr.getnnz(1)]
                segend = sum(has_row)
                seg0 = [0, segend]
                crd0 = [i for i, r in enumerate(has_row) if r]
                seg1 = [0] + list(itertools.accumulate(map(int, csr.getnnz(1))))
                crd1 = csr.indices
                data = csr.data
                dcsr = DoublyCompressedMatrix(csr.shape, seg0, crd0, seg1, crd1, data)
                return dcsr
            elif format_str == "dcsc":
                csc = scipy.sparse.csc_matrix(coo)
                has_col = [rc > 0 for rc in csc.getnnz(0)]
                segend = sum(has_col)
                seg0 = [0, segend]
                crd0 = [i for i, c in enumerate(has_col) if c]
                seg1 = [0] + list(itertools.accumulate(map(int, csc.getnnz(0))))
                crd1 = csc.indices
                data = csc.data
                dcsc = DoublyCompressedMatrix(csc.shape, seg0, crd0, seg1, crd1, data)
                return dcsc
            else:
                assert (False)

    def writeout(self, coo, format_str, filename):
        tensor = self.convert_format(coo, format_str)

        with open(filename, "w") as outfile:
            if format_str == "dense":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 0\n")
                outfile.write(str(tensor.shape[0]) + '\n')
                outfile.write("mode 1\n")
                outfile.write(str(tensor.shape[1]) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.getA1()) + '\n')
            elif format_str == "denseT":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 1\n")
                outfile.write(str(tensor.shape[0]) + '\n')
                outfile.write("mode 0\n")
                outfile.write(str(tensor.shape[1]) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.getA1()) + '\n')
            elif format_str == "csr":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 0\n")
                outfile.write(str(tensor.shape[0]) + '\n')
                outfile.write("mode 1\n")
                outfile.write(array_str(tensor.indptr) + '\n')
                outfile.write(array_str(tensor.indices) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.data) + '\n')
            elif format_str == "csc":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 1\n")
                outfile.write(str(tensor.shape[1]) + '\n')
                outfile.write("mode 0\n")
                outfile.write(array_str(tensor.indptr) + '\n')
                outfile.write(array_str(tensor.indices) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.data) + '\n')
            elif format_str == "coo":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 0\n")
                outfile.write(array_str(tensor.row) + '\n')
                outfile.write("mode 1\n")
                outfile.write(array_str(tensor.col) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.data) + '\n')
            elif format_str == "cooT":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 1\n")
                outfile.write(array_str(tensor.col) + '\n')
                outfile.write("mode 0\n")
                outfile.write(array_str(tensor.row) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.data) + '\n')
            elif format_str == "dcsr":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 0\n")
                outfile.write(array_str(tensor.seg0) + '\n')
                outfile.write(array_str(tensor.crd0) + '\n')
                outfile.write("mode 1\n")
                outfile.write(array_str(tensor.seg1) + '\n')
                outfile.write(array_str(tensor.crd1) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.data) + '\n')
            elif format_str == "dcsc":
                outfile.write("shape\n")
                outfile.write(shape_str(tensor.shape) + '\n')
                outfile.write("mode 1\n")
                outfile.write(array_str(tensor.seg0) + '\n')
                outfile.write(array_str(tensor.crd0) + '\n')
                outfile.write("mode 0\n")
                outfile.write(array_str(tensor.seg1) + '\n')
                outfile.write(array_str(tensor.crd1) + '\n')
                outfile.write("vals\n")
                outfile.write(array_str(tensor.data) + '\n')
            else:
                assert (False)

            os.chmod(filename, 0o666)

    def writeout_separate(self, coo, dir_path, tensorname, omit_dense=True):

        csr_dir = Path(os.path.join(dir_path, "ds01"))
        if os.path.exists(csr_dir):
            shutil.rmtree(csr_dir)
        csr_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        dcsr_dir = Path(os.path.join(dir_path, "ss01"))
        if os.path.exists(dcsr_dir):
            shutil.rmtree(dcsr_dir)
        dcsr_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        csc_dir = Path(os.path.join(dir_path, "ds10"))
        if os.path.exists(csc_dir):
            shutil.rmtree(csc_dir)
        csc_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        dcsc_dir = Path(os.path.join(dir_path, "ss10"))
        if os.path.exists(dcsc_dir):
            shutil.rmtree(dcsc_dir)
        dcsc_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        dcsr = self.convert_format(coo, "dcsr")

        # Shape shared between all formats
        filename = os.path.join(dir_path, tensorname + "_shape.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsr.shape))
            os.chmod(filename, 0o666)

            os.symlink(filename, os.path.join(dcsr_dir, tensorname + "_shape.txt"))
            os.symlink(filename, os.path.join(csr_dir, tensorname + "_shape.txt"))
            os.symlink(filename, os.path.join(dcsc_dir, tensorname + "_shape.txt"))
            os.symlink(filename, os.path.join(csc_dir, tensorname + "_shape.txt"))

        filename = os.path.join(dcsr_dir, tensorname + "0_seg.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsr.seg0))
            os.chmod(filename, 0o666)

        filename = os.path.join(dcsr_dir, tensorname + "0_crd.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsr.crd0))
            os.chmod(filename, 0o666)

        filename = os.path.join(dcsr_dir, tensorname + "1_seg.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsr.seg1))
            os.chmod(filename, 0o666)

        filename = os.path.join(dir_path, tensorname + "1_crd_s1.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsr.crd1))
            os.chmod(filename, 0o666)
            os.symlink(filename, os.path.join(dcsr_dir, tensorname + "1_crd.txt"))
            os.symlink(filename, os.path.join(csr_dir, tensorname + "1_crd.txt"))

        filename = os.path.join(dir_path, tensorname + "_vals_s.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsr.data))
            os.chmod(filename, 0o666)
            os.symlink(filename, os.path.join(dcsr_dir, tensorname + "_vals.txt"))
            os.symlink(filename, os.path.join(csr_dir, tensorname + "_vals.txt"))

        csr = self.convert_format(coo, "csr")
        filename = os.path.join(csr_dir, tensorname + "1_seg.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(csr.indptr))
            os.chmod(filename, 0o666)

        dcsc = self.convert_format(coo, "dcsc")

        filename = os.path.join(dcsc_dir, tensorname + "1_seg.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsc.seg0))
            os.chmod(filename, 0o666)

        filename = os.path.join(dcsc_dir, tensorname + "1_crd.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsc.crd0))
            os.chmod(filename, 0o666)

        filename = os.path.join(dcsc_dir, tensorname + "0_seg.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsc.seg1))
            os.chmod(filename, 0o666)

        filename = os.path.join(dir_path, tensorname + "0_crd.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsc.crd1))
            os.chmod(filename, 0o666)
            os.symlink(filename, os.path.join(dcsc_dir, tensorname + "0_crd.txt"))
            os.symlink(filename, os.path.join(csc_dir, tensorname + "0_crd.txt"))

        filename = os.path.join(dir_path, tensorname + "_vals_sT.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(dcsc.data))
            os.chmod(filename, 0o666)
            os.symlink(filename, os.path.join(dcsc_dir, tensorname + "_vals.txt"))
            os.symlink(filename, os.path.join(csc_dir, tensorname + "_vals.txt"))

        csc = self.convert_format(coo, "csc")
        filename = os.path.join(csc_dir, tensorname + "0_seg.txt")
        with open(filename, "w") as ofile:
            ofile.write(array_newline_str(csc.indptr))
            os.chmod(filename, 0o666)

        if not omit_dense:
            dense = self.convert_format(coo, "dense")

            dense_dir = Path(os.path.join(dir_path, "dd01"))
            dense_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
            os.symlink(filename, os.path.join(dense_dir, "B_shape.txt"))

            filename = os.path.join(dense_dir, tensorname + "_vals.txt")
            with open(filename, "w") as ofile:
                ofile.write(array_newline_str(dense))
                os.chmod(filename, 0o666)

    
# UfuncInputCache attempts to avoid reading the same tensor from disk multiple
# times in a benchmark run.
class InputCacheTensor:
    def __init__(self):
        self.lastLoaded = None
        self.lastName = None
        self.tensor = None

    def load(self, tensor, suiteSparse, cast, format_str):
        if self.lastName == str(tensor):
            return self.tensor
        else:
            self.lastLoaded = tensor.load()
            self.lastName = str(tensor)
            if cast:
                self.tensor = safeCastPydataTensorToInts(self.lastLoaded)
            else:
                self.tensor = self.lastLoaded
            return self.tensor


# PydataMatrixMarketTensorLoader loads tensors in the matrix market format
# into pydata.sparse matrices.
# class PydataMatrixMarketTensorLoader:
#     def __init__(self):
#         pass
#
#     def load(self, path):
#         coo = scipy.io.mmread(path)
#         return sparse.COO.from_scipy_sparse(coo)

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
        SUITESPARSE_PATH = os.environ['SUITESPARSE_PATH']
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
        # if int(tensor.data[i]) == 0:
        #     data[i] = 1
        # else:
        #     data[i] = int(tensor.data[i])
        data[i] = round_special(tensor.data[i])
    return sparse.COO(tensor.coords, data, tensor.shape)


def parse_taco_format(infilename, outdir, tensorname, format_str):
    
    with open(infilename, 'r') as inf:
        level = -1
        count = 0
        seg = True
        level_done = False
        for line in inf:
            if count == 0:
                dim_start = line.find('(') + 1
                dim_end = line.find(')')
                dims = line[dim_start : dim_end]
                dims = dims.split('x')

                shapefile = os.path.join(outdir, tensorname + '_shape.txt')
                with open(shapefile, 'w+') as shapef:
                    shapef.write(array_newline_str(dims))
            else:
                if line.find(':') > -1:
                    level += 1
                    seg = True
                    level_done = False
                else:
                    start = line.find('[') + 1
                    end = line.find(']')
                    line = line[start: end]
                    line = line.split(', ')

                    if level_done:
                        # This is a values array
                        valfile = os.path.join(outdir, tensorname + '_val.txt')
                        with open(valfile, 'w+') as valf:
                            valf.write(array_newline_str(line))
                    else:
                        level_format = format_str[level]
                        if level_format == 's':
                            if seg:
                                segfile = os.path.join(outdir, tensorname + str(level)
                                                       + '_seg.txt') 
                                with open(segfile, 'w+') as segf:
                                    segf.write(array_newline_str(line))
                                seg = False
                            else:
                                crdfile = os.path.join(outdir, tensorname + str(level)
                                                       + '_crd.txt')
                                with open(crdfile, 'w+') as crdf:
                                    crdf.write(array_newline_str(line)) 
                                level_done = True
                    
            count += 1
    
