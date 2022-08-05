from ast import dump
from operator import mod
from venv import create
import numpy
import random
import scipy.sparse as ss
import tempfile
from sam.onyx.fiber_tree import *
import argparse
import math
import csv
import os
from sam.sim.test.test import *


class MatrixGenerator():

    def __init__(self, name='B', shape=None, sparsity=0.6, format='CSF', dump_dir=None, tensor=None) -> None:

        # assert dimension is not None
        # self.dimension = dimension
        self.shape = shape
        self.array = None
        self.sparsity = sparsity
        self.format = format
        self.name = name

        self.fiber_tree = None

        if dump_dir is not None:
            self.dump_dir = dump_dir
            if not os.path.isdir(self.dump_dir):
                os.mkdir(self.dump_dir)
            else:
                # Otherwise clean it
                for filename in os.listdir(self.dump_dir):
                    ret = os.remove(self.dump_dir + "/" + filename)
        else:
            self.dump_dir = tempfile.gettempdir()

        if tensor is not None:
            self.array = tensor
            self.shape = self.array.shape
        else:
            assert shape is not None
            self._create_matrix()
        self._create_fiber_tree()

    def _create_matrix(self):
        '''
        Routine to create the actual matrix from the dimension/shape
        '''
        self.array = numpy.random.randint(low=0, high=int(math.pow(2, 8)) - 1, size=self.shape)
        for idx, x in numpy.ndenumerate(self.array):
            if random.random() < self.sparsity:
                self.array[idx] = 0

    def _create_fiber_tree(self):
        self.fiber_tree = FiberTree(tensor=self.array)

    def dump_outputs(self, format=None, tpose=False, dump_shape=True):
        '''
        Dump the matrix into many files depending on matrix format
        '''
        print(f"Using dump directory - {self.dump_dir}")

        # Transpose it first if necessary
        if tpose is True:
            self.array = numpy.transpose(self.array)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

        if format is not None:
            self.format = format

        if self.format == 'CSF':
            # In CSF format, need to iteratively create seg/coord arrays
            tmp_lvl_list = []
            tmp_lvl_list.append(self.fiber_tree.get_root())

            seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
            self.write_array(seg_arr, name=f"tensor_{self.name}_mode_0_seg")
            self.write_array(coord_arr, name=f"tensor_{self.name}_mode_0_crd")

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
                    self.write_array(tmp_lvl_list, name=f"tensor_{self.name}_mode_vals")
                else:
                    seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
                    self.write_array(seg_arr, name=f"tensor_{self.name}_mode_{i}_seg")
                    self.write_array(coord_arr, name=f"tensor_{self.name}_mode_{i}_crd")
                i = i + 1
        elif self.format == "UNC":
            flat_array = []
            for val in numpy.nditer(self.array):
                flat_array.append(val)
            self.write_array(flat_array, name=f"tensor_{self.name}_mode_vals")

        if dump_shape:
            self.write_array(self.array.shape, name=f"shape")

        # Transpose it back
        if tpose is True:
            self.array = numpy.transpose(self.array)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

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

    def get_compressed_arrays(self, format=None, tpose=False, dump_shape=True):
        '''
        Dump the matrix into many arrays in a dict depending on matrix format
        '''

        result_dict = {}
        # Transpose it first if necessary
        if tpose is True:
            self.array = numpy.transpose(self.array)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

        if format is not None:
            self.format = format

        if self.format == 'CSF':
            # In CSF format, need to iteratively create seg/coord arrays
            tmp_lvl_list = []
            tmp_lvl_list.append(self.fiber_tree.get_root())

            seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
            result_dict[f"tensor_{self.name}_mode_0_seg"] = seg_arr
            result_dict[f"tensor_{self.name}_mode_0_crd"] = coord_arr

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
                    result_dict[f"tensor_{self.name}_mode_vals"] = tmp_lvl_list
                else:
                    seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
                    result_dict[f"tensor_{self.name}_mode_{i}_seg"] = seg_arr
                    result_dict[f"tensor_{self.name}_mode_{i}_crd"] = coord_arr
                i = i + 1
        elif self.format == "UNC":
            flat_array = []
            for val in numpy.nditer(self.array):
                flat_array.append(val)
            result_dict[f"tensor_{self.name}_mode_vals"] = flat_array

            for i, d in enumerate(self.array.shape):
                result_dict[f"tensor_{self.name}_mode_{i}"] = d

        if dump_shape:
            result_dict[f"shape"] = self.array.shape

        # Transpose it back
        if tpose is True:
            self.array = numpy.transpose(self.array)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

        return result_dict

    def get_matrix(self):
        return self.array

    def __str__(self):
        return str(self.array)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, val):
        self.array[key] = val


def get_runs(v1, v2):
    """Get the average run length/runs of each vector

    Args:
        v1 (vector): vector 1
        v2 (vector): vector 2
    """

    # Ensure both are vectors of the same length
    assert len(v1.shape) == 1
    assert len(v2.shape) == 1
    assert v1.shape[0] == v2.shape[0]

    run_idx = [[], []]

    # First coiterate to get to first mismatch
    on_run = False
    run_start = None
    run_end = None
    run_side = None

    last_nonzero_v1 = None
    last_nonzero_v2 = None

    for idx, val in numpy.ndenumerate(v1):
        v2_val = v2[idx]
        idx = idx[0]
        if (val == 0 and v2_val != 0) or (val != 0 and v2_val == 0):
            # If run side is 0 and val is 0, we continue,
            if not on_run:
                # If not on run, determine the side and run_start, on_run
                if val != 0:
                    run_side = 0
                else:
                    run_side = 1
                run_start = idx
                on_run = True
            elif val != 0 and run_side == 0:
                # Continuing the run
                pass
            elif val != 0 and run_side == 1:
                # Here we are seeing the run end on the other side
                run_end = last_nonzero_v2
                run_idx[run_side].append((run_start, run_end))
                # Now we are starting a new run
                run_side = 0
                run_start = idx
            elif v2_val != 0 and run_side == 1:
                # Continuing the run
                pass
            elif v2_val != 0 and run_side == 0:
                # Here we are seeing the run end on the other side
                run_end = last_nonzero_v1
                run_idx[run_side].append((run_start, run_end))
                # Now we are starting a new run
                run_side = 1
                run_start = idx
        elif val == 0 and v2_val == 0:
            # If both are 0 it's fine
            pass
        else:
            # If both are 1 the run is over at the previous nonzero value, and we are no longer on a run (assuming we were)
            if on_run:
                if run_side == 0:
                    run_end = last_nonzero_v1
                else:
                    run_end = last_nonzero_v2
                run_idx[run_side].append((run_start, run_end))
                on_run = False

        if val != 0:
            last_nonzero_v1 = idx
        if v2_val != 0:
            last_nonzero_v2 = idx

    # Now off the end, if we have started a run on either side, we should terminate it
    if on_run:
        run_idx[run_side].append((run_start, v1.shape[0] - 1))

    return run_idx


def get_run_lengths(run_idx, v1, v2):

    run_len = [[], []]

    for side in range(2):
        tmp_vec = v1
        if side == 1:
            tmp_vec = v2
        for idx_tuple in run_idx[side]:
            tmp_run_len = 0
            idx1, idx2 = idx_tuple
            for idx in range(idx2 + 1 - idx1):
                if tmp_vec[idx + idx1] != 0:
                    tmp_run_len += 1
            run_len[side].append(tmp_run_len)

    return run_len


def delete_run(vec, run_list):
    """Delete random run from a vector

    Args:
        vec (_type_): _description_
        run_list (_type_): _description_
    """

    # Early out
    if len(run_list) == 0:
        return

    random_idx = random.randint(0, vec.size)
    run_to_del = mod(random_idx, len(run_list))

    idx1, idx2 = run_list[run_to_del]
    for i in range(idx2 + 1 - idx1):
        vec[i + idx1] = 0


def run_statistics(name, seed, shape, dump_dir, sparsity):

    random.seed(seed)
    numpy.random.seed(seed)
    vec1 = MatrixGenerator(name=name, shape=shape, dump_dir=dump_dir, sparsity=sparsity)
    vec2 = MatrixGenerator(name=f"{name}_alt", shape=shape, dump_dir=dump_dir, sparsity=sparsity)

    # Now delete runs from the first
    for i in range(5):
        run_idx = get_runs(vec1.get_matrix(), vec2.get_matrix())
        delete_run(vec1.get_matrix(), run_idx[0])

    run_list = get_run_lengths(run_idx, vec1.get_matrix(), vec2.get_matrix())

    if len(run_list[0]) > 0:
        avg1 = sum(run_list[0]) / len(run_list[0])
    else:
        avg1 = 0
    if len(run_list[1]) > 0:
        avg2 = sum(run_list[1]) / len(run_list[1])
    else:
        avg2 = 0

    return (avg1, avg2)


def create_matrix_from_point_list(name, pt_list, shape) -> MatrixGenerator:
    mat_base = numpy.zeros(shape)
    dims = len(shape)
    for pt_idx in range(len(pt_list[0])):
        pt_base = []
        for i in range(dims):
            pt_base.append(pt_list[i][pt_idx])
        mat_base[tuple(pt_base)] = pt_list[dims][pt_idx]

    mg = MatrixGenerator(name=f"{name}", shape=shape, sparsity=0.7, format='CSF', dump_dir=None, tensor=mat_base)
    return mg


def get_tensor_from_files(name, files_dir, shape, base=10, early_terminate=None) -> MatrixGenerator:
    all_files = os.listdir(files_dir)
    dims = len(shape)
    segs = []
    crds = []
    scalar = False
    if dims == 1 and shape[0] == 1:
        scalar = True
    vals = None
    if not scalar:
        for mode in range(dims):
            seg_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'seg' in fil][0]
            crd_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'crd' in fil][0]
            segs.append(read_inputs(f"{files_dir}/{seg_f}", intype=int, base=base, early_terminate=early_terminate))
            crds.append(read_inputs(f"{files_dir}/{crd_f}", intype=int, base=base, early_terminate=early_terminate))
    val_f = [fil for fil in all_files if name in fil and f'mode_vals' in fil][0]
    vals = read_inputs(f"{files_dir}/{val_f}", intype=int, base=base, early_terminate=early_terminate)

    if scalar:
        mat_sc = numpy.zeros([1])
        mat_sc[0] = vals[0]
        mg = MatrixGenerator(name=name, shape=shape, tensor=mat_sc)
    else:
        pt_list = get_point_list(crds, segs, val_arr=vals)
        mg = create_matrix_from_point_list(name, pt_list, shape)

    return mg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--dump_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default='B')
    parser.add_argument("--shape", type=int, nargs="*", default=[10])
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    sparsity = args.sparsity
    name = args.name
    dump_dir = args.dump_dir
    shape = args.shape
    csv_out = args.output_csv

    averages_list = []
    for override_seed in range(1000):
        avg1, avg2 = run_statistics(name, override_seed, shape, dump_dir, sparsity)
        averages_list.append([override_seed, avg1, avg2])

    # Write a csv out
    fields = ['seed', 'average_runs_1', 'average_runs_2']

    with open(csv_out, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(averages_list)

    quit()

    random.seed(seed)
    numpy.random.seed(seed)
    vec1 = MatrixGenerator(name=name, shape=shape, dump_dir=dump_dir, sparsity=sparsity)
    vec2 = MatrixGenerator(name=f"{name}_alt", shape=shape, dump_dir=dump_dir, sparsity=sparsity)

    run_list = get_runs(vec1.get_matrix(), vec2.get_matrix())
    # mg.dump_outputs()

    avg1 = sum(run_list[0]) / len(run_list[0])
    avg2 = sum(run_list[1]) / len(run_list[1])

    print(f"Average Run Length V1: {avg1}")
    print(f"Average Run Length V2: {avg2}")
