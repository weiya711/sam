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

    def __init__(self, name='B', shape=None, sparsity=0.6, format='CSF', dump_dir=None, tensor=None, value_cap=None) -> None:

        # assert dimension is not None
        # self.dimension = dimension
        self.shape = shape
        self.array = None
        self.sparsity = sparsity
        self.format = format
        self.name = name
        if value_cap is None:
            self.value_cap = int(math.pow(2, 8)) - 1
        else:
            self.value_cap = value_cap

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
            self._create_matrix(value_cap=self.value_cap)
        self._create_fiber_tree()

    def _create_matrix(self, value_cap=int(math.pow(2, 8)) - 1):
        '''
        Routine to create the actual matrix from the dimension/shape
        '''
        self.array = numpy.random.randint(low=0, high=value_cap, size=self.shape)
        for idx, x in numpy.ndenumerate(self.array):
            if random.random() < self.sparsity:
                self.array[idx] = 0

    def _create_fiber_tree(self):
        self.fiber_tree = FiberTree(tensor=self.array)

    def dump_outputs(self, format=None, tpose=False, dump_shape=True, glb_override=False, glb_dump_dir=None):
        '''
        Dump the matrix into many files depending on matrix format
        '''

        use_dir = self.dump_dir
        print_hex = False
        if glb_override:
            use_dir = glb_dump_dir
            print_hex = True

        print(f"Using dump directory - {use_dir}")

        all_zeros = not np.any(self.array)

        # Transpose it first if necessary
        if tpose is True:
            self.array = numpy.transpose(self.array)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

        if format is not None:
            self.format = format

        if self.format == 'CSF':
            # Handle the all zeros case...
            if all_zeros:
                fake_lines_seg = ["0000",
                                  "0000"]
                fake_lines_crd = ["0000"]
                # If it's a scalar/length 1 vec
                if len(self.shape) == 1 and self.shape[0] == 1:
                    fake_lines_val = ["0000"]
                else:
                    fake_lines_val = ["0000"]
                for mode in range(len(self.array.shape)):
                    if glb_override:
                        lines = [len(fake_lines_seg), *fake_lines_seg, len(fake_lines_crd), *fake_lines_crd]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_{mode}", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(fake_lines_seg, name=f"tensor_{self.name}_mode_{mode}_seg",
                                         dump_dir=use_dir, hex=print_hex)
                        self.write_array(fake_lines_crd, name=f"tensor_{self.name}_mode_{mode}_crd",
                                         dump_dir=use_dir, hex=print_hex)
                if glb_override:
                    lines = [len(fake_lines_val), *fake_lines_val]
                    self.write_array(fake_lines_val, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                else:
                    self.write_array(fake_lines_val, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)

                return

            # In CSF format, need to iteratively create seg/coord arrays
            tmp_lvl_list = []
            tmp_lvl_list.append(self.fiber_tree.get_root())

            seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
            if glb_override:
                lines = [len(seg_arr), *seg_arr, len(coord_arr), *coord_arr]
                self.write_array(lines, name=f"tensor_{self.name}_mode_0", dump_dir=use_dir, hex=print_hex)
            else:
                self.write_array(seg_arr, name=f"tensor_{self.name}_mode_0_seg", dump_dir=use_dir, hex=print_hex)
                self.write_array(coord_arr, name=f"tensor_{self.name}_mode_0_crd", dump_dir=use_dir, hex=print_hex)

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
                    if glb_override:
                        lines = [len(tmp_lvl_list), *tmp_lvl_list]
                        # self.write_array(tmp_lvl_list, name=f"tensor_{self.name}_mode_vals" dump_dir=use_dir)
                        self.write_array(lines, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(tmp_lvl_list, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                else:
                    seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
                    if glb_override:
                        lines = [len(seg_arr), *seg_arr, len(coord_arr), *coord_arr]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_{i}", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(seg_arr, name=f"tensor_{self.name}_mode_{i}_seg", dump_dir=use_dir, hex=print_hex)
                        self.write_array(coord_arr, name=f"tensor_{self.name}_mode_{i}_crd", dump_dir=use_dir, hex=print_hex)
                i = i + 1
        elif self.format == "UNC":
            flat_array = []
            for val in numpy.nditer(self.array):
                flat_array.append(val)
            if glb_override:
                lines = [len(flat_array), *flat_array]
                self.write_array(lines, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
            else:
                self.write_array(flat_array, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
        elif self.format == "COO":
            crd_dict = dict()
            order = len(self.array.shape)
            for i in range(order + 1):
                crd_dict[i] = []
            it = np.nditer(self.array, flags=['multi_index'])
            while not it.finished:
                crd_dict[order].append(it[0])
                point = it.multi_index
                for i in range(order):
                    crd_dict[i].append(point[i])
                is_not_finished = it.iternext()
            for key in crd_dict:
                if key == order:
                    if glb_override:
                        lines = [len(crd_dict[key]), *crd_dict[key]]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(crd_dict[key], name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                else:
                    if glb_override:
                        lines = [len(crd_dict[key]), *crd_dict[key]]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_{key}_crd", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(crd_dict[key],
                                         name=f"tensor_{self.name}_mode_{key}_crd",
                                         dump_dir=use_dir,
                                         hex=print_hex)

        if dump_shape:
            self.write_array(self.array.shape, name=f"tensor_{self.name}_mode_shape", dump_dir=use_dir, hex=print_hex)

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

    def write_array(self, str_list, name, dump_dir=None, hex=False):
        """Write an array/list to a file

        Args:
            list (list): array/list of values
            name (str): name of file
        """
        if dump_dir is None:
            dump_dir = self.dump_dir

        full_path = dump_dir + "/" + name
        with open(full_path, "w+") as wr_file:
            for item in str_list:
                item_int = int(item)
                if hex:
                    wr_file.write(f"{item_int:04X}\n")
                else:
                    wr_file.write(f"{item_int}\n")

    def get_shape(self):
        return self.shape

    def get_name(self):
        return self.name

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


def convert_aha_glb_output_file(glbfile, output_dir):

    glbfile_s = os.path.basename(glbfile).rstrip(".txt")

    if 'mode_vals' in glbfile:
        # num_blocks = 1
        files = [f"{output_dir}/{glbfile_s}"]
    else:
        # num_blocks = 2
        files = [f"{output_dir}/{glbfile_s}_seg",
                 f"{output_dir}/{glbfile_s}_crd"]

    straightline = []

    # Straighten the file out
    with open(glbfile, "r") as glbfile_h:
        file_contents = glbfile_h.readlines()
        for line in file_contents:
            sp_line = line.strip().split(" ")
            for sp_line_tok in sp_line:
                sp_line_tok_stripped = sp_line_tok.strip()
                straightline.append(int(sp_line_tok_stripped, base=16))

    # Now we have straightline having the items in order
    # Now write them to the output
    sl_ptr = 0
    for file_path in files:
        num_items = straightline[sl_ptr]
        sl_ptr += 1
        with open(file_path, "w+") as fh_:
            for _ in range(num_items):
                fh_.write(f"{straightline[sl_ptr]:04X}\n")
                sl_ptr += 1


def get_tensor_from_files(name, files_dir, shape, base=10, format='CSF', early_terminate=None) -> MatrixGenerator:
    all_files = os.listdir(files_dir)
    dims = len(shape)

    # Get vals first since all formats will have vals
    val_f = [fil for fil in all_files if name in fil and f'mode_vals' in fil][0]
    vals = read_inputs(f"{files_dir}/{val_f}", intype=int, base=base, early_terminate=early_terminate)

    mg = None
    if dims == 1 and shape[0] == 1:     # scalar
        mat_sc = numpy.zeros([1])
        mat_sc[0] = vals[0]
        mg = MatrixGenerator(name=name, shape=shape, tensor=mat_sc)
    elif format == 'CSF':
        created_empty = False
        segs = []
        crds = []
        for mode in range(dims):
            seg_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'seg' in fil][0]
            crd_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'crd' in fil][0]
            seg_t_ = read_inputs(f"{files_dir}/{seg_f}", intype=int, base=base, early_terminate=early_terminate)
            segs.append(seg_t_)
            # Empty matrix...
            if mode == 0 and len(seg_t_) == 2 and seg_t_[0] == 0 and seg_t_[1] == 0:
                mg = MatrixGenerator(name=name, shape=shape, sparsity=1.0)
                created_empty = True
                break
            crd_t_ = read_inputs(f"{files_dir}/{crd_f}", intype=int, base=base, early_terminate=early_terminate)
            crds.append(crd_t_)
        if not created_empty:
            pt_list = get_point_list(crds, segs, val_arr=vals)
            mg = create_matrix_from_point_list(name, pt_list, shape)
    elif format == 'COO':
        crds = []
        for mode in range(dims):
            crd_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'crd' in fil][0]
            crds.append(read_inputs(f"{files_dir}/{crd_f}", intype=int, base=base, early_terminate=early_terminate))

        pt_list = copy.deepcopy(crds)
        pt_list.append(vals)
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
