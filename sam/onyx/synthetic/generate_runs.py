import numpy
from sam.onyx.generate_matrices import *
import argparse
import random


def generate_runs_vectors(run1, run2):

    vec_size = 1000

    vec1 = numpy.zeros(vec_size)
    vec2 = numpy.zeros(vec_size)

    # First create one item in first 5% of the second vector
    # inject_idx = random.randint(0, int(0.05 * vec_size))
    inject_idx = 0
    vec2[inject_idx] = random.randint(0, 1000)

    inject_idx = inject_idx + 1

    for run_idx in range(run1):
        vec1[inject_idx + run_idx] = random.randint(0, 1000)

    # Now create a run on the other side
    inject_idx = inject_idx + run_idx + 1

    for run_idx in range(run2):
        vec2[inject_idx + run_idx] = random.randint(0, 1000)

    inject_idx = inject_idx + run_idx + 1

    vec1[inject_idx] = random.randint(0, 1000)

    return vec1, vec2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices - random')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number", type=int, default=1)
    # parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_lengths", type=int, nargs=2, default=[10, 10])
    # parser.add_argument("--shape", type=int, nargs="*", default=[10, 10])
    # parser.add_argument("--num_trials", type=int, default=1000)
    # parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    number = args.number
    # sparsity = args.sparsity
    output_dir = args.output_dir
    # name = args.name
    # shape = args.shape
    run_lengths = args.run_lengths

    random.seed(seed)
    numpy.random.seed(seed)

    for random_mat in range(number):

        v1, v2 = generate_runs_vectors(run_lengths[0], run_lengths[1])

        tmp_v1 = MatrixGenerator(name=f'B{random_mat}', format='CSF',
                                 dump_dir=f"{output_dir}/runs_{random_mat}_{run_lengths[0]}_{run_lengths[1]}",
                                 tensor=v1)

        tmp_v2 = MatrixGenerator(name=f'C{random_mat}', format='CSF',
                                 dump_dir=f"{output_dir}/runs_{random_mat}_{run_lengths[0]}_{run_lengths[1]}",
                                 tensor=v2)

        tmp_v1.dump_outputs()
        tmp_v2.dump_outputs()
