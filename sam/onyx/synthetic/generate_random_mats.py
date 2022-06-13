import numbers
from sam.onyx.generate_matrices import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices - random')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default='B')
    parser.add_argument("--shape", type=int, nargs="*", default=[10, 10])
    # parser.add_argument("--num_trials", type=int, default=1000)
    # parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    sparsity = args.sparsity
    output_dir = args.output_dir
    name = args.name
    shape = args.shape
    number = args.number

    for random_mat in range(number):
        tmp_mat = MatrixGenerator(name=f'B{random_mat}', shape=shape, sparsity=sparsity,
                                  format='CSF', dump_dir=f"{output_dir}/random_{random_mat}_sp_{sparsity}", tensor=None)
        tmp_mat.dump_outputs()
