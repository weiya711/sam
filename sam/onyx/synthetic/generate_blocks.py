import numbers
from sam.onyx.generate_matrices import *
import argparse

TRIES = 1000


def generate_blocks_vectors(num_nonzeros, len, vec_size):

    vec1 = numpy.zeros(vec_size)
    vec2 = numpy.zeros(vec_size)

    filled_vec1 = numpy.zeros(vec_size)
    filled_vec2 = numpy.zeros(vec_size)

    nnz_1 = 0
    nnz_2 = 0
    done = False
    # Create a num of blocks in the first
    # for i in range(num):
    while done is False:
        # Pick them with the len margin on either side
        picked_safe = False
        num_tries = 0
        while picked_safe is False:
            assert num_tries < TRIES, f"Couldn't find a place to put a block...failing"
            # Pick a block placement somewhere on the vector
            inject_idx1 = random.randint(0, vec_size - 1)
            # Assume safe
            safe = True
            # Check that there's room for it
            for margin_idx in range(-1 * len, 2 * len):
                check_margin_idx = inject_idx1 + margin_idx
                if check_margin_idx < 0 or check_margin_idx >= vec_size:
                    continue
                elif filled_vec1[check_margin_idx] != 0:
                    safe = False
            if safe:
                picked_safe = True
            num_tries += 1
        # Set the second block to randomly overlap
        inject_idx2 = inject_idx1 + random.randint(0, len // 2) + len // 4
        for idx_off in range(len):
            if (inject_idx1 + idx_off) >= 0 and (inject_idx1 + idx_off) < vec_size:
                vec1[inject_idx1 + idx_off] = random.randint(0, 1000)
                filled_vec1[inject_idx1 + idx_off] = 1
                nnz_1 += 1
            if (inject_idx2 + idx_off) >= 0 and (inject_idx2 + idx_off) < vec_size:
                vec2[inject_idx2 + idx_off] = random.randint(0, 1000)
                filled_vec2[inject_idx2 + idx_off] = 1
                nnz_2 += 1

            # if nnz_1 >= num_nonzeros or nnz_2 >= num_nonzeros:
                # break

        # nnz_1 += len
        # nnz_2 += len
        if nnz_1 >= num_nonzeros and nnz_2 >= num_nonzeros:
            done = True

    return vec1, vec2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices - block')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_nonzeros", type=int, default=400)
    parser.add_argument("--len_blocks", type=int, default=20)
    # parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    # parser.add_argument("--name", type=str, default='B')
    parser.add_argument("--shape", type=int, nargs="*", default=[10, 10])
    parser.add_argument("--output_format", type=str, default='CSF')
    # parser.add_argument("--num_trials", type=int, default=1000)
    # parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    nnz = args.number_nonzeros
    len_blocks = args.len_blocks
    # sparsity = args.sparsity
    output_dir = args.output_dir
    # name = args.name
    shape = args.shape
    # number = args.number
    output_format = args.output_format

    numpy.random.seed(seed)
    random.seed(seed)

    # for random_mat in range(number):
    # for idx, sparsity in enumerate(sparsities):
    # print(sparsity)
    vec1, vec2 = generate_blocks_vectors(nnz, len_blocks, shape[0])

    v1 = MatrixGenerator(name=f'B', shape=shape, format='CSF',
                         dump_dir=f"{output_dir}/B_blocks_{nnz}_{len_blocks}", tensor=vec1)
    v2 = MatrixGenerator(name=f'C', shape=shape, format='CSF',
                         dump_dir=f"{output_dir}/C_blocks_{nnz}_{len_blocks}", tensor=vec2)

    print(v1)
    print(v2)
    v1.dump_outputs(format=output_format)
    v2.dump_outputs(format=output_format)
