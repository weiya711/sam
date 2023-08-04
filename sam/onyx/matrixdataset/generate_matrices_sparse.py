import numpy as np
import scipy.sparse as sp

def generate_sparse_matrix(rows, cols, sparsity, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_elements = rows * cols
    num_nonzero_elements = int(num_elements * sparsity)
    values = np.random.rand(num_nonzero_elements)
    indices = np.random.choice(num_elements, num_nonzero_elements, replace=False)
    matrix = sp.csr_matrix((values, indices.reshape(-1, 1)), shape=(rows, cols))
    return matrix

def save_matrix_to_mtx(matrix, filename):
    sp.save_npz(filename, matrix)

def main():
    num_matrices = 1
    matrix_sizes = [(10,10)]
    sparsities = [0.5]
    # matrix_sizes = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]  # You can adjust the sizes as needed
    # sparsities = [0.1, 0.2, 0.3, 0.4, 0.5]  # Adjust the sparsity levels as needed (0.0 = dense, 1.0 = fully sparse)

    seed = 42  # Set your desired seed here for reproducibility

    for i in range(num_matrices):
        rows = 10
        cols = 10
        sparsity = 0.5
        matrix = generate_sparse_matrix(rows, cols, sparsity, seed=seed)
        print(f"Matrix {i + 1} (Size: {rows}x{cols}, Sparsity: {sparsity}):")
        print(matrix.toarray())
        print()

        filename = f"matrix_{i + 1}.mtx"
        save_matrix_to_mtx(matrix, filename)

if __name__ == "__main__":
    main()
