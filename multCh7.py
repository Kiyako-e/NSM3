import numpy as np
from itertools import permutations
import math

def multCh7(our_matrix):
    k = our_matrix.shape[1]
    n = our_matrix.shape[0]
    n_perm = math.factorial(k) ** n
    outp = np.zeros((n, k, int(n_perm)), dtype=our_matrix.dtype)
    sorted_rows = np.apply_along_axis(np.sort, axis=1, arr=our_matrix)
    possible_row_arr = np.zeros((int(math.factorial(k)), k, n), dtype=our_matrix.dtype)

    for i in range(n):
        possible_row_arr[:, :, i] = np.array([list(p) for p in permutations(sorted_rows[i])])

    def get_mat(index):
        possible_mat = None
        for j in range(n):
            if possible_mat is None:
                possible_mat = possible_row_arr[index[j], :, j]
            else:
                possible_mat = np.vstack((possible_mat, possible_row_arr[index[j], :, j]))
        return possible_mat

    index_grid = np.array([list(p) for p in np.array(np.meshgrid(*[np.arange(int(math.factorial(k))) for _ in range(n)])).T.reshape(-1, n)], dtype=int)

    for i in range(int(n_perm)):
        outp[:, :, i] = get_mat(index_grid[i])

    return outp
