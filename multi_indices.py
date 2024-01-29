import numpy as np
def generate_multi_indices(l, k, n_params, m, j, n_seed=None):
    '''
    :param l: variable of interest
    :param k: number of multi-indices
    :param n_params: number of parameters needed
    :param m: number of total variables
    :param j: which index to place variables in
    :param n_seed: seed for randomization
    :return:
    '''
    if n_seed is not None:
        np.random.seed(n_seed)

    assert k >= n_params, "k must be greater than or equal to n_params"

    multi_indices = []
    for _ in range(k):
        word_length = np.random.randint(10)
        multi_index = [None] * (max(j, word_length) + 1)  # Ensure sufficient length
        multi_index[j] = l  # Place variable of interest at index j

        # Append additional random elements, avoiding index j
        for i in range(len(multi_index)):
            if i != j:
                multi_index[i] = np.random.randint(0, m)  # Adjust the upper bound as needed

        multi_indices.append(tuple(multi_index))

    return multi_indices
