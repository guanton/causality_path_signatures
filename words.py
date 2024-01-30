import numpy as np
from itertools import product


def generate_all_words(l, m, j, k):
    '''
    Generates all words such that the variable of interest, l, is placed in index j (or the last letter of the word)
    :param l: variable of interest
    :param m: number of total variables
    :param j: which index to place variable interest in for each word
    :param k: maximum word length
    :return: words, interest_indices
    '''
    words = []
    interest_indices = []
    seen_words = set()
    for length in range(1, k + 1):
        for indices in product(range(m), repeat=length):
            # Convert to list for modification
            word = list(indices)
            # Place variable of interest at index j or at the end if length is less than j
            if length <= j:
                word[-1] = l
                interest_idx = len(word) - 1
            else:
                word[j] = l
                interest_idx = j
            # Convert to tuple so it is hashable for set
            word_tuple = tuple(word)
            # Add the word and interest index if not seen before
            if word_tuple not in seen_words:
                seen_words.add(word_tuple)
                words.append(word)
                interest_indices.append(interest_idx)
    return words, interest_indices



def generate_multi_indices_random(l, n, n_params, m, j, k, n_seed=None):
    '''
    :param l: variable of interest
    :param n: number of multi-indices
    :param n_params: number of parameters needed
    :param m: number of total variables
    :param j: which index to place variable interest in
    :param k: maximum length of multi-indices
    :
    :param n_seed: seed for randomization
    :return: multi_indices, interest_indices
    '''
    if n_seed is not None:
        np.random.seed(n_seed)

    assert n >= n_params, "number of multi-indices must be greater than or equal to n_params"

    multi_indices = []
    interest_indices = []
    for _ in range(n):
        word_length = np.random.randint(k)
        multi_index = [None] * (max(j, word_length) + 1)  # Ensure sufficient length
        multi_index[j] = l  # Place variable of interest at index j
        interest_indices.append[j]
        # Append additional random elements, avoiding index j
        for i in range(len(multi_index)):
            if i != j:
                multi_index[i] = np.random.randint(0, m)  # Adjust the upper bound as needed

        multi_indices.append(tuple(multi_index))

    return multi_indices, interest_indices

if __name__ == '__main__':
    # Choose test parameters
    l = 2  # Variable of interest
    m = 4  # Total number of variables
    j = 4  # Position to place variable of interest
    k = 3  # Maximum length of multi-indices

    # Call the function
    multi_indices, interest_indices = generate_all_words(l,m,j,k)

    # Review the results
    print("Words:")
    for mi in multi_indices:
        print(mi)

    print("\nInterest Indices:")
    print(interest_indices)