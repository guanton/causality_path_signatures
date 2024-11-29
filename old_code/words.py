import numpy as np
from itertools import product

def generate_all_words(l, m, k, j=None):
    '''
    Generates all words of length from 1 to k that contain the variable of interest l at least once
    :param l: variable of interest
    :param m: number of total variables
    :param k: maximum word length
    :param j: which index to place variable interest in for each word
    :return: words, interest_indices
    '''
    words = []
    interest_indices = []
    seen_words = set()  # Use set to avoid repetition

    if j is None:
        for length in range(1, k + 1):
            for i in range(length):  # Position where l will be placed
                for indices in product(range(m), repeat=length - 1):  # Generate other parts of the word
                    word = list(indices[:i]) + [l] + list(indices[i:])  # Insert l at the ith position
                    word_tuple = tuple(word)  # Convert to tuple for set operation
                    if word_tuple not in seen_words:
                        seen_words.add(word_tuple)
                        words.append(word)
                        interest_indices.append(i)  # Record the index of l
    else:
        for length in range(1, k + 1):
            for indices in product(range(m), repeat=length):
                word = list(indices)
                # Place variable of interest at index j or at the end if length is less than j
                if length <= j:
                    word[-1] = l
                    interest_idx = len(word) - 1
                else:
                    word[j] = l
                    interest_idx = j
                word_tuple = tuple(word)  # Convert to tuple for set operation
                if word_tuple not in seen_words:
                    seen_words.add(word_tuple)
                    words.append(word)
                    interest_indices.append(interest_idx)

    return words, interest_indices



# def generate_all_words(l, m, j, k):
#     '''
#     Generates all words such that the variable of interest, l, is placed in index j (or the last letter of the word)
#     :param l: variable of interest
#     :param m: number of total variables
#     :param j: which index to place variable interest in for each word
#     :param k: maximum word length
#     :return: words, interest_indices
#     '''
#     words = []
#     interest_indices = []
#     seen_words = set()
#     for length in range(1, k + 1):
#         for indices in product(range(m), repeat=length):
#             # Convert to list for modification
#             word = list(indices)
#             # Place variable of interest at index j or at the end if length is less than j
#             if length <= j:
#                 word[-1] = l
#                 interest_idx = len(word) - 1
#             else:
#                 word[j] = l
#                 interest_idx = j
#             # Convert to tuple so it is hashable for set
#             word_tuple = tuple(word)
#             # Add the word and interest index if not seen before
#             if word_tuple not in seen_words:
#                 seen_words.add(word_tuple)
#                 words.append(word)
#                 interest_indices.append(interest_idx)
#     return words, interest_indices

def generate_words_random(l, n, m, k, n_seed=None):
    '''
    :param l: variable of interest
    :param n: number of randomly created words
    :param m: number of total variables
    :param k: maximum word length
    :param n_seed: seed for randomization
    :return: words, interest_indices
    '''
    if n_seed is not None:
        np.random.seed(n_seed)

    # Feasibility check
    max_possible_words = sum([m**j - (m-1)**j for j in range(k)])  # All words minus those without the variable l
    if n > max_possible_words:
        raise ValueError(f"Cannot generate {n} distinct words with m={m} and k={k}")

    seen_words = set()
    words = []
    interest_indices = []
    while len(seen_words) < n:
        word_length = np.random.randint(1, k + 1)  # Random length from 1 to k
        word = np.random.randint(0, m, word_length).tolist()

        # Ensure l is in the word
        if l not in word:
            insert_index = np.random.randint(0, word_length)
            word[insert_index] = l

        # Record the first index of l
        first_index_of_l = word.index(l)
        word_tuple = tuple(word)  # Convert to tuple for set operations

        # Add to sets if it's a new word
        if word_tuple not in seen_words:
            seen_words.add(word_tuple)
            words.append(word)
            interest_indices.append(first_index_of_l)

    return words, interest_indices


if __name__ == '__main__':
    # Choose test parameters
    l = 2  # Variable of interest
    m = 4  # Total number of variables
    j = 4  # Position to place variable of interest
    k = 3  # Maximum length of multi-indices
    n = 10

    # Call the function
    multi_indices, interest_indices = generate_words_random(l, n, m, k, 0)#generate_all_words(l,m,j,k)

    # Review the results
    print("Words:")
    for mi in multi_indices:
        print(mi)

    print("\nInterest Indices:")
    print(interest_indices)