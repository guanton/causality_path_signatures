import numpy as np
def generate_subintervals(t, sub_mode, n = None):
    '''
    generates subintervals according to the specified mode
    :param t: array of times (ex. np.linspace(0, 1, 100))
    :param sub_mode: random, all, zeros, or adjacent
    :param n: number of subintervals to be created
    :return:
    '''
    subintervals = []
    if sub_mode == 'random':
        if n is None:
            n = 2*len(t)
        for _ in range(n):
            # Randomly choose indices for min_t and max_t
            min_index = np.random.randint(0, len(t) - 1)
            max_index = np.random.randint(min_index + 1, len(t))  # Ensure max_index > min_index
            # Extract subinterval indices
            indices = np.where((t >= t[min_index]) & (t <= t[max_index]))[0].tolist()
            subintervals.append(indices)
    elif sub_mode == 'all':
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                min_t = t[i]
                max_t = t[j]
                indices = np.where((t >= min_t) & (t <= max_t))[0].tolist()
                subintervals.append(indices)
    elif sub_mode == 'zeros':
        min_t = t[0]
        for i in range(1, len(t)):
            max_t = t[i]
            indices = np.where((t >= min_t) & (t <= max_t))[0].tolist()
            subintervals.append(indices)
    elif sub_mode == 'adjacent':
        delta = 5
        for i in range(len(t) - delta):
            min_t = t[i]
            max_t = t[i + delta]
            indices = np.where((t >= min_t) & (t <= max_t))[0].tolist()
            subintervals.append(indices)
    elif sub_mode == 'one':
        min_t = t[0]
        max_t = t[-1]
        indices = np.where((t >= min_t) & (t <= max_t))[0].tolist()
        subintervals.append(indices)
    return subintervals

def create_subintervals_dict(subintervals, t):
    '''
    organizes subintervals according to their order in the list and time values
    :param subintervals: list of subintervals (list of pairs of indices)
    :param t: array of times (ex. np.linspace(0, 1, 100))
    :return:
    '''
    sub_dict = {}
    counter = 0
    for subinterval in subintervals:
        t_1 = t[subinterval[0]]
        t_2 = t[subinterval[1]]
        sub_dict[counter] = (t_1, t_2)
        counter += 1
    return sub_dict

if __name__ == '__main__':
    t = np.linspace(0,1, 10)
    sub_mode = 'random'
    subs = generate_subintervals(t, sub_mode)
    print(subs)