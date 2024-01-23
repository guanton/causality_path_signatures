import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from generate_temporal_data import *
from sklearn.preprocessing import StandardScaler

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
            indices = np.where((t >= t[min_index]) & (t <= t[max_index]))
            subintervals.append(indices[0])
    elif sub_mode == 'all':
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                min_t = t[i]
                max_t = t[j]
                indices = np.where((t >= min_t) & (t <= max_t))
                subintervals.append(indices[0])
    elif sub_mode == 'zeros':
        min_t = t[0]
        for i in range(1, len(t)):
            max_t = t[i]
            indices = np.where((t >= min_t) & (t <= max_t))
            subintervals.append(indices[0])
    elif sub_mode == 'adjacent':
        delta = 5
        for i in range(len(t) - delta):
            min_t = t[i]
            max_t = t[i + delta]
            indices = np.where((t >= min_t) & (t <= max_t))
            subintervals.append(indices[0])
    elif sub_mode == 'one':
        min_t = t[0]
        max_t = t[-1]
        indices = np.where((t >= min_t) & (t <= max_t))
        subintervals.append(indices[0])
    return subintervals

def generate_multi_indices(l, k, n_params, m, n_seed=None):
    if n_seed is not None:
        np.random.seed(n_seed)

    assert k >= n_params, "k must be greater than or equal to n_params"

    multi_indices = []
    for _ in range(k):
        multi_index = [l] # begin with the variable of interest
        word_length = np.random.randint(0, m-1)
        # Append additional elements to the multi-index
        for _ in range(word_length):
            multi_index.append(np.random.randint(0, m))  # Adjust the upper bound as needed

        multi_indices.append(tuple(multi_index))

    return multi_indices




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


def compute_level_1_path(x_i, a, b):
    '''
    This computes S^i_(a,b) where (a,b) are the endpoints of the subinterval
    :param x_i: time_series data for x_i
    :param a: start time index
    :param b: end time index
    :return: path integral S^i_(a,b)
    '''
    path = x_i[b] - x_i[a]
    return path

def compute_path_monomial(X, m, copy, monomial, a, b):
    '''
    :param X: df for time series data for all variables X_i
    :param copy: which copy of the time series we are using
    :param monomial: the monomial that we will be integrating against
    :param a: start time index
    :param b: end time index
    :return:
    '''
    relevant_variables = [v for v in range(m) if monomial[v] != 0]
    degrees = [monomial[v] for v in relevant_variables]


    monomial_value_a, monomial_value_b = 1, 1
    for v, degree in zip(relevant_variables, degrees):
        x_v = X.loc[:, (v, copy)].to_numpy()
        monomial_value_a *= x_v[a] ** degree
        monomial_value_b *= x_v[b] ** degree
    return monomial_value_b - monomial_value_a


def compute_integral(t, integrator, derivatives_df, i_k, copy, j):
    '''
    :param t: array of times
    :param integrator: integrator function as array of its values from time t[0] to t[j]
    :param derivatives_df: df for the derivatives of the time series data
    :param i_k: the index of the variable that we are integrating aginst
    :param copy: the index of the copy that we are considering for the variables
    :param j: t[j] is the endpoint of the integral
    :return:
    '''
    mi = (i_k, copy)
    h = (t[-1] - t[0])/(len(t)-1)
    derivatives_array = derivatives_df.loc[:j, mi].to_numpy()
    integral = 0
    assert len(integrator) == j, "integrator dimension does not align with the time interval"
    for i in range(j):
        rect = integrator[i] * derivatives_array[i] * h
        integral += rect
    return integral

def compute_iterated_integral(X, m, derivatives_df, copy, multi_index, subinterval, monomial = None):
    t = X.index.to_numpy()
    integrator = [0] * len(subinterval)  # Initialize integrator as a list
    for k in range(len(multi_index)):
        for s in subinterval:
            i_k = multi_index[k]  # extract current index
            if k == 0:
                if monomial is None:
                    x_i = X.loc[:, (i_k, copy)].to_numpy()
                    integrator[s] = compute_level_1_path(x_i, 0, s)
                else:
                    integrator[s] = compute_path_monomial(X, m, copy, monomial, 0, s)
            else:
                integrator_s = integrator[:s]
                integrator[s] = compute_integral(t, integrator_s, derivatives_df, i_k, copy, s)
    return integrator[-1]


def compute_M(X, m, l, derivatives_df, n_params, ordered_monomials, multi_indices, sub, copy):
    """
        This computes the corresponding matrix M (n_params x n_params) for variable x_i
        :param X: df for time series data for all variables X_i
        :param l: the causal variable of interest
        :param n_params: number of parameters to recover (monomial coefficients)
        :param ordered_monomials: dictionary of monomials, ordered
        :param multi_indices: list of multi-indices (all beginning with i)
        :param sub:
        :return:
        """
    M = np.zeros((len(multi_indices), n_params))
    for i, multi_index in enumerate(multi_indices):
        assert multi_index[0] == l, f"the multi-index must begin with the specified variable of interest {i}"
        for j in range(n_params):
            monomial = ordered_monomials[j]
            M[i, j] = compute_iterated_integral(X, m, derivatives_df, copy, multi_index, sub, monomial = monomial)
    return M


def compute_b(X, m, l, derivatives_df, multi_indices, sub, copy):
    b = np.empty((len(multi_indices), 1), dtype=np.float64)
    for i, multi_index in enumerate(multi_indices):
        assert multi_index[0] == l, f"the multi-index must begin with the specified variable of interest {i}"
        b[i, 0] = compute_iterated_integral(X, m, derivatives_df, copy, multi_index, sub)
    return b



def solve_parameters(X, m, derivatives_df, multi_indices, ordered_monomials, sub, copy, alpha=1, tol = 1e-1, solver = 'direct', level = 1):
    """
    :param X: time series data for all variables x_i
    :param n_monomials:  number of parameters to recover
    :param subintervals: list of n_monomials subintervals
    :param M:  matrix M (n_integrals x n_params) for variable x_i (it is assumed we use the same subintervals for each
    variable and hence the same matrix M).
    :param b:  vector b of iterated integrls (n_integrals x 1)
    :param ordered_monomials: dictionary of monomials, ordered
    :param alpha: regularization parameter for ridge regression (L^2)
    :param tol: sparsity filter for reporting terms
    :return:
    """
    recovered_causal_params = {}
    n_params = len(ordered_monomials)
    for l in range(m):
        b = compute_b(X, m, l, derivatives_df, multi_indices[l], sub, copy)
        M = compute_M(X, m, l, derivatives_df, n_params, ordered_monomials, multi_indices[l], sub, copy)
        # compute b using the level 1 iterated integrals
        if solver == 'ridge':
            # Use Ridge regression with regularization
            clf = Ridge(alpha=alpha)
            clf.fit(M, b)
            params_i = clf.coef_[0]
        if solver == 'LR':
            model = LinearRegression().fit(M, b)
            params_i = [x[0] for x in model.coef_.T]
        if solver == 'OLS':
            params_i, residuals, rank, singular_values = np.linalg.lstsq(M, b, rcond=None)
            params_i = [x[0] for x in params_i]
        if solver == 'pseudo-inverse':
            params_i = [x[0] for x in np.linalg.pinv(M) @ b]
        if solver == 'lasso':
            # Use Ridge regression with regularization
            clf = Lasso(alpha=alpha)
            clf.fit(M, b)
            params_i = clf.coef_
            # print('params_lasso:', params_i)
        elif solver == 'direct':
            params_i = np.linalg.solve(M, b)
            params_i = [x[0] for x in params_i]
        recovered_causal_params[l] = {}
        for j in range(n_params):
            if abs(params_i[j]) > tol:
                recovered_causal_params[l][params_i[j]]=ordered_monomials[j]
        string = f'dx_{l}/dt= '
        for j in range(len(params_i)):
            if abs(params_i[j]) > tol:
                string += f'{round(params_i[j], 2)}{get_termstring(ordered_monomials[j])} +'
        string = string[:-1]
        print(string)
    return recovered_causal_params


def compute_M_subs(X, n_params, ordered_monomials, subintervals, t, n, level, sample_noise = True, sub_dict = None, driving_noise_scale = 0.1):
    """
    This computes the corresponding matrix M (n_params x n_params) for variable x_i
    :param X: time series data for all variables x_i
    :param i: the causal variable of interest
    :param n_params: number of parameters to recover
    :param ordered_monomials: dictionary of monomials, ordered
    :param dimension of b
    :param t: array of times (ex. np.linspace(0, 1, 100))
    :param n: number of variables
    :return:
    """
    if level == 1:
        M = np.zeros((len(subintervals), n_params))
    elif level == 2:
        M = np.zeros(((n+1)*len(subintervals), n_params))
    # iterate over the subintervals to determine the rows
    for i, subinterval in enumerate(subintervals):
        # retrieve the actual time values of the subinterval
        t_sub = [t[idx] for idx in subinterval]
        # iterate over the ordered monomials for the column
        for j in range(n_params):
            if sample_noise:
                t_1, t_2 = sub_dict[i]
                noise = np.random.randn() * driving_noise_scale * np.sqrt(t_2 - t_1)
            else:
                noise = 0
            # extract the monomial (list representation)
            monomial = ordered_monomials[j]
            relevant_variables = [v for v in range(n) if monomial[v] != 0]
            degrees = [monomial[v] for v in relevant_variables]
            # collect all monomial values over all time indices in the subinterval
            monomial_values_subinterval = [np.prod([X[idx, v] ** degree for v, degree in zip(relevant_variables, degrees)]) for idx in subinterval]
            # integrate the jth monomial with respect to the ith subinterval using trapezoidal method
            integral = np.trapz(monomial_values_subinterval, x= t_sub)
            # set corresponding matrix entry
            M[i, j] = integral + noise
    if level == 2:
        for i in range(n):
            x_i = X[:, i]
            for k, subinterval in enumerate(subintervals):
                t_sub = [t[idx] for idx in subinterval]
                for j in range(n_params):
                    monomial = ordered_monomials[j]
                    relevant_variables = [v for v in range(n) if monomial[v] != 0]
                    degrees = [monomial[v] for v in relevant_variables]
                    # collect all monomial values over all time indices in the subinterval
                    monomial_values_subinterval = [
                        np.prod([X[idx, v] ** degree for v, degree in zip(relevant_variables, degrees)]) for idx in
                        subinterval]
                    level_1_paths_subinterval = [compute_level_1_path(x_i, subinterval[0], idx) for idx in
                                                 subinterval]
                    # for monomial_value, level_1_path in zip(monomial_values_subinterval, level_1_paths_subinterval):
                    #     # print('pair:', monomial_value, level_1_path)

                    integrand = [np.prod([monomial_value, level_1_path]) for monomial_value, level_1_path in
                                 zip(monomial_values_subinterval, level_1_paths_subinterval)]
                    integral = np.trapz(integrand, t_sub)
                    # set corresponding matrix entry
                    M[(i+1)*len(subintervals) + k, j] = integral
    return M





def solve_parameters_sub(X, t, n_monomials, subintervals, M, ordered_monomials, alpha=1, tol = 1e-1, solver = 'direct', level = 1):
    """
    :param X: time series data for all variables x_i
    :param n_monomials:  number of parameters to recover
    :param subintervals: list of n_monomials subintervals
    :param M:  matrix M (n_integrals x n_params) for variable x_i (it is assumed we use the same subintervals for each
    variable and hence the same matrix M).
    :param b:  vector b of iterated integrls (n_integrals x 1)
    :param ordered_monomials: dictionary of monomials, ordered
    :param alpha: regularization parameter for ridge regression (L^2)
    :param tol: sparsity filter for reporting terms
    :return:
    """
    # scaler = StandardScaler()
    # M = scaler.fit_transform(M)
    recovered_causal_params = {}
    n = X.shape[1]
    for i in range(n):
        x_i = X[:, i]
        # compute b
        b = compute_level_1_paths(x_i, subintervals)
        if level == 2:
            # indices = np.where((t >= t[0]) & (t <= t[-1]))
            level_2_paths = compute_level_2_paths(X, i, n, t, subintervals = subintervals)
            merged_paths = np.concatenate([b, level_2_paths], axis=0)
            b = merged_paths
        recovered_causal_params[i] = {}
        # compute b using the level 1 iterated integrals
        if solver == 'ridge':
            # Use Ridge regression with regularization
            clf = Ridge(alpha=alpha)
            clf.fit(M, b)
            params_i = clf.coef_[0]
        if solver == 'LR':
            model = LinearRegression().fit(M, b)
            params_i = [x[0] for x in model.coef_.T]
        if solver == 'OLS':
            params_i, residuals, rank, singular_values = np.linalg.lstsq(M, b, rcond=None)
            params_i = [x[0] for x in params_i]
        if solver == 'pseudo-inverse':
            params_i = [x[0] for x in np.linalg.pinv(M) @ b]
        if solver == 'lasso':
            # Use Ridge regression with regularization
            clf = Lasso(alpha=alpha)
            clf.fit(M, b)
            params_i = clf.coef_
            # print('params_lasso:', params_i)
        elif solver == 'direct':
            params_i = np.linalg.solve(M, b)
            params_i = [x[0] for x in params_i]
        recovered_causal_params[i] = {}
        for j in range(n_monomials):
            if abs(params_i[j]) > tol:
                recovered_causal_params[i][params_i[j]]=ordered_monomials[j]
        string = f'dx_{i}/dt= '
        for j in range(len(params_i)):
            if abs(params_i[j]) > tol:
                string += f'{round(params_i[j], 2)}{get_termstring(ordered_monomials[j])} +'
        string = string[:-1]
        print(string)
    return recovered_causal_params

if __name__ == '__main__':
    m = 2
    n_series = 3
    pa_dict = generate_causal_graph(m, [(0, 0), (1, 0)])
    list_poly_strings = ['3x_0x_1^2 + -7x_1 +5', '-0.2']
    # i = 0
    # for p in list_poly_strings:
    #     print(f'dx_{i}/dt= {p}')
    #     i += 1
    causal_params = parse_polynomial_strings(list_poly_strings, pa_dict)
    print_causal_relationships(causal_params)
    t = np.linspace(0, 1, 10)
    X = generate_temporal_data(causal_params, t, driving_noise_scale=0.1, measurement_noise_scale=0, n_series=n_series,
                               zero_init=True, n_seed=0)
    print(X.head())
    derivative_df = calculate_derivative_df(X, t, m, n_series)
    print(derivative_df.head())
    sub = generate_subintervals(t, 'one')[0].tolist()
    print(sub)
    print('integral:', compute_integral(t, [1, 1,1], derivative_df, 1, 1, 3))
    print('integral:', compute_integral(t, [1,1], derivative_df, 1, 0, 2))
    print('integral:', compute_integral(t, [0, 2], derivative_df, 1, 0, 2))
    integral = compute_iterated_integral(X, m, derivative_df, 0, [0], sub, monomial = [1, 2])
    print(integral)



def compute_level_1_paths(x_i, subintervals, samples_per_sub = 1):
    """
    This computes the corresponding b column vector (n_params x 1) for variable x_i
    :param x_i: time_series data for x_i
    :param subintervals: time indices in the relevant subintervals
    :return:
    """
    level_1_paths = np.empty((len(subintervals), 1), dtype=np.float64)
    for i, subinterval in enumerate(subintervals):
        level_1_paths[i, 0] = compute_level_1_path(x_i, subinterval[0], subinterval[-1])
    return level_1_paths

def compute_level_2_paths(X, i, n, t, subintervals):
    x_i = X[:, i]
    level_2_paths = np.empty((n*len(subintervals), 1), dtype=np.float64)
    for j in range(n):
        for k in range(len(subintervals)):
            x_j = X[:, j]
            level_2_paths[j + k, 0] = compute_level_2_iterated_integral(x_j, x_i, t, subintervals[k])
    return level_2_paths



def compute_level_2_iterated_integral(x_i, x_j, t, subinterval):
    a = subinterval[0]
    t_sub = [t[idx] for idx in subinterval]
    integrand = [np.prod([compute_level_1_path(x_i, a, subinterval[s]), (x_j[s]-x_j[max(0, s-1)])]) for s in range(len(subinterval))]
    integral = np.trapz(integrand, t_sub)
    return integral
#
# def solve_parameters(X, n_monomials, subintervals, M, order_mapping):
#     n = X.shape[1]
#     M = np.zeros((n_monomials, n_monomials))
#     for i in range(n):
#         x_i = X[:, i]
#         b = compute_level_1_paths(x_i, subintervals)
#         params_i = np.linalg.solve(M, b)
#         print(f'coefficients for {x_i}')
#         for j in range(len(params_i)):
#             if params_i[j] != 0:
#                 print(f'{params_i[j]} {get_termstring(order_mapping[j])}')
#     return


