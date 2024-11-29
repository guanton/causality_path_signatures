import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from subintervals import *
from words import *
from formatting_helpers import *
from sklearn.preprocessing import StandardScaler
from integrals import *
def compute_M(X, m, l, derivatives_df, n_params, ordered_monomials, words, interest_indices, subs, copy):
    """
    This computes the corresponding matrix M (n_params x n_params) for variable x_l
    :param X: df for time series data for all variables X_i
    :param l: the causal variable of interest
    :param n_params: number of parameters to recover (monomial coefficients)
    :param ordered_monomials: dictionary of monomials, ordered
    :param words: list of multi-indices (all featuring l at some position)
    :param interest_indices: list of indices for the position of l (to be replaced by monomial)
    :param sub:
    :param copy:
    :return:
    """
    # t = X.index
    # X = convert_df_to_array(X)
    # level = 1
    # sub_dict = create_subintervals_dict(subs, t)
    # return compute_M_subs(X, n_params, ordered_monomials, subs, t, m, level, sample_noise = True, sub_dict = sub_dict, driving_noise_scale = 0.1)
    M = np.zeros((len(subs)*len(words), n_params))
    for k, sub in enumerate(subs):
        for i, word in enumerate(words):
            # assert word[0] == l, f"the multi-index must begin with the specified variable of interest {i}"
            for j in range(n_params):
                monomial = ordered_monomials[j]
                M[i+k*len(words), j] = compute_iterated_integral(X, m, derivatives_df, copy, l, word, sub, monomial=monomial, j_index = interest_indices[i])
    return M


def compute_b(X, m, l, derivatives_df, words, subs, copy):
    # x_l = X.loc[:, (l,copy)].tolist()
    # return compute_level_1_paths(x_l, subs)

    b = np.empty((len(subs)*len(words), 1), dtype=np.float64)
    for i, sub in enumerate(subs):
        for j, word in enumerate(words):
            b[i*len(word) + j, 0] = compute_iterated_integral(X, m, derivatives_df, copy, l, word, sub)
    return b



def solve_parameters(X, m, derivatives_df, words_per_variable, interest_indices_per_variable, ordered_monomials, sub, copies, alpha=1, tol = 1e-1, solver = 'direct', level = 1):
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
        bs = []
        Ms = []
        for copy in copies:
            b = compute_b(X, m, l, derivatives_df, words_per_variable[l], sub, copy)
            bs.append(b)
            M = compute_M(X, m, l, derivatives_df, n_params, ordered_monomials, words_per_variable[l], interest_indices_per_variable[l], sub, copy)
            Ms.append(M)
        # average over the n_series different copies
        if len(copies) > 1:
            b = np.average(bs, axis=0)
            M = np.average(Ms, axis=0)
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
        recovered_causal_params[l] = []
        for j in range(n_params):
            if abs(params_i[j]) > tol:
                recovered_causal_params[l].append((params_i[j], ordered_monomials[j]))
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
        recovered_causal_params[i] = []
        for j in range(n_monomials):
            if abs(params_i[j]) > tol:
                recovered_causal_params[i].append((params_i[j], ordered_monomials[j]))
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


