import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from generate_temporal_data import get_termstring
from sklearn.preprocessing import StandardScaler


def generate_all_subintervals(t):
    subintervals = []
    n = len(t)
    for i in range(n):
        for j in range(i + 1, n):
            min_t = t[i]
            max_t = t[j]
            indices = np.where((t >= min_t) & (t <= max_t))
            subintervals.append(indices[0])
    return subintervals

def generate_random_subintervals(t, n):
    subintervals = []
    n_points = len(t)
    for _ in range(n):
        # Randomly choose indices for min_t and max_t
        min_index = np.random.randint(0, n_points-1)
        max_index = np.random.randint(min_index + 1, n_points)  # Ensure max_index > min_index
        # Extract subinterval indices
        indices = np.where((t >= t[min_index]) & (t <= t[max_index]))
        subintervals.append(indices[0])
    return subintervals

def generate_all_subintervals_0(t):
    subintervals = []
    min_t = t[0]
    for i in range(1, len(t)):
        max_t = t[i]
        indices = np.where((t >= min_t) & (t <= max_t))
        subintervals.append(indices[0])
    return subintervals

def generate_subintervals_n_params(t, n_params, random_start = False):
    '''
    generates a random set of subintervals on which we will compute iterated integarals
    :param t: array of times (default: np.linspace(0, 1, 100))
    :param n_params: number of parameters to recover = number of subintervals to generate
    :return:
    subintervals: list of lists (each subinterval is a list of time indices)
    '''
    subintervals = []
    min_t = t[0]
    max_t = t[-1]
    if random_start:
        # Create random starting points for the subintervals
        start_points = np.sort(np.random.choice(t, size=n_params, replace=False))
        # Generate random subinterval end points for each start point
        end_points = start_points + np.random.uniform(0.01, max_t - min_t,
                                                      size=n_params)  # Adjust the range and size as needed
    else:
        start_points = np.full(n_params, min_t)
        end_points = np.random.uniform(min_t + 0.01, max_t,
                                                      size=n_params)  # Adjust the range and size as needed
    # Ensure that end points are within the time interval
    end_points = np.clip(end_points, min_t, max_t)
    for i in range(n_params):
        # Find the indices corresponding to the time range for each sampled subinterval
        indices = np.where((t >= start_points[i]) & (t <= end_points[i]))
        subintervals.append(indices[0])
    return subintervals

def compute_level_1_paths(x_i, subintervals):
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

def compute_level_2_iterated_integral(x_i, x_j, t, subinterval):
    a = subinterval[0]
    t_sub = [t[idx] for idx in subinterval]
    integrand = [np.prod([compute_level_1_path(x_i, a, subinterval[s]), (x_j[s]-x_j[max(0, s-1)])]) for s in range(len(subinterval))]
    integral = np.trapz(integrand, t_sub)
    return integral



def compute_level_k_iterated_integral(k, indices, t, subinterval):
    assert k == indices.sum()
    t_sub = [t[idx] for idx in subinterval]
    for i in range(k):
        curr_idx = indices[i]


def compute_M(X, n_params, ordered_monomials, subintervals, t, n, level):
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
    elif level ==2:
        M = np.zeros(((n+1)*len(subintervals), n_params))
    # iterate over the subintervals to determine the rows
    for i, subinterval in enumerate(subintervals):
        # retrieve the actual time values of the subinterval
        t_sub = [t[idx] for idx in subinterval]
        # iterate over the ordered monomials for the column
        for j in range(n_params):
            # extract the monomial (list representation)
            monomial = ordered_monomials[j]
            relevant_variables = [v for v in range(n) if monomial[v] != 0]
            degrees = [monomial[v] for v in relevant_variables]
            # collect all monomial values over all time indices in the subinterval
            monomial_values_subinterval = [np.prod([X[idx, v] ** degree for v, degree in zip(relevant_variables, degrees)]) for idx in subinterval]
            # integrate the jth monomial with respect to the ith subinterval using trapezoidal method
            integral = np.trapz(monomial_values_subinterval, x= t_sub)
            # set corresponding matrix entry
            M[i, j] = integral
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



def solve_parameters(X, t, n_monomials, subintervals, M, ordered_monomials, alpha=1, tol = 1e-1, solver = 'direct', level = 1):
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
            level_2_paths = compute_level_2_paths(X, i, n, t, subintervals)
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


