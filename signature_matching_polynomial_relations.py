import numpy as np
from sklearn.linear_model import Ridge
from generate_temporal_data import get_termstring


def generate_all_subintervals(t, random_start = False):
    subintervals = []
    min_t = t[0]
    for i in range(1, len(t)):
        max_t = t[i]
        indices = np.where((t >= min_t) & (t <= max_t))
        subintervals.append(indices[0])
    return subintervals

def generate_subintervals(t, n_params, random_start = False):
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
    :param x_i: time_series data for x_I
    :param subintervals: time indices in the relevant subintervals
    :return:
    """
    level_1_paths = np.empty((len(subintervals), 1), dtype=np.float64)
    for i, subinterval in enumerate(subintervals):
        start = subinterval[0]
        end = subinterval[-1]
        path = x_i[end] - x_i[start]
        level_1_paths[i, 0] = path
    return level_1_paths

def compute_M(X, n_params, ordered_monomials, subintervals, t, n):
    """
    This computes the corresponding matrix M (n_params x n_params) for variable x_i
    :param X: time series data for all variables x_i
    :param n_params: number of parameters to recover
    :param ordered_monomials: dictionary of monomials, ordered
    :param subintervals: list of n_params subintervals
    :param t: array of times (ex. np.linspace(0, 1, 100))
    :param n: number of variables
    :return:
    """
    M = np.zeros((len(t)-1, n_params))
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
            integral = np.trapz(monomial_values_subinterval, t_sub)
            # set corresponding matrix entry
            M[i, j] = integral
    return M


def solve_parameters(X, n_monomials, subintervals, M, order_mapping, alpha=1, tol = 1e-1, solver = 'direct'):
    """

    :param X: time series data for all variables x_i
    :param n_monomials:  number of parameters to recover
    :param subintervals: list of n_monomials subintervals
    :param M:  matrix M (n_params x n_params) for variable x_i (it is assumed we use the same subintervals for each
    variable and hence the same matrix M).
    :param order_mapping: dictionary of monomials, ordered
    :param alpha: regularization parameter for ridge regression (L^2)
    :param tol: sparsity filter for reporting terms
    :return:
    """
    n = X.shape[1]
    # Initialize an empty array to store the coefficients
    coefficients = np.zeros((n, n_monomials))
    for i in range(n):
        x_i = X[:, i]
        # compute b using the level 1 iterated integrals
        b = compute_level_1_paths(x_i, subintervals)
        if solver == 'ridge':
            # Use Ridge regression with regularization
            clf = Ridge(alpha=alpha)
            clf.fit(M, b)
            params_i = clf.coef_
            coefficients[i] = params_i
            string = f'dx_{i}/dt='
            for j in range(len(params_i[0])):
                if abs(params_i[0][j]) > tol:
                    string += f'{round(params_i[0][j],2)} {get_termstring(order_mapping[j])} +'
            string = string[:-1]
            print(string)
        elif solver == 'direct':
            params_i = np.linalg.solve(M, b)
            string = f'dx_{i}/dt='
            for j in range(len(params_i)):
                if abs(params_i[j][0]) > tol:
                    string += f'{round(params_i[0][j],2)} {get_termstring(order_mapping[j])} +'
            string = string[:-1]
            print(string)
    return coefficients
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


