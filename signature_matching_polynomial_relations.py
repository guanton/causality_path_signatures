import numpy as np
from sklearn.linear_model import Ridge
from generate_temporal_data import get_termstring

def solve_M(X, n_params, order_mapping, subintervals, t):
    M = np.zeros((n_params, n_params))
    # Compute the matrix M
    for i, subinterval in enumerate(subintervals):
        t_sub = [t[idx] for idx in subinterval]
        for j in range(n_params):
            # extract the n-tuple representing the polynomial term in hand
            term = order_mapping[j]
            term_values_in_sub = []
            for idx in subinterval[0]:
                term_at_idx = np.prod([X[idx, k] ** term[k] for k in range(len(term))])
                term_values_in_sub.append(term_at_idx)
            # integrate the term with respect to the ith subinterval
            integral = np.trapz(term_values_in_sub, t_sub)
            # set corresponding matrix entry
            M[i, j] = integral
    return M


def solve_parameters(X, n_monomials, subintervals, M, order_mapping, alpha=1.0, tol = 1e-3):
    n = X.shape[1]
    # Initialize an empty array to store the coefficients
    coefficients = np.zeros((n, n_monomials))

    for i in range(n):
        x_i = X[:, i]
        b = compute_level_1_paths(x_i, subintervals)

        # Use Ridge regression with regularization
        clf = Ridge(alpha=alpha)
        clf.fit(M, b)
        params_i = clf.coef_
        coefficients[i] = params_i

        print(f'x_{i}=')
        # print(params_i)
        for j in range(len(params_i[0])):
            if abs(params_i[0][j]) > tol:
                print(f'{params_i[0][j]} {get_termstring(order_mapping[j])} +')
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

def compute_level_1_paths(x_i, subintervals):
    level_1_paths = np.empty((len(subintervals), 1), dtype=np.float64)
    for i, subinterval in enumerate(subintervals):
        start = subinterval[0][0]
        end = subinterval[0][-1]
        path = x_i[end] - x_i[start]
        level_1_paths[i, 0] = path
    return level_1_paths

def generate_subintervals(t, n_params):
    subintervals = []

    # Create random starting points for the subintervals
    start_points = np.sort(np.random.choice(t, size=n_params, replace=False))

    # Generate random subinterval end points for each start point
    end_points = start_points + np.random.uniform(0.01, 1, size=n_params)  # Adjust the range and size as needed

    # Ensure that end points are within the time interval
    end_points = np.clip(end_points, 0, 1)

    for i in range(n_params):
        # Find the indices corresponding to the time range
        indices = np.where((t >= start_points[i]) & (t <= end_points[i]))
        subintervals.append(indices)

    return subintervals