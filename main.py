from generate_temporal_data import *
from signature_matching_polynomial_relations import *


def generate_time_series(p, n, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, specified_coeffs = None, monomial_density = None):
    '''
    :param p: degree of polynomial relations for SDEs
    :param n: number of causal variables
    :param specified_edges: list of edges (list of pairs)
    :param driving_noise_scale: scale of the driving noise
    :param measurement_noise_scale: scale of the measurement noise
    :param n_steps: number of observations per time series
    :param monomial_density: proportion of monomials used for causal polynomial relations
    :param specified_coeffs: optional parameter for
    :return:
    '''
    # Generate the complete degree dictionary
    n_monomials, ordered_monomials = generate_monomials(p, n)
    # filter edges based on the specified edges
    nbrs_dict = generate_causal_graph(n, specified_edges=specified_edges)
    # Generate causal parameters
    causal_params = generate_polynomial_relations(nbrs_dict, ordered_monomials, n_seed=2023, specified_coeffs=specified_coeffs, monomial_density=monomial_density)
    # Define the time points for each time series
    t = np.linspace(0, 1, n_steps)
    # generate the data that is compatible with the specified causal relationships
    X = generate_temporal_data(causal_params, t, driving_noise_scale=driving_noise_scale, measurement_noise_scale=measurement_noise_scale)
    return X, t, causal_params, ordered_monomials

def estimate_coefficients(X, t, n, ordered_monomials, solver = 'direct', tol = 0.1, alpha = None):
    n_monomials = len(ordered_monomials)
    # create the necessary subintervals
    subintervals = generate_all_subintervals(t)
    # create the matrix M
    M = compute_M(X, n_monomials, ordered_monomials, subintervals, t, n)
    print(f'Recovered relations from signature matching using {solver} solver and cutoff {tol}:')
    solve_parameters(X, n_monomials, subintervals, M, ordered_monomials, solver=solver, alpha = alpha)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for creating the data
    p = 3 # polynomial degree considered
    n = 2  # number of causal variables
    n_steps = 100  # number of time points per variable
    specified_edges = [(0, 1)]  # list of edges in the causal graph
    monomial_density = None
    driving_noise_scale = 0
    measurement_noise_scale = 0
    # Generate the data
    X, t, causal_params, ordered_monomials = generate_time_series(p, n, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, monomial_density=monomial_density)
    print('Variance of signal: ', np.var(X))
    # Noisy version
    driving_noise_scale = 1
    measurement_noise_scale = 0.1
    # Generate the data
    X_, t, causal_params, ordered_monomials = generate_time_series(p, n, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, monomial_density=monomial_density)
    # Plot the time series data
    plot_time_series_comp(t, X, X_, n, causal_params)
    W = []
    # Choose parameters for solving coefficients
    solver = 'ridge'
    alpha = 1
    tol = 0.1
    # Solve parameters from the original data
    print('Recovered relations from noiseless data: ')
    estimate_coefficients(X, t, n, ordered_monomials, solver=solver, tol=tol, alpha = alpha)
    # Solve parameters from the noisy data
    print(f'Recovered relations from noisy data: ')
    estimate_coefficients(X_, t, n, ordered_monomials, solver=solver, tol=tol, alpha = alpha)


