from generate_temporal_data import *
from signature_matching_polynomial_relations import *


def generate_time_series(p, n, specified_edges, noise, n_steps, specified_coeffs = None):
    '''
    :param p: degree of polynomial relations for SDEs
    :param n: number of causal variables
    :param specified_edges: list of edges (
    :param noise:
    :param n_steps:
    :param specified_coeffs:
    :return:
    '''
    # Generate the complete degree dictionary
    n_monomials, degree_dict, ordered_monomials = generate_monomials(p, n)
    # filter edges based on the specified edges
    nbrs_dict = generate_causal_graph(n, specified_edges=specified_edges)
    # Generate causal parameters
    causal_params = generate_polynomial_relations(nbrs_dict, ordered_monomials, n_seed=2023, specified_coeffs=specified_coeffs)
    print('causal params:', causal_params)
    # Define the time points for each time series
    t = np.linspace(0, 1, n_steps)
    # generate the data that is compatible with the specified causal relationships
    X = generate_temporal_data(causal_params, t, noise=noise)
    return X, t, causal_params, ordered_monomials

def estimate_coefficients(X, t, n, ordered_monomials, solver = 'direct', tol = 0.1):
    n_monomials = len(ordered_monomials)
    # create the necessary subintervals
    subintervals = generate_subintervals(t, n_monomials)
    # create the matrix M
    M = compute_M(X, n_monomials, ordered_monomials, subintervals, t, n)
    print(f'Recovered relations from signature matching using {solver} solver and cutoff {tol}:')
    solve_parameters(X, n_monomials, subintervals, M, ordered_monomials, solver=solver)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for creating the data
    p = 2 # polynomial degree considered
    n = 2 # number of causal variables
    n_steps = 50 # number of time points per variable
    specified_edges = [(0,1)] # list of edges in the causal graph
    noise = None # do not add noise
    # Generate the data
    X, t, causal_params, ordered_monomials = generate_time_series(p, n, specified_edges, noise, n_steps)

    # Noisy version
    specified_edges = [(0, 1)]  # list of edges in the causal graph
    noise = 'driving' # add Brownian motion noise
    # Generate the data
    X_, t, causal_params, ordered_monomials = generate_time_series(n, p, specified_edges, noise, n_steps)
    # Plot the time series data
    plot_time_series_comp(t, X, X_, n, causal_params)

    # Choose parameters for solving coefficients
    solver = 'direct'
    tol = 0.1
    # Solve parameters from the original data
    print('original: ')
    estimate_coefficients(X, t, n, ordered_monomials, solver=solver, tol=tol)
    # Solve parameters from the noisy data
    print('Driving noise ')
    estimate_coefficients(X_, t, n, ordered_monomials, solver=solver, tol=tol)


