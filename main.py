from generate_temporal_data import *
from signature_matching_polynomial_relations import *


def generate_time_series(p, n, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, specified_coeffs = None, monomial_density = None, start_t=0, end_t=1):
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
    t = np.linspace(start_t, end_t, n_steps)
    # generate the data that is compatible with the specified causal relationships
    X = generate_temporal_data(causal_params, t, driving_noise_scale=driving_noise_scale, measurement_noise_scale=measurement_noise_scale)
    return X, t, causal_params, ordered_monomials

def estimate_coefficients(X, t, n, ordered_monomials, solver = 'direct', tol = 0.1, alpha = None, sub_mode = 'zeros', n_subs = None, level = 1):
    n_monomials = len(ordered_monomials)
    # create the necessary subintervals
    if sub_mode == 'random':
        if n_subs is None:
            n_subs = 2*len(t)
        subintervals = generate_random_subintervals(t, n_subs)
    elif sub_mode == 'all':
        subintervals = generate_all_subintervals(t)
    elif sub_mode == 'zeros':
        subintervals = generate_all_subintervals_0(t)
    elif sub_mode == 'n_params':
        subintervals = generate_subintervals_n_params(t, n_monomials)
    M = compute_M(X, n_monomials, ordered_monomials, subintervals, t, n, level)
    print(f'Recovered relations from level {level} signature matching using {solver} solver and cutoff {tol}:')
    recovered_causal_params = solve_parameters(X, t, n_monomials, subintervals, M, ordered_monomials, solver=solver, alpha = alpha, level = level)
    return recovered_causal_params

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for creating the data
    p = 1 # polynomial degree considered
    n = 2 # number of causal variables
    n_steps = 100  # number of time points per variable
    specified_edges = [(0,1)] # list of edges in the causal graph
    monomial_density = None# None assumes that all monomials (valid w.r.t graph) are included
    start_t = 0
    end_t = 1
    driving_noise_scale = 0
    measurement_noise_scale = 0
    # Generate the original data
    X, t, causal_params, ordered_monomials = generate_time_series(p, n, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, monomial_density=monomial_density, start_t=start_t, end_t=end_t)
    print('Actual polynomial relationships')
    print_causal_relationships(causal_params)
    # Noisy version
    driving_noise_scale = 1
    measurement_noise_scale = 0
    # Generate the noisy data
    X_, t, causal_params, ordered_monomials = generate_time_series(p, n, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, monomial_density=monomial_density, start_t=start_t, end_t=end_t)
    W = []
    # Choose parameters for solving coefficients
    solver = 'lasso' # implement hierarchical lasso later
    alpha = 0.01 # regularization
    level = 2 # level of signature matching
    tol = 0.1
    n_subs = None
    sub_mode = 'zeros'
    # Solve parameters from the original data
    print('From the original noiseless data: ')
    estimate_coefficients(X, t, n, ordered_monomials, solver=solver, tol=tol, alpha = alpha, sub_mode = sub_mode, n_subs = n_subs)
    # Solve parameters from the noisy data
    print(f'From the noisy data: ')
    # for i in range(10, 101, 10):
    #     estimate_coefficients(X_, t, n, ordered_monomials, solver=solver, tol=tol, alpha = alpha, sub_mode = 'random', n_subs = i)
    # plot recovered data
    recovered_causal_params = estimate_coefficients(X_, t, n, ordered_monomials, solver=solver, tol=tol, alpha=alpha, sub_mode = sub_mode, n_subs = n_subs, level = level)
    driving_noise_scale = 0
    measurement_noise_scale = 0
    X_recovered = generate_temporal_data(recovered_causal_params, t, driving_noise_scale, measurement_noise_scale)
    # Plot the time series data
    plot_time_series_comp(t, [X, X_, X_recovered], ['Original', 'Noisy', 'Recovered'], n,
                         causal_params=causal_params)

    # plot_time_series_comp(t, X_, X_recovered, n, causal_params)
    # plot recovered version

