from generate_temporal_data import *
from solve_polynomial_relations import *
from math import comb
import pandas as pd

def generate_time_series(list_poly_strings, p, m, specified_edges, driving_noise_scale, measurement_noise_scale, n_steps, n_series = 1, specified_coeffs = None, monomial_density = None, start_t=0, end_t=1, n_seed = None ):
    '''
    :param list_poly_strings:
    :param p: degree of polynomial relations for SDEs
    :param m: number of causal variables
    :param specified_edges: list of edges (list of pairs)
    :param driving_noise_scale: scale of the driving noise
    :param measurement_noise_scale: scale of the measurement noise
    :param n_steps: number of observations per time series
    :param monomial_density: proportion of monomials used for causal polynomial relations
    :param specified_coeffs: optional parameter for
    :return:
    '''
    # Generate all possible monomials and order them
    ordered_monomials = generate_monomials(p, m)
    # create the causal graph based on the specified edges
    pa_dict = generate_causal_graph(m, specified_edges=specified_edges)
    # Generate causal parameters
    causal_params = parse_polynomial_strings(list_poly_strings, pa_dict)
    # Define the time points for each time series
    t = np.linspace(start_t, end_t, n_steps)
    X = generate_temporal_data(causal_params, m, t, driving_noise_scale=driving_noise_scale, measurement_noise_scale=measurement_noise_scale, n_series=n_series,
                               zero_init=True, n_seed=n_seed)
    derivative_df = calculate_derivative_df(X, t, m, n_series)
    return X, t, causal_params, ordered_monomials, derivative_df


def estimate_coefficients(X, m, derivative_df, ordered_monomials, k, copies, solver = 'direct', tol = 0.1, alpha = None, n_series = 1, n_seed = 0, index_place = 1, sub_mode = 'one'):
    '''
    :param X:
    :param m:
    :param derivative_df:
    :param ordered_monomials:
    :param k: max word length
    :param copies:
    :param solver:
    :param tol:
    :param alpha:
    :param n_series:
    :param n_seed:
    :param index_place:
    :return:
    '''
    subs = generate_subintervals(t, sub_mode)
    n_params = len(ordered_monomials)
    # create the necessary words for each variable
    words_per_variable = []
    interest_indices_per_variable = []
    for l in range(m):
        words_for_l, interest_indices_l = generate_all_words(l, m, index_place, k)
        words_per_variable.append(words_for_l)
        interest_indices_per_variable.append(interest_indices_l)
    # print('words_per_variable:', words_per_variable)
    # print('interest_indices:', interest_indices_per_variable)
    recovered_causal_params = solve_parameters(X, m, derivative_df, words_per_variable, interest_indices_per_variable, ordered_monomials, subs, copies, alpha=alpha, tol = tol, solver = solver)
    return recovered_causal_params
def estimate_coefficients_sub(X, t, n, ordered_monomials, solver = 'direct', tol = 0.1, alpha = None, sub_mode = 'zeros',
                          n_subs = None, sample_noise = True, level = 1, driving_noise_scale = 0.1, n_samples = 1):
    '''
    :param X: observed time series data
    :param t: set of time indices
    :param n: number of causal variables x_i
    :param ordered_monomials: set of considered monomials for polynomial relationships
    :param solver: 'lasso', 'ridge', 'direct', 'pseudo-inverse'
    :param tol: cutoff parameter for reporting monomial coefficients
    :param alpha: regularization parameter
    :param sub_mode: 'all', 'random', 'zeros', 'adjacent'
    :param n_subs: optional parameter to be used for 'random' subintervals
    :param level: level of iterated integrals for signature matching
    :return:
    '''
    n_monomials = len(ordered_monomials)
    # create the necessary subintervals
    subintervals = generate_subintervals(t, sub_mode)
    sub_dict = create_subintervals_dict(subintervals, t)
    M = compute_M_subs(X, n_monomials, ordered_monomials, subintervals, t, n, level, driving_noise_scale=driving_noise_scale, sub_dict = sub_dict, sample_noise = sample_noise)
    print(f'Recovered relations from level {level} signature matching using {solver} solver with alpha = {alpha} and cutoff {tol}:')
    recovered_causal_params = solve_parameters_sub(X, t, n_monomials, subintervals, M, ordered_monomials, solver=solver, alpha = alpha, level = level)
    return recovered_causal_params

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for creating the data
    m = 2 # number of causal variables
    p = 2 # polynomial degree considered
    n_steps = 100  # number of time points per variable
    n_series = 1
    specified_edges = [(0,0), (1,0), (1,1)] # list of edges in the causal graph
    list_poly_strings = ['5x_1x_0 -1x_0 + 2x_0+0.3', '-0.2 + -1x_1']
    start_t = 0
    end_t = 1
    n_seed = 0
    print('Seed:', n_seed)
    # Generate the original data
    X, t, causal_params, ordered_monomials, derivative_df = generate_time_series(list_poly_strings, p, m, specified_edges, driving_noise_scale=0, measurement_noise_scale=0, n_steps=n_steps, n_series = n_series, start_t=start_t, end_t=end_t, n_seed = n_seed )
    print('Actual polynomial relationships')
    print_causal_relationships(causal_params)
    measurement_noise_scale = 0
    driving_noise_scale = 0.1
    # Generate the noisy data
    X_, t, causal_params, ordered_monomials, derivative_df_ = generate_time_series(list_poly_strings, p, m, specified_edges, driving_noise_scale=driving_noise_scale, measurement_noise_scale=measurement_noise_scale, n_steps=n_steps, n_series = n_series, start_t=start_t, end_t=end_t, n_seed = n_seed)
    # x_i = X.loc[:, (0, 0)].to_numpy()
    # print(x_i)
    # print(derivative_df_)
    # mi = (0, 0)
    # derivatives_array = derivative_df_.loc[:5, mi].to_numpy()
    # print(derivatives_array)
    # print(derivatives_array[5])
    W = []
    # Choose parameters for solving coefficients
    solver = 'lasso'
    alpha = 0.0001
    tol = 0.01
    n_subs = None
    method = 'integrals'
    # # Solve parameters from the original data
    # print('From the original noiseless data: ')
    # estimate_coefficients(X, t, n, ordered_monomials, solver=solver, tol=tol, alpha = alpha, sub_mode = sub_mode, n_subs = n_subs)
    # Solve parameters from the noisy data
    print(f'Approximated from the noisy data with method {method}: ')
    if method == 'integrals':
        k = 1
        copies = np.arange(n_series).tolist()
        index_place = 0
        sub_mode = 'zeros'
        recovered_causal_params_ = estimate_coefficients(X_, m, derivative_df_, ordered_monomials, k, copies, solver = solver, tol = tol, alpha = alpha, n_series = n_series, n_seed = n_seed, index_place = index_place, sub_mode=sub_mode)
        X_recovered_ = generate_temporal_data(recovered_causal_params_, m, t, driving_noise_scale=0, measurement_noise_scale=0, n_series=1,
                               zero_init=True, n_seed = 0)
    elif method == 'subs':
        sub_mode = 'zeros'
        n_series = 1
        X_ = convert_df_to_array(X_)
        recovered_causal_params_ = estimate_coefficients_sub(X_, t, m, ordered_monomials, solver=solver, sample_noise=False,
                                                         tol=tol, alpha=alpha,
                                                         sub_mode=sub_mode, n_subs=n_subs, level=1)
        X_recovered_ = generate_temporal_data(recovered_causal_params_, m, t, driving_noise_scale=0,
                                              measurement_noise_scale=0, n_series=1,
                                              zero_init=True, n_seed=0)
        X_ = convert_array_to_df(X_, t, m)
    # Plot the time series data
    plot_time_series_comp([X_, X_recovered_], ['Original', 'Recovered'], m, n_series, causal_params_list = [causal_params, recovered_causal_params_])


    # plot_time_series_comp(t, X_, X_recovered, n, causal_params)
    # plot recovered version

