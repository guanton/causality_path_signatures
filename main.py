from generate_temporal_data import *
from signature_matching_polynomial_relations import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for maximal degree and number of variables
    p = 1
    n = 2

    # Generate the complete degree dictionary
    n_monomials, degree_dict, ordered_monomials = generate_monomials(p, n)
    print(ordered_monomials)
    print(f'total number of possible monomials up to degree {p} over {n} variables:', n_monomials)
    nbrs_dict = generate_causal_graph(n, specified_edges = [(0,1)])

    # Generate causal parameters based on the degree dictionary
    noise_addition = False
    causal_params = generate_polynomial_relations(nbrs_dict, ordered_monomials)
    print('actual polynomial relations')
    for i in range(n):
        # Display causal relationships if available
        if causal_params is not None and i in causal_params:
            terms = causal_params[i].items()
            causal_str = f'dx_{i}' + f'/dt = {rhs_as_sum(terms, latex=False)}'
            if noise_addition:
                causal_str += f'+ W^{i}'
        print(causal_str)

    # Define the time points for each time series
    n_steps = 100
    t = np.linspace(0, 1, n_steps)  # Increase the number of time points for smoother data
    # genreate the data that is compatible with the specified causal relationships
    X = generate_temporal_data(causal_params, t, noise_addition= noise_addition)
    # # Define the input functions for each x_i(t)
    # def function_x0(t):
    #     return t + 3
    #
    #
    # def function_x1(t):
    #     return (t + 3)**2
    #
    #
    # functions = [function_x0, function_x1]
    # X = generate_temporal_data_from_fns(n, functions)

    # Plot the time series data
    plot_time_series(t, X, n, causal_params)

    '''
    Solving with signature matching
    '''
    # generate the subintervals that will be used to compute the level-1 path signatures on different windows
    subintervals = generate_subintervals(t, n_monomials)
    M = compute_M(X, n_monomials, ordered_monomials, subintervals, t, n)
    solver = 'direct'
    tol = 0.1
    print(f'Recovered relations from signature matching using {solver} solver and cutoff {tol}:')
    solve_parameters(X, n_monomials, subintervals, M, ordered_monomials, solver = solver)


    # debugging
    # # Example usage to access time series data:
    # for i in range(n):
    #     variable_data = X[:, i]
    #     print(f"Time Series Data for x{i + 1}:", variable_data)
