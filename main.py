from generate_temporal_data import *
from signature_matching_polynomial_relations import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for maximal degree and number of variables
    p = 4
    n = 2

    # Generate the complete degree dictionary
    n_monomials, degree_dict, ordered_monomials = generate_monomials(p, n)
    print(n_monomials)
    print("Degree Dictionary:")
    print(degree_dict)

    for d in degree_dict:
        print('Degree: ', d)
        for term in degree_dict[d]:
            print(get_termstring(term))
    print("Order Dictionary:")
    print(ordered_monomials)

    nbrs_dict = generate_causal_graph(n)

    # Generate causal parameters based on the degree dictionary
    causal_params = generate_polynomial_relations(nbrs_dict, ordered_monomials, monomial_density=0.1)
    print("Causal Parameters:")
    print(causal_params)

    # Define the time points for each time series
    n_steps = 100
    t = np.linspace(0, 1, n_steps)  # Increase the number of time points for smoother data
    # genreate the data that is compatible with the specified causal relationships
    X = generate_temporal_data(causal_params, n_steps = n_steps)
    # Plot the time series data
    plot_time_series(t, X, n, causal_params)

    # print('X_0:', X[: ,0])
    '''
    Solving with signature matching
    '''
    # generate the subintervals that will be used to compute the level-1 path signatures on different windows
    subintervals = generate_subintervals(t, n_monomials)
    M = solve_M(X, n_monomials, ordered_monomials, subintervals, t, n)
    print(M)
    solve_parameters(X, n_monomials, subintervals, M, ordered_monomials)



    # debugging
    # # Example usage to access time series data:
    # for i in range(n):
    #     variable_data = X[:, i]
    #     print(f"Time Series Data for x{i + 1}:", variable_data)
