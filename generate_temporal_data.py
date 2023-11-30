import numpy as np
import random
from itertools import product, combinations
import matplotlib.pyplot as plt


def generate_monomials(p, n):
    """
    This function returns all possible monomial relations on (x_1, ..., x_n) up to degree p
    :param p: maximal degree p considered for polynomial relationships dx_i/dt=p(X)
    :param n: number of causal variables x_i
    :return:
    n_monomials: number of parameters (coefficients for monomials) that we will need to recover using path signatures
    ordered_monomials: a dictionary that orders each monomial of degree 0 to p. The keys represent the index (0, 1, ...
    n_monomials-1) and the value is the corresponding monomial, represented by a list.
    """
    ordered_monomials = {}
    term_order = 0
    for k in range(p + 1):
        # use itertools.product to generate all possible combinations of n values, where each value is in [0,k]]
        for combo in product(range(k + 1), repeat=n):
            if sum(combo) == k:
                term = list(combo)
                ordered_monomials[term_order] = term
                # Increment the order and count variables
                term_order += 1
    n_monomials = term_order
    return n_monomials, ordered_monomials


def generate_causal_graph(n, specified_edges = None, edge_density = None, include_self_edges = True, random_m = False):
    """
    Generates the graph structure for the temporal data that will be created
    :param n: number of causal variables x_i in the graph
    :param specified_edges (optional): list of edges (pairs of vertices)
    :param edge_density (optional): density of edges in the causal graph
    :return:
    nbrs_dict: a dictionary where the keys are vertices and values are the neighbouring vertices
    """
    all_edges = list(combinations(range(n), 2))
    if include_self_edges:
        all_edges += [(v, v) for v in range(n)]  # Include self-edges
    if specified_edges is None:
        if edge_density is None:
            if random_m:
                n_edges = random.randint(0, len(all_edges)-1)
            else:
                n_edges = len(all_edges)
        else:
            m = len(all_edges)
            n_edges = int(m*edge_density)
        edges = random.sample(all_edges, n_edges)
    else:
        edges = specified_edges
    # Create an empty dictionary to store the graph structure.
    nbrs_dict = {v: [] for v in range(n)}
    for edge in edges:
        # Add each edge to the neighbours of both vertices.
        v1, v2 = edge
        nbrs_dict[v1].append(v2)
        nbrs_dict[v2].append(v1)
    return nbrs_dict


def check_monomial(term, nbrs):
    """
    Helper function that checks whether all factors in a monomial term correspond to neighbours in the causal graph
    :param term: monomial represented by a list
    :param nbrs: list of vertices (neighbours for some vertex)
    :return:
    """
    vertices_in_monomial = np.nonzero(term)[0]
    for v in vertices_in_monomial:
        if v not in nbrs:
            return False
    return True

def generate_polynomial_relations(nbrs_dict, ordered_monomials, monomial_density = None, specified_coeffs = None, n_seed = None, ensure_constant = True):
    """
    :param nbrs_dict: a dictionary where the keys are vertices and values are the neighbouring vertices
    :param ordered_monomials: a dictionary that orders each monomial of degree 0 to p. The keys represent the index (0, 1, ...
    n_monomials-1) and the value is the corresponding monomial, represented by a list.
    :param monomial_density:
    :param specified_coeffs:
    :param n_seed:
    :param ensure_constant:
    :return:
    causal_params: a two-layered dictionary where the key i represents dx_i/dt for variable x_i and the value is another
    dictionary, which captures all polynomial terms for dx_i/dt. The keys of this dictionary are the coefficients and
    the values are the corresponding terms (represented as n-arrays)
    """
    if n_seed is not None:
        random.seed(n_seed)
        print(f'Set seed to {n_seed}')
    n = len(nbrs_dict.keys())
    all_monomials = list(ordered_monomials.values())
    causal_params = {}
    # iterate over each vertex in causal graph
    for i in range(n):
        nbrs = nbrs_dict[i]
        # select the terms in the polynomial for dx_i/dt based on nbrs of i in causal graph
        valid_monomials_i = list(filter(lambda term: check_monomial(term, nbrs), all_monomials))
        if monomial_density is not None:
            # drop a proportion of monomials if applicable
            monomials_i = random.sample(valid_monomials_i, int(monomial_density*len(valid_monomials_i)))
            n_monomials_i = len(monomials_i)
        else:
            monomials_i = valid_monomials_i
            n_monomials_i = len(valid_monomials_i)
        if specified_coeffs is None:
            # sample coefficients for each monomial
            coefficients = [random.uniform(-5, 5) for _ in range(n_monomials_i)]
            causal_params[i] = {coeff: term for coeff, term in zip(coefficients, monomials_i)}
        else:
            assert len(specified_coeffs[i]) == len(all_monomials), f"the number of specified coefficients {len(specified_coeffs)}" \
                                                              f"does not match the number of monomials {len(all_monomials)}"
            coefficients = specified_coeffs[i]
            causal_params[i] = {coeff: term for coeff, term in zip(coefficients, monomials_i)}
    return causal_params


def generate_temporal_data(causal_params, t, driving_noise_scale = 0, measurement_noise_scale = 0, n_series = 1, zero_init=True):
    """
    :param causal_params:
    :param t_min:
    :param t_max:
    :param n_steps:
    :param noise_addition:
    :param n_series:
    :return:
    """
    n_steps = len(t)
    n = len(causal_params.keys())
    X = np.zeros((n_steps, n))
    # create noise time series
    W = None
    E = None
    if driving_noise_scale > 0:
        W = []
        for i in range(n):
            # Simulate Brownian motion noise for each variable x_i
            W_i = []
            for step in range(1, n_steps):
                W_i.append(np.random.randn() * driving_noise_scale* np.sqrt(t[step]-t[step-1]))
            W.append(W_i)
    if measurement_noise_scale > 0:
        E = []
        for i in range(n):
            E_i = []
            for step in range(0, n_steps):
                E_i.append(np.random.randn() * measurement_noise_scale)  # standard normal noise
            E.append(E_i)
    # set initial values
    if zero_init:
        for i in range(n):
            X[0, i] = 0
    else:
        for i in range(n):
            X[0, i] = random.uniform(-1, 1)
    # Iterate over each time step
    for step in range(1, n_steps):
        delta_t = t[step] - t[step - 1]
        # Iterate over each variable (i)
        for i in range(n):
            polynomial_dict = causal_params[i]
            # Initialize the polynomial value
            polynomial_value = 0
            # Iterate over the coefficients and degrees of monomials
            for coefficient, monomial in polynomial_dict.items():
                monomial_value = coefficient
                relevant_variables = [v for v in range(n) if monomial[v] != 0]
                degrees = [monomial[v] for v in relevant_variables]
                for v, degree in zip(relevant_variables, degrees):
                    monomial_value *= X[step - 1, v] ** degree
                polynomial_value += monomial_value
            # Update the variable's value for the current time step
            if driving_noise_scale > 0:
                X[step, i] = X[step - 1, i] + polynomial_value * delta_t + W[i][step-1]
            else:
                X[step, i] = X[step - 1, i] + polynomial_value * delta_t
    if measurement_noise_scale > 0:
        for i in range(n):
            for step in range(n_steps):
                X[step, i] += E[i][step]
    return X

def generate_temporal_data_from_fns(n, temporal_functions, t_min=0, t_max=1, n_steps=100, noise_addition=False, n_series=1):
    """
    :param n: Number of causal variables
    :param temporal_functions: list of functions where ith function determines variable i
    :param t_min: Minimum time
    :param t_max: Maximum time
    :param n_steps: Number of time steps
    :param noise_addition: Whether to add noise
    :param n_series: Number of series to generate
    :return: Generated time series data
    """
    assert n == len(temporal_functions), "number of variables does not match number of functions"
    t = np.linspace(t_min, t_max, n_steps)
    X = np.zeros((n_steps, n))
    # set initial values
    if noise_addition:
        W = []
        for i in range(n):
            X[0, i] = random.uniform(-1, 1)
            # Simulate Brownian motion (W(s))
            W.append(np.sqrt(t[1] - t[0]) * np.random.randn(n_steps))
    for step in range(n_steps):
        for i in range(n):
            # Calculate the value for each x_i(t) based on functions
            X[step, i] = temporal_functions[i](t[step])
            if noise_addition:
                X[step, i] += W[i][step]
    return X


def plot_time_series(t, X, n, causal_params=None):
    # Extract the time series data for each variable
    variable_names = [f'x{i}' for i in range(0, n)]  # Variable names x1, x2, ..., xn

    # Plot each variable
    plt.figure(figsize=(10, 6))

    for i in range(n):
        # Plot time series data for ith variable
        plt.plot(t, X[:, i], label=variable_names[i])

        # Display causal relationships if available
        if causal_params is not None and i in causal_params:
            terms = causal_params[i].items()
            causal_str = f'$dx_{{{i}}}$' + f'/dt = {rhs_as_sum(terms)}'

            # Calculate the position for the text box near the curve
            x_position = t[-1] - 0.1  # Slightly to the left of the end of the curve
            y_position = X[-1, i]

            # Define the text box properties
            textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)


            # Create a multiline text box
            plt.text(x_position, y_position, causal_str, fontsize=10, ha='right', va='top', bbox=textbox_props)

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data')

    # Add a legend to label each variable
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()



def plot_time_series_comp(t, X, X_, n, causal_params=None):
    # Extract the time series data for each variable
    variable_names = [f'x{i}' for i in range(0, n)]  # Variable names x1, x2, ..., xn

    # Plot each variable for the original data (X)
    plt.figure(figsize=(12, 6))
    # Plot each variable for the original data (X)
    for i in range(n):
        color = plt.gca()._get_lines.get_next_color()  # Get the next color from the default color cycle
        plt.plot(t, X[:, i], label=f'{variable_names[i]} (Original)', linestyle='-', color=color)
        plt.plot(t, X_[:, i], linestyle='--', label=f'{variable_names[i]} (Noisy)', color=color)
        # Display causal relationships if available
        if causal_params is not None and i in causal_params:
            terms = causal_params[i].items()
            causal_str = f'$dx_{{{i}}}$' + f'/dt = {rhs_as_sum(terms)}'

            # Calculate the position for the text box near the curve
            x_position = t[-1] - 0.1  # Slightly to the left of the end of the curve
            y_position = X_[-1, i]

            # Define the text box properties
            textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)

            # Create a multiline text box
            plt.text(x_position, y_position, causal_str, fontsize=10, ha='right', va='top', bbox=textbox_props)

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data with Causal Relationships')

    # Add a legend to label each variable
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()


'''
Helper functions for converting terms (as n-arrays) into strings
'''

# Function to convert term values into a sum representation
def rhs_as_sum(terms, latex = True):
    term_strings = []
    for coeff, term in terms:
        if all(term[j] == 0 for j in range(len(term))):
            term_strings.append(f'{coeff:.2f}')
        else:
            coeff_string = f'{coeff:.2f}'
            term_string = ''
            for j in range(len(term)):
                if term[j] > 0:
                    term_string += get_termstring(term, latex)#f'$x_{{{j}}}^{{{term[j]}}}$'
            term_strings.append(coeff_string + term_string)
    if len(terms) > 0:
        polynomial_str = ' + '.join(term_strings)
        return polynomial_str
    else:
        return '0'

def get_termstring(term, latex = False):
    term_string = ''
    for j in range(len(term)):
        if term[j] > 0:
            if latex:
                term_string += f'$x_{{{j}}}^{{{term[j]}}}$'
            else:
                term_string += f'x_{j}^{term[j]}'
    if all(term[j] == 0 for j in range(len(term))):
        term_string += '1'
    return term_string

def print_causal_relationships(causal_params):
    n = len(causal_params.keys())
    for i in range(n):
        # Display causal relationships if available
        if causal_params is not None and i in causal_params:
            terms = causal_params[i].items()
            causal_str = f'dx_{i}' + f'/dt = {rhs_as_sum(terms, latex=False)}'
        print(causal_str)

#
# # Function to scale variables to prevent large jumps
# def scale_variables(X, scaling_factors):
#     return X / scaling_factors
#
# # Function to check if any value in X exceeds a threshold (e.g., 10^5)
# def exceeds_threshold(X, threshold):
#     return any(np.abs(X) > threshold)
#
# def generate_data(max_reinitialization_attempts, t, causal_params):
#     # Attempt to find suitable initial conditions
#     reinitialization_attempts = 0
#     while reinitialization_attempts < max_reinitialization_attempts:
#         X0 = np.random.uniform(-1, 1, n)  # Initialize X0 with random values between -1 and 1
#
#         # Solve the system of ODEs with scaled variables
#         X_scaled = scale_variables(X0, np.max(X0))  # Scale the initial conditions
#         X_smooth = odeint(system_of_odes, X_scaled, t, args=(causal_params,), rtol=1e-8, atol=1e-8)
#
#         # Reverse the scaling to obtain realistic data
#         X_smooth = scale_variables(X_smooth, np.max(X0))
#
#         # Solve the system of ODEs with the original initial conditions
#         X = odeint(system_of_odes, X0, t, args=(causal_params,), rtol=1e-8, atol=1e-8)
#         #
#         # # If the result does not exceed the threshold, break the loop
#         # for i in range(X.shape[1]):  # Assuming X is a 2D NumPy array
#         #     if np.any(X[:, i] > threshold):
#         #         break
#         # If the result does not exceed the threshold, break the loop
#         if not exceeds_threshold(X[-1], threshold):
#             break
#         reinitialization_attempts += 1
#         if reinitialization_attempts == max_reinitialization_attempts:
#             print("Warning: Maximum reinitialization attempts reached.")
#     return X

