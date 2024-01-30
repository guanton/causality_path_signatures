import random
from itertools import product, cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from formatting_helpers import *
matplotlib.use('TkAgg')


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
    print(f'There are {term_order} monomials up to degree {p} over {n} variables')
    return ordered_monomials

def generate_causal_graph(m, specified_edges):
    """
    Generates the graph structure for the temporal data that will be created
    :param m: number of causal variables x_i in the graph
    :param specified_edges: list of edges in the graph (pairs of vertices)
    :return:
    pa_dict: a dictionary where the keys are vertices and values are its parent vertices
    """
    # Create an empty dictionary to store the graph structure.
    pa_dict = {v: [] for v in range(m)}
    for edge in specified_edges:
        v1, v2 = edge
        pa_dict[v2].append(v1)
    return pa_dict

def check_monomial(term, pas):
    """
    Helper function that checks whether all factors in a monomial term correspond to the parents in the causal graph
    :param term: monomial represented by a list
    :param pas: list of parents for a given vertex
    :return:
    """
    vertices_in_monomial = np.nonzero(term)[0]
    for v in vertices_in_monomial:
        if v not in pas:
            return False
    return True


def monomial_string_to_array(monomial_str, m):
    """
    Convert a monomial string to the corresponding array
    :param monomial_str: String representation of a monomial, e.g., '5x_0x_1'
    :param m: Number of causal variables
    :return: Tuple representation of the monomial (coefficient, array)
    """
    # Find the first occurrence of 'x_'
    index_x = monomial_str.find('x_')

    # Extracting the coefficient part
    coefficient_str = monomial_str[:index_x] if index_x >= 0 else monomial_str

    # Extracting the monomial part
    monomial_part = monomial_str[index_x:] if index_x >= 0 else ''

    # Convert coefficient to float
    if coefficient_str == '-':
        coefficient = -1.0
    elif coefficient_str is None:
        coefficient = 1.0
    else:
        coefficient = float(coefficient_str)

    # Creating the array representation
    monomial_array = [0] * m
    i = 0
    while i < len(monomial_part):
        if monomial_part[i] == 'x':
            i += 2  # Skip 'x_'
            digit = int(monomial_part[i])
            assert digit < m, "included variable is outside specified range"
            if i + 1 < len(monomial_part):
                if monomial_part[i+1] == '^':
                    power = int(monomial_part[i+2])
                else:
                    power = 1
            else:
                power = 1
            # print(f'power for x_{digit}: {power}')
            monomial_array[digit] = power  # Adjust index to match 0-based indexing
        else:
            i += 1

    return coefficient, monomial_array


def parse_polynomial_strings(list_poly_strings, pa_dict):
    '''
    :param list_poly_strings: list of m poly strings (ex. of string: "x_1^2x_3 + x_2")
    :param pa_dict: a dictionary where the keys are vertices (int) and values are its parent vertices (int)
    :return: causal_params: a dictionary where the keys are vertices and values are list of tuples (coeff, monomial)
    corresponding to the terms associated to dx_i/dt
    '''
    causal_params = {}
    assert len(list_poly_strings) == len(pa_dict), "number of variables is not consistent between list_poly_strings and pa_dict"
    m = len(list_poly_strings)
    for i in range(m):
        coefficients_i = []
        monomials_i = []
        pas = pa_dict[i]  # extract the parents of the current vertex x_i
        poly_string = list_poly_strings[i]  # extract the polynomial string associated to dx_i/dt
        monomial_strings = re.split(r'\s*\+\s*', poly_string)  # extract the individual monomials
        monomial_strings = [term.strip() for term in monomial_strings]  # remove white space
        for monomial_string in monomial_strings:
            coefficient, monomial_array = monomial_string_to_array(monomial_string, m)
            if check_monomial(monomial_array, pas):
                coefficients_i.append(coefficient)
                monomials_i.append(monomial_array)
        causal_params[i] = [(coeff, monomial) for coeff, monomial in zip(coefficients_i, monomials_i)]
    return causal_params

def initialize_time_series(X, t, m, n_series = 1, zero_init = True, n_seed = 0):
    '''
    :param X:
    :param t:
    :return:
    '''
    if n_seed is not None:
        random.seed(n_seed)
    # set initial values
    for i in range(m):
        for copy in range(n_series):
            if zero_init:
                X.loc[t[0], (i, copy)] = 0
            else:
                X.loc[t[0], (i, copy)] = random.uniform(-1, 1)
    return X


def initialize_noise_series(driving_noise_scale, measurement_noise_scale, m, n_series, t, n_seed=0):
    W = pd.DataFrame(index=t)
    E = pd.DataFrame(index=t)

    if n_seed is not None:
        np.random.seed(n_seed)

    n_steps = len(t) - 1
    n_meas = len(t)
    if driving_noise_scale > 0:
        # Initialize a multi-index DataFrame for W
        mi_W = pd.MultiIndex.from_product([range(m), range(n_series)], names=['variable', 'copy'])
        W = pd.DataFrame(index=t, columns=mi_W)

        # Simulate Brownian motion noise for each variable x_i and copy
        for i in range(m):
            for copy in range(n_series):
                W_i_copy = np.random.randn(n_steps) * driving_noise_scale * np.sqrt(np.diff(t))
                W.loc[:, (i, copy)] = np.insert(W_i_copy, 0, 0)

    if measurement_noise_scale > 0:
        # Initialize a multi-index DataFrame for E
        mi_E = pd.MultiIndex.from_product([range(m), range(n_series)], names=['variable', 'copy'])
        E = pd.DataFrame(index=t, columns=mi_E)

        # Simulate measurement noise for each variable x_i and copy
        for i in range(m):
            for copy in range(n_series):
                E_i_copy = np.random.randn(n_meas) * measurement_noise_scale
                E.loc[:, (i, copy)] = E_i_copy

    return W, E


def calculate_polynomial_value(X, m, coeff_monomials, t_value, copy):
    """
    Calculate the polynomial value for a variable based on its coefficients and monomials.

    Parameters:
    - coeff_monomials: List of tuples, where each tuple contains a float coefficient and a monomial (in array form).
    - prev_values: Pandas Series representing the previous values of variables.

    Returns:
    - Polynomial value for the current time step.
    """
    polynomial_value = 0
    for coeff, monomial in coeff_monomials:
        monomial_value = coeff
        relevant_variables = [v for v in range(m) if monomial[v] != 0]
        degrees = [monomial[v] for v in relevant_variables]
        for v, degree in zip(relevant_variables, degrees):
            monomial_value *= X.loc[t_value, (v, copy)] ** degree
        polynomial_value += monomial_value
    return polynomial_value

def generate_temporal_data(causal_params, m, t, driving_noise_scale=0, measurement_noise_scale=0, n_series=1,
                           zero_init=True, n_seed = 0):
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
    m = len(causal_params.keys())
    mi = pd.MultiIndex.from_product([range(m), range(n_series)], names=['variable', 'copy'])
    X = pd.DataFrame(index=t, columns=mi)
    X = initialize_time_series(X, t, m, n_series=n_series, zero_init=zero_init, n_seed=0)
    # create noise time series
    W, E = initialize_noise_series(driving_noise_scale, measurement_noise_scale, m, n_series, t, n_seed)

    for step in range(1, n_steps):
        delta_t = t[step] - t[step - 1]
        # Iterate over each variable x_i
        for i in range(m):
            # Extract coefficients and monomials for variable x_i
            coeff_monomials = causal_params[i]
            # Update the variable's value for each copy at the current time step
            for copy in range(n_series):
                # Calculate the polynomial value for the current time step
                polynomial_value = calculate_polynomial_value(X, m, coeff_monomials, t[step-1], copy)
                X.loc[t[step], (i, copy)] = X.loc[t[step - 1], (i, copy)] + polynomial_value * delta_t
                if driving_noise_scale > 0:
                    X.loc[t[step], (i, copy)] += W.loc[t[step - 1], (i, copy)]

    if measurement_noise_scale > 0:
            X += E.values
    return X


def calculate_derivative_df(X, t, m, n_series):
    delta_t = np.diff(t)  # Calculate time differences
    df_list = []  # List to store individual DataFrames
    for i in range(m):
        for copy in range(n_series):
            series_to_differentiate = X.loc[:, (i, copy)].to_numpy()

            # Calculate derivative using finite differences
            derivative_values = np.diff(series_to_differentiate) / delta_t
            derivative_values = np.insert(derivative_values, 0, 0)  # Insert slope 0 for the first point

            # Create a MultiIndex for derivative_df
            mi = pd.MultiIndex.from_product([[i], [copy]], names=['variable', 'copy'])

            # Create a DataFrame with the correct structure
            temp_df = pd.DataFrame(derivative_values, index=t[:], columns=mi)
            df_list.append(temp_df)

    # Concatenate all DataFrames in the list
    derivative_df = pd.concat(df_list, axis=1)

    # Fill NaN values with 0
    derivative_df = derivative_df.fillna(0)

    return derivative_df


def plot_time_series(X, m, n_series, causal_params=None):
    # Extract the time series data for each variable
    variable_names = [f'x{i}' for i in range(0, m)]  # Variable names x1, x2, ..., xn

    # Plot each variable
    plt.figure(figsize=(10, 6))
    time_values = X.index.to_numpy()  # Convert Index to NumPy array

    for i in range(m):
        color = plt.gca()._get_lines.get_next_color()  # Get the next color from the default color cycle
        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)
        # Plot time series data for ith variable
        for copy in range(n_series):
            series_to_plot = X.loc[:, (i, copy)].to_numpy()
            plt.plot(time_values, series_to_plot, label=f'{variable_names[i]} - Copy {copy}', linestyle=next(linecycler),
                     color=color)

        # Display causal relationships if available
        if causal_params is not None and i in causal_params:
            terms = causal_params[i]
            causal_str = f'$dx_{{{i}}}$' + f'/dt = {rhs_as_sum(terms)}'

            # Calculate the position for the text box near the curve
            x_position = time_values[-1] - 0.1  # Slightly to the left of the end of the curve
            y_position = X.loc[X.index[-1], (i, 0)]

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



def plot_time_series_comp(X_list, labels, m, n_series, causal_params_list=None):
    '''
    :param t:
    :param X_list: [X given no noise, observed noisy X, recovered X using polynomial relations from noisy)
    :param labels:
    :param m:
    :param causal_params_list:
    :return:
    '''
    # Extract the time series data for each variable
    variable_names = [f'x{i}' for i in range(0, m)]  # Variable names x1, x2, ..., xn
    X = X_list[0]
    time_values = X.index.to_numpy()
    # Plot each variable for the original data (X)
    plt.figure(figsize=(12, 6))
    # Plot each variable for the original data (X)
    for i in range(m):
        color = plt.gca()._get_lines.get_next_color()  # Get the next color from the default color cycle
        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)
        for j in range(len(X_list)):
            X = X_list[j]
            series_to_plot =  X.loc[:, (i, slice(None))].mean(axis=1).to_numpy()
            plt.plot(time_values, series_to_plot, label=f'{variable_names[i]} ({labels[j]}) - mean over {n_series} copies',
                     linestyle=next(linecycler),
                     color=color)
        # Display causal relationships if available
        if causal_params_list is not None:
            for idx in range(len(causal_params_list)):
                causal_params = causal_params_list[idx]
                if i in causal_params:
                    terms = causal_params[i]
                    causal_str = f'$dx_{{{i}}}$' + f'/dt = {rhs_as_sum(terms)}'
                    x_position = time_values[-1] - 0.1  # Slightly to the left of the end of the curve
                    y_position = X.loc[X.index[-1], (i, 0)] + 0.2*idx
                    # Define the text box properties
                    textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
                    # Create a multiline text box
                    plt.text(x_position, y_position, causal_str, fontsize=10, ha='right', va='top', bbox=textbox_props)

                #
                # for idx in range(len(causal_str_list)):
                #     causal_str = causal_str_list[idx]
                #     # Calculate the position for the text box near the curve
                #     x_position = time_values[-1] - idx  # Slightly to the left of the end of the curve
                #     X = X_list[idx]
                #     y_position = X.loc[X.index[idx], (i, 0)]
                # y_position = X_list[2][-1, i]  # Assuming the third X is the recovered one



    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data with Causal Relationships')

    # Add a legend to label each variable
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # coeff, monomial_array = monomial_string_to_array('-1', 2)
    # print(coeff)
    m=2
    n_series = 3
    pa_dict = generate_causal_graph(m, [(0,0), (1, 0)])
    list_poly_strings = ['3x_0x_1^2 + -7x_1 +5', '-0.2']
    # i = 0
    # for p in list_poly_strings:
    #     print(f'dx_{i}/dt= {p}')
    #     i += 1
    causal_params = parse_polynomial_strings(list_poly_strings, pa_dict)
    print_causal_relationships(causal_params)
    t = np.linspace(0, 1, 500)
    X = generate_temporal_data(causal_params, m, t, driving_noise_scale=0.1, measurement_noise_scale=0, n_series=n_series,
                           zero_init=True, n_seed=0)
    print(X.head())
    derivative_df = calculate_derivative_df(X, t, m, n_series)

    print(derivative_df.head())
    plot_time_series_comp([X], ['raw'], m, n_series, causal_params=causal_params)


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

# def generate_polynomial_relations(pa_dict, ordered_monomials, monomial_density=None, specified_coeffs=None, n_seed=None,
#                                   ensure_constant=True):
#     """
#     :param pa_dict: a dictionary where the keys are vertices and values are its parent vertices
#     :param ordered_monomials: a dictionary that orders each monomial of degree 0 to p. The keys represent the index (0, 1, ...
#     n_monomials-1) and the value is the corresponding monomial, represented by a list.
#     :param monomial_density:
#     :param specified_coeffs:
#     :param n_seed:
#     :param ensure_constant:
#     :return: causal_params: a dictionary where the keys are vertices and values are list of tuples (coeff, monomial)
#     corresponding to the terms associated to dx_i/dt
#     """
#     if n_seed is not None:
#         random.seed(n_seed)
#     n = len(pa_dict.keys())
#     all_monomials = list(ordered_monomials.values())
#     causal_params = {}
#     # iterate over each vertex in causal graph
#     for i in range(n):
#         pas = pa_dict[i]
#         # select the terms in the polynomial for dx_i/dt based on nbrs of i in causal graph
#         valid_monomials_i = list(filter(lambda term: check_monomial(term, pas), all_monomials))
#         if monomial_density is not None:
#             # drop a proportion of monomials if applicable
#             monomials_i = random.sample(valid_monomials_i, int(monomial_density * len(valid_monomials_i)))
#             n_monomials_i = len(monomials_i)
#         else:
#             monomials_i = valid_monomials_i
#             n_monomials_i = len(valid_monomials_i)
#         if specified_coeffs is None:
#             # sample coefficients for each monomial
#             coefficients = [random.uniform(-1, 1) for _ in range(n_monomials_i)]
#             causal_params[i] = [(coeff, term) for coeff, term in zip(coefficients, monomials_i)]
#         else:
#             assert len(specified_coeffs[i]) == len(
#                 all_monomials), f"the number of specified coefficients {len(specified_coeffs)}" \
#                                 f"does not match the number of monomials {len(all_monomials)}"
#             coefficients = specified_coeffs[i]
#             causal_params[i] = [(coeff, term) for coeff, term in zip(coefficients, monomials_i)]
#     return causal_params

# def generate_temporal_data_from_fns(n, temporal_functions, t_min=0, t_max=1, n_steps=100, noise_addition=False,
#                                     n_series=1):
#     """
#     :param n: Number of causal variables
#     :param temporal_functions: list of functions where ith function determines variable i
#     :param t_min: Minimum time
#     :param t_max: Maximum time
#     :param n_steps: Number of time steps
#     :param noise_addition: Whether to add noise
#     :param n_series: Number of series to generate
#     :return: Generated time series data
#     """
#     assert n == len(temporal_functions), "number of variables does not match number of functions"
#     t = np.linspace(t_min, t_max, n_steps)
#     X = np.zeros((n_steps, n))
#     # set initial values
#     if noise_addition:
#         W = []
#         for i in range(n):
#             X[0, i] = random.uniform(-1, 1)
#             # Simulate Brownian motion (W(s))
#             W.append(np.sqrt(t[1] - t[0]) * np.random.randn(n_steps))
#     for step in range(n_steps):
#         for i in range(n):
#             # Calculate the value for each x_i(t) based on functions
#             X[step, i] = temporal_functions[i](t[step])
#             if noise_addition:
#                 X[step, i] += W[i][step]
#     return X