import numpy as np
import random
from itertools import product
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def generate_complete_degree_dictionary(p, n):
    """
    This function returns all possible polynomial relations on (x_1, ..., x_n) up to degree p
    :param p: maximal degree p considered for polynomial relationships dx_i/dt=p(X)
    :param n: number of causal variables x_i
    :return:
    degree_dict: a dictionary where the keys represent degrees k=0, 1, ..., p, and values represent the list of all
    possible terms (combos of variables) whose total degree is k. Each combination is given by an n-tuple where entry i
    in the tuple corresponds to the degree of x_i within the degree k term (ex. [1,0,2] corresponds to x_0*x_2^2 for k=3
    """
    degree_dict = {}
    for k in range(p + 1):
        degree_dict[k] = []
        # use itertools.product to generate all possible combinations of n values, where each value is in [0,k]]
        for combo in product(range(k + 1), repeat=n):
            if sum(combo) == k:
                degree_dict[k].append(list(combo))
    return degree_dict


def count_params(degree_dict):
    """
    This function counts the total number of possible polynomial relations on (x_1, ..., x_n) up to degree p
    :param degree_dict: dictionary of all possible polynomial causal relations from generate_complete_degree_dictionary
    :return:
    n_params: number of parameters (coefficients) that we need to recover using path signatures
    """
    n_params = 0
    for k in degree_dict.keys():
        n_params += len(degree_dict[k])
    return n_params


def generate_causal_parameters(degree_dict, n):
    """
    This function creates random polynomial causal relations related to each dx_i/dt
    :param degree_dict: dictionary of all possible polynomial causal relations from generate_complete_degree_dictionary
    :param n: number of causal variables x_i
    :return:
    causal_params: a two-layered dictionary where the key i represents dx_i/dt for variable x_i and the value is another
    dictionary, which captures all polynomial terms for dx_i/dt. The keys of this dictionary are the coefficients and
    the values are the corresponding terms (represented as n-arrays)
    """
    causal_params = {}
    n_params = count_params(degree_dict)
    for i in range(n):
        # encourage sparser polynomial relations with probability weights
        weights = [1 / (p+1) for p in range(n_params)]
        # choose the number of terms in the polynomial for dx_i/dt
        n_params_i = random.choices(range(n_params), weights=weights)[0]
        all_terms = [term for k in degree_dict.keys() for term in degree_dict[k]]
        # select the terms in the polynomial for dx_i/dt
        selected_terms = random.sample(all_terms, n_params_i)
        # generate coefficients from random uniform
        coefficients = [random.uniform(-1, 1) for _ in range(n_params_i)]
        causal_params[i] = {coeff: term for coeff, term in zip(coefficients, selected_terms)}
    return causal_params


def system_of_odes(X, t, causal_params):
    '''
    :param X: list of n causal variables (for which we will generate time_series data)
    :param t: time indices
    :param causal_params: causal polynomial relationships for each dx_i/dt
    :return:
    '''
    n = len(X)
    dx_dt = [0.0] * n

    def calculate_term_value(term, X):
        '''
        Helper function that unpacks the term from the n-array representation
        :param term: n-array representation of term
        :param X: list of n causal variables
        :return: value of term
        '''
        value = 1.0
        for i in range(n):
            value *= X[i] ** term[i]
        return value

    for i in range(n):
        dx_dt[i] = sum(coeff * calculate_term_value(term, X) for coeff, term in causal_params[i].items())

    return dx_dt

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


# Function to convert term values into a sum representation
def rhs_as_sum(terms):
    print('terms:', terms)
    term_strings = []
    for coeff, term in terms:
        if all(term[j] == 0 for j in range(len(term))):
            term_strings.append(f'{coeff:.2f}')
        else:
            coeff_string = f'{coeff:.2f}'
            term_string = ''
            for j in range(len(term)):
                if term[j] > 0:
                    term_string += f'$x_{{{j}}}^{{{term[j]}}}$'
            term_strings.append(coeff_string + term_string)
    if len(terms) > 0:
        polynomial_str = ' + '.join(term_strings)
        return polynomial_str
    else:
        return '0'

# Function to scale variables to prevent large jumps
def scale_variables(X, scaling_factors):
    return X / scaling_factors

# Function to check if any value in X exceeds a threshold (e.g., 10^5)
def exceeds_threshold(X, threshold):
    return any(np.abs(X) > threshold)

def generate_data(max_reinitialization_attempts, t):
    # Attempt to find suitable initial conditions
    reinitialization_attempts = 0
    while reinitialization_attempts < max_reinitialization_attempts:
        X0 = np.random.uniform(-1, 1, n)  # Initialize X0 with random values between -1 and 1

        # Solve the system of ODEs with scaled variables
        X_scaled = scale_variables(X0, np.max(X0))  # Scale the initial conditions
        X_smooth = odeint(system_of_odes, X_scaled, t, args=(causal_params,), rtol=1e-8, atol=1e-8)

        # Reverse the scaling to obtain realistic data
        X_smooth = scale_variables(X_smooth, np.max(X0))

        # Solve the system of ODEs with the original initial conditions
        X = odeint(system_of_odes, X0, t, args=(causal_params,), rtol=1e-8, atol=1e-8)

        # If the result does not exceed the threshold, break the loop
        if not exceeds_threshold(X[-1], threshold):
            break
        reinitialization_attempts += 1
        if reinitialization_attempts == max_reinitialization_attempts:
            print("Warning: Maximum reinitialization attempts reached.")
    return X



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Choose parameters for maximal degree and numer of variables
    p = 3
    n = 3

    # Generate the complete degree dictionary
    degree_dict = generate_complete_degree_dictionary(p, n)
    print("Degree Dictionary:")
    print(degree_dict)

    # Count the total number of parameters
    n_params = count_params(degree_dict)
    print("Total Number of Parameters:", n_params)

    # Generate causal parameters based on the degree dictionary
    causal_params = generate_causal_parameters(degree_dict, n)
    print("Causal Parameters:")
    print(causal_params)

    # Initial conditions
    threshold = 1e5

    # Maximum number of reinitialization attempts
    max_reinitialization_attempts = 10
    # Define the time points at which you want to evaluate the solution
    t = np.linspace(0, 1, 100)  # Increase the number of time points for smoother data
    X = generate_data(max_reinitialization_attempts, t)

    # Plot the time series data
    plot_time_series(t, X, n, causal_params)



    # debugging
    # # Example usage to access time series data:
    # for i in range(n):
    #     variable_data = X[:, i]
    #     print(f"Time Series Data for x{i + 1}:", variable_data)
