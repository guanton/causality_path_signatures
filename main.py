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


def plot_time_series(t, X, n):
    # Assuming you have already solved the ODEs and have the time series data in the X array.
    # n is the number of variables.
    # Extract the time series data for each variable
    variable_names = [f'x{i}' for i in range(1, n + 1)]  # Variable names x1, x2, ..., xn

    # Plot each variable
    plt.figure(figsize=(10, 6))

    for i in range(n):
        # plot time series data for ith variable
        plt.plot(t, X[:, i], label=variable_names[i])

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data')

    # Add a legend to label each variable
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Example usage:
    p = 3
    n = 4
    degree_dict = generate_complete_degree_dictionary(p, n)
    print(degree_dict)
    print(count_params(degree_dict))
    causal_params = generate_causal_parameters(degree_dict, n)
    print(causal_params)
    # Initial conditions
    # X0 = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    X0 = np.ones(n)
    # Time points at which you want to evaluate the solution
    t = np.linspace(0, 10, 100)  # Example: from 0 to 10, divided into 100 time points

    # Solve the system of ODEs
    X = odeint(system_of_odes, X0, t, args=(causal_params,), rtol=1e-8, atol=1e-8)
    plot_time_series(t, X, n)
    #
    # X contains the time series data for x1, x2, and x3
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    # Example usage to access time series data:
    print("Time Series Data for x1:", x1)
    print("Time Series Data for x2:", x2)
    print("Time Series Data for x3:", x3)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
