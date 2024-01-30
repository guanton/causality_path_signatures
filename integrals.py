import numpy as np
from formatting_helpers import *
def compute_level_1_path(x_l, a, b):
    '''
    This computes S^l_(a,b) where (a,b) are the endpoints of the subinterval
    :param x_l: time_series data for x_l
    :param a: start time index
    :param b: end time index
    :return: path integral S^l_(a,b)
    '''
    path = x_l[b] - x_l[a]
    return path

def compute_path_monomial(X, m, copy, monomial, a, b):
    '''
    :param X: df for time series data for all variables X_i
    :param copy: which copy of the time series we are using
    :param monomial: the monomial that we will be integrating against
    :param a: start time index
    :param b: end time index
    :return:
    '''
    t = X.index
    t_sub = [t[idx] for idx in range(a, b+1)] # b+1 for including b
    # collect all monomial values over all time indices in the subinterval
    monomial_values_subinterval = extract_monomial_time_values(X, monomial, b, m, copy)
    integral = np.trapz(monomial_values_subinterval, x=t_sub)
    return integral
    #
    # relevant_variables = [v for v in range(m) if monomial[v] != 0]
    # degrees = [monomial[v] for v in relevant_variables]
    #
    # monomial_value_a, monomial_value_b = 1, 1
    # for v, degree in zip(relevant_variables, degrees):
    #     x_v = X.loc[:, (v, copy)].to_numpy()
    #     monomial_value_a *= x_v[a] ** degree
    #     monomial_value_b *= x_v[b] ** degree
    # return monomial_value_b - monomial_value_a


def compute_iterated_integral(X, m, derivatives_df, copy, l, word, subinterval, monomial=None, j_index=None):
    '''
    Compute the iterated integral.
    :param X: DataFrame containing time series data for all variables X_i
    :param m: Number of total variables
    :param derivatives_df: DataFrame containing the derivatives of the time series data
    :param copy: Index of the copy being considered for the variables
    :param l: The causal variable of interest
    :param word: The multi-index for which the integral is being computed
    :param subinterval: The subinterval of integration
    :param monomial: The monomial that replaces dx_l in the integral, if applicable
    :param j_index: Index at which the monomial is to be integrated, if applicable
    :return: The computed iterated integral
    '''
    t = X.index.to_numpy()
    # X = convert_df_to_array(X)
    integrators = {}
    for idx in range(len(word)):
        integrators[idx] = [0] * len(subinterval)  # Initialize each integrator as a list of values over the subinterval
    # integrate letter by letter
    for idx in range(len(word)):
        for s in subinterval:
            letter = word[idx]  # extract current letter
            # base case is level 1 path
            if idx == 0:
                # case where we are replacing the first index with the monomial for x_l
                if j_index == 0 and monomial is not None:
                    assert letter == l, "the provided index must correspond to the variable of interest"
                    # t_sub = [t[idx] for idx in subinterval]
                    # relevant_variables = [v for v in range(m) if monomial[v] != 0]
                    # degrees = [monomial[v] for v in relevant_variables]
                    # # collect all monomial values over all time indices in the subinterval
                    # monomial_values_subinterval = [
                    #     np.prod([X[idx, v] ** degree for v, degree in zip(relevant_variables, degrees)]) for idx in
                    #     subinterval]
                    # # integrate the jth monomial with respect to the ith subinterval using trapezoidal method
                    # integral = np.trapz(monomial_values_subinterval, x=t_sub)
                    integrators[0][s] = compute_path_monomial(X, m, copy, monomial, 0, s)
                else:
                    # otherwise, integrate normally against the given variable
                    x_i = X.loc[:, (letter, copy)].to_numpy()#X[:, letter]
                    integrators[0][s] = compute_level_1_path(x_i, 0, s)
            else:
                # we will be integrating against the previous level
                integrator_s = integrators[idx - 1][:s+1]
                # case where letter corresponds to the monomial for x_l
                if j_index == idx and monomial is not None:
                    assert letter == l, "the provided index must correspond to the variable of interest"
                    monomial_time_values = extract_monomial_time_values(X, monomial, s, m, copy)
                    integrators[idx][s] = compute_integral(t, integrator_s, derivatives_df, letter, copy, s, monomial_time_values)
                else:
                    # otherwise, integrate normally against the given variable
                    integrators[idx][s] = compute_integral(t, integrator_s, derivatives_df, letter, copy, s)
    return integrators[len(word)-1][len(subinterval)-1]

def extract_monomial_time_values(X, monomial, j, m, copy):
    '''
    :param X: df for time series data for all variables X_i
    :param monomial: array represetnation of monomial
    :param j: t[j] is the endpoint of the integral
    :param m: number of causal variables
    :param copy: which time series copy we are using
    '''
    relevant_variables = [v for v in range(m) if monomial[v] != 0]
    degrees = [monomial[v] for v in relevant_variables]
    monomial_time_values = [1 for s in range(j+1)]
    for v, degree in zip(relevant_variables, degrees):
        x_v = X.loc[:, (v, copy)].to_numpy()
        for s in range(j+1):
            monomial_time_values[s] *= x_v[s] ** degree
    return monomial_time_values
def compute_integral(t, integrator, derivatives_df, i_k, copy, j, monomial_time_values=None):
    '''
    :param t: array of times
    :param integrator: integrator function as array of its values from time t[0] to t[j]
    :param derivatives_df: df for the derivatives of the time series data
    :param i_k: the index of the variable that we are integrating against
    :param copy: the index of the copy that we are considering for the variables
    :param j: t[j] is the endpoint of the integral
    :param monomial_time_values: if not None, we will instead integrate against the provided monomial
    :return: integral value
    '''
    mi = (i_k, copy)
    if monomial_time_values is None:
        values_to_integrate = derivatives_df.loc[:t[j], mi].to_numpy()  # inclusive indexing for pandas
        # print('ok: ', values_to_integrate)

    else:
        values_to_integrate = np.array(monomial_time_values[:j+1])  # exclusive indexing for list

    # Ensure that the length of the integrator matches the length of the values to integrate
    assert len(integrator) == len(values_to_integrate), f"integrator dimension: {len(integrator)} does not align with dimension of values to integrate: {len(values_to_integrate)}"

    # Compute the integral using the trapezoidal rule
    t_sub = t[:j+1]
    integral = np.trapz(values_to_integrate * integrator[:j+1], x=t_sub)

    return integral

# def compute_integral(t, integrator, derivatives_df, i_k, copy, j, monomial_time_values = None):
#     '''
#     :param t: array of times
#     :param integrator: integrator function as array of its values from time t[0] to t[j]
#     :param derivatives_df: df for the derivatives of the time series data
#     :param i_k: the index of the variable that we are integrating aginst
#     :param copy: the index of the copy that we are considering for the variables
#     :param j: t[j] is the endpoint of the integral
#     :param monomial_time_values: if not None, we will instead integrate against the provided monomial
#     :return:
#     '''
#
#     mi = (i_k, copy)
#     h = (t[-1] - t[0])/(len(t)-1)
#     derivatives_array = derivatives_df.loc[:j, mi].to_numpy()
#     integral = 0
#     assert len(integrator) == j, "integrator dimension does not align with the time interval"
#     if monomial_time_values is None:
#         for i in range(j):
#             rect = integrator[i] * derivatives_array[i] * h
#             integral += rect
#     else:
#         for i in range(j):
#             rect = integrator[i] * monomial_time_values[i] * h
#             integral += rect
#     return integral
#
#  t_sub = [t[idx] for idx in range(j)]
#
#     integral = np.trapz(monomial_values_subinterval, x=t_sub)
#     return integral
