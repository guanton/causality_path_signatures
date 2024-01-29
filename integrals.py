
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

def compute_path_monomial(X, m, copy, l, monomial, a, b):
    '''
    :param X: df for time series data for all variables X_i
    :param copy: which copy of the time series we are using
    :param l: the causal variable of interest
    :param monomial: the monomial that we will be integrating against
    :param a: start time index
    :param b: end time index
    :return:
    '''
    relevant_variables = [v for v in range(m) if monomial[v] != 0]
    degrees = [monomial[v] for v in relevant_variables]

    monomial_value_a, monomial_value_b = 1, 1
    for v, degree in zip(relevant_variables, degrees):
        x_v = X.loc[:, (v, copy)].to_numpy()
        monomial_value_a *= x_v[a] ** degree
        monomial_value_b *= x_v[b] ** degree
    return monomial_value_b - monomial_value_a

def compute_iterated_integral(X, m, derivatives_df, copy, l, multi_index, subinterval, monomial=None, j_index=None):
    '''
    Compute the iterated integral.
    :param X: DataFrame containing time series data for all variables X_i
    :param m: Number of total variables
    :param derivatives_df: DataFrame containing the derivatives of the time series data
    :param copy: Index of the copy being considered for the variables
    :param l: The causal variable of interest
    :param multi_index: The multi-index for which the integral is being computed
    :param subinterval: The subinterval of integration
    :param monomial: The monomial that replaces dx_l in the integral, if applicable
    :param j_index: Index at which the monomial is to be integrated, if applicable
    :return: The computed iterated integral
    '''
    t = X.index.to_numpy()
    integrator = [0] * len(subinterval)  # Initialize integrator as a list
    for k in range(len(multi_index)):
        for s in subinterval:
            i_k = multi_index[k]  # extract current index
            if k == 0:
                if i_k == l and j_index == k and monomial is not None:
                    integrator[s] = compute_path_monomial(X, m, copy, l, monomial, 0, s)
                else:
                    x_i = X.loc[:, (i_k, copy)].to_numpy()
                    integrator[s] = compute_level_1_path(x_i, 0, s)
            else:
                integrator_s = integrator[:s]
                integrator[s] = compute_integral(t, integrator_s, derivatives_df, i_k, copy, s)
    return integrator[-1]

def compute_integral(t, integrator, derivatives_df, i_k, copy, j):
    '''
    :param t: array of times
    :param integrator: integrator function as array of its values from time t[0] to t[j]
    :param derivatives_df: df for the derivatives of the time series data
    :param i_k: the index of the variable that we are integrating aginst
    :param copy: the index of the copy that we are considering for the variables
    :param j: t[j] is the endpoint of the integral
    :return:
    '''
    mi = (i_k, copy)
    h = (t[-1] - t[0])/(len(t)-1)
    derivatives_array = derivatives_df.loc[:j, mi].to_numpy()
    integral = 0
    assert len(integrator) == j, "integrator dimension does not align with the time interval"
    for i in range(j):
        rect = integrator[i] * derivatives_array[i] * h
        integral += rect
    return integral