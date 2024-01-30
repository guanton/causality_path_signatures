import pandas as pd

'''
Helper functions for converting terms (as n-arrays) into strings
'''

# Function to convert term values into a sum representation
def rhs_as_sum(terms, latex=True):
    term_strings = []
    for coeff, term in terms:
        if all(term[j] == 0 for j in range(len(term))):
            term_strings.append(f'{coeff:.2f}')
        else:
            coeff_string = f'{coeff:.2f}'
            term_string = ''
            term_string += get_termstring(term, latex)  # f'$x_{{{j}}}^{{{term[j]}}}$'
            term_strings.append(coeff_string + term_string)
    if len(terms) > 0:
        polynomial_str = ' + '.join(term_strings)
        return polynomial_str
    else:
        return '0'


def get_termstring(term, latex=False):
    term_string = ''
    for j in range(len(term)):
        if term[j] > 0:
            if latex:
                term_string += f'$x_{{{j}}}^{{{term[j]}}}$'
            else:
                term_string += f'x_{j}^{term[j]}'
    # if all(term[j] == 0 for j in range(len(term))):
    #     term_string += '1'
    return term_string

def print_causal_relationships(causal_params):
    n = len(causal_params.keys())
    for i in range(n):
        # Display causal relationships if available
        if causal_params is not None and i in causal_params:
            terms = causal_params[i]
            causal_str = f'dx_{i}' + f'/dt = {rhs_as_sum(terms, latex=False)}'
        print(causal_str)

'''
Helper functions to convert between numpy array and pandas df representations of data
'''
def convert_array_to_df(X_array, t, m):
    """
    Convert NumPy array X_array to a DataFrame.

    Parameters:
    - X_array: NumPy array with time indices as the first dimension and variables as the second dimension
    - t: Array of time indices
    - m: Number of variables

    Returns:
    - X_df: DataFrame with multi-level columns (variable, copy) and time indices
    """

    # Assuming copy is fixed at 0
    copy_values = [0]

    # Create MultiIndex for columns
    mi = pd.MultiIndex.from_product([range(m), copy_values], names=['variable', 'copy'])

    # Create DataFrame
    X_df = pd.DataFrame(X_array, index=t, columns=mi)

    return X_df

def convert_df_to_array(X):
    """
    Convert DataFrame X to a NumPy array.

    Parameters:
    - X: DataFrame with multi-level columns (variable, copy) and time indices

    Returns:
    - X_array: numpy array with time indices as the first dimension and variables as the second dimension
    """

    # Assuming copy is fixed at 0
    copy_value = 0

    # Use .xs to cross-section the DataFrame
    X_fixed_copy = X.xs(key=copy_value, axis=1, level='copy').to_numpy()

    return X_fixed_copy