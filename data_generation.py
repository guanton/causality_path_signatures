import diffrax
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Heun, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
from scipy.integrate import cumulative_trapezoid
import time
import iisignature
from typing import Callable
from plots import *
import pandas as pd
import json
import itertools
import argparse

def generate_signature_multi_indices(d, q):
    """
    Generate the multi-indices for the signature terms up to level q.

    Parameters:
    d (int): Number of channels (dimensions) in the path.
    q (int): Level of the signature.

    Returns:
    list: A list of tuples representing the multi-indices.
    """
    indices = []
    for level in range(1, q + 1):
        # Generate all possible combinations of indices at this level
        level_indices = itertools.product(range(d), repeat=level)
        indices.extend(level_indices)
    return indices

def generate_all_variable_indices(m: int, q: int) -> dict:
    """
    Generates a mapping from variable names to their indices in the tensors.

    Returns:
    dict: A dictionary where keys are variable names and values are indices arranged.
    from 0 to M - 1 = m+1 + ... + (m+1)^q
    """
    variable_indices = {'Y_0': 0, 't': 1}  # Zero-order term and time variable
    idx = 2  # Starting index for primary variables

    # Level 1 variables (primary variables)
    for i in range(1, m + 1):  # Primary variables indexed from 1 to m
        variable_indices[f'Y_{i}'] = idx
        idx += 1

    # Higher-level variables
    for level in range(2, q + 1):
        level_size = (m + 1) ** level
        for i in range(level_size):
            indices = []
            temp_i = i
            for _ in range(level):
                indices.append(temp_i % (m + 1))
                temp_i = temp_i // (m + 1)
            indices = indices[::-1]  # Reverse to get correct order
            var_name = 'Y_' + '_'.join(str(idx_j) for idx_j in indices)
            variable_indices[var_name] = idx
            idx += 1

    return variable_indices

def generate_variable_indices(m: int, q: int) -> dict:
    """
    Generates a mapping from variable names to their indices in the tensors.

    Returns:
    dict: A dictionary where keys are variable names and values are indices arranged.
    """
    variable_indices = {}  # Initialize empty dict
    idx = 0  # Starting index

    variable_names = ['t'] + [f'Y_{i}' for i in range(1, m + 1)]  # ['t', 'Y_1', 'Y_2']

    # Level 1 variables (primary variables including time)
    for var_name in variable_names:
        variable_indices[var_name] = idx
        idx += 1

    # Higher-level variables
    for level in range(2, q + 1):
        for idx_tuple in itertools.product(range(len(variable_names)), repeat=level):
            var_name = '_'.join(variable_names[i] for i in idx_tuple)
            variable_indices[var_name] = idx
            idx += 1

    return variable_indices


def set_coefficients_from_dict(a, b, variable_indices, coefficients, n):
    """
    Sets the coefficients in the drift and diffusion tensors based on the provided dictionary.

    Parameters:
    a (jnp.ndarray): Drift coefficient matrix.
    b (jnp.ndarray): Diffusion coefficient tensor.
    variable_indices (dict): Mapping of variable names to indices.
    coefficients (dict): Nested dictionary containing the coefficients.
    n (int): Dimension of Brownian motion.
    """
    # Reset coefficients to zero
    a = a.at[:].set(0.0)
    b = b.at[:].set(0.0)

    # Set drift coefficients from drift sub-dictionary
    drift_coeffs = coefficients.get('drift', {})
    for var_name, dependencies in drift_coeffs.items():
        i = variable_indices.get(var_name)
        if i is None:
            continue
        for dep_var, value in dependencies.items():
            k = variable_indices.get(dep_var)
            if k is None:
                continue
            a = a.at[i, k].set(value)

    # Set diffusion coefficients
    diffusion_coeffs = coefficients.get('diffusion', {})
    for var_name, dependencies in diffusion_coeffs.items():
        i = variable_indices.get(var_name)
        if i is None:
            continue
        for dep_var, bm_components in dependencies.items():
            k = variable_indices.get(dep_var)
            if k is None:
                continue
            # Convert keys to int before checking
            for j, value in bm_components.items():
                j_int = int(j)  # Ensure we have an integer index
                if j_int < n:
                    b = b.at[i, j_int, k].set(value)

    return a, b




def compute_total_variables(m: int, q: int) -> (int, list):
    """
    Computes the total number of variables M in the expanded system and the number of variables at each level.
    """
    M = 1  # Zero-order term
    level_sizes = []
    for k in range(1, q + 1):
        size = (m + 1) ** k
        level_sizes.append(size)
        M += size
    return M, level_sizes

def drift_function_with_time(m: int, q: int, a: jnp.ndarray) -> Callable:
    M, level_sizes = compute_total_variables(m, q)
    level_start_indices = [1]  # Start index for first order terms
    for size in level_sizes[:-1]:
        level_start_indices.append(level_start_indices[-1] + size)  # start index for higher order terms

    def drift(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        drift_vector = jnp.zeros(M)
        drift_vector = drift_vector.at[0].set(0.0)  # Zero-order term remains constant
        drift_vector = drift_vector.at[1].set(1.0)  # Time variable derivative is 1

        # Level 1 (primary variables)
        for i in range(2, m + 2):  # Indices for true primary variables are 2 to m+1 (0-indexed)
            sum_a_y = jnp.dot(a[i], y) # compute linear drift
            drift_vector = drift_vector.at[i].set(sum_a_y)

        # Drift for higher levels are defined recursively
        for level in range(2, q + 1):
            curr_level_start = level_start_indices[level - 1]
            curr_level_size = level_sizes[level - 1] # -1 because zeroth order is omitted

            for idx in range(curr_level_size):
                # Multi-index for current variable
                indices = []
                temp_idx = idx
                for _ in range(level):
                    indices.append(temp_idx % (m + 1))
                    temp_idx = temp_idx // (m + 1)
                indices = indices[::-1]  # Reverse to get correct order
                i = curr_level_start + idx  # Index of current variable

                # Compute product term
                product_term = drift_vector[1 + indices[-1]]
                for k in indices[:-1]:
                    product_term *= y[1 + k]
                sum_a_y = jnp.dot(a[i], y)
                drift_vector = drift_vector.at[i].set(product_term + sum_a_y)
        return drift_vector

    return drift

def diffusion_function_with_time(m: int, n: int, q: int, b: jnp.ndarray) -> Callable:
    M, level_sizes = compute_total_variables(m, q)
    level_start_indices = [1]
    for size in level_sizes[:-1]:
        level_start_indices.append(level_start_indices[-1] + size)

    def diffusion(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        diffusion_matrix = jnp.zeros((M, n))
        diffusion_matrix = diffusion_matrix.at[0, :].set(0.0)  # Zero-order term
        diffusion_matrix = diffusion_matrix.at[1, :].set(0.0)  # Time variable

        # Level 1 (primary variables)
        for i in range(2, m + 2):
            for j in range(n):
                sum_b_y = jnp.dot(b[i, j], y)
                diffusion_matrix = diffusion_matrix.at[i, j].set(sum_b_y)

        # Higher levels
        for level in range(2, q + 1):
            curr_level_start = level_start_indices[level - 1]
            curr_level_size = level_sizes[level - 1]

            for idx in range(curr_level_size):
                # Multi-index for current variable
                indices = []
                temp_idx = idx
                for _ in range(level):
                    indices.append(temp_idx % (m + 1))
                    temp_idx = temp_idx // (m + 1)
                indices = indices[::-1]  # Reverse to get correct order
                i = curr_level_start + idx

                for j in range(n):
                    # Compute product term
                    product_term = diffusion_matrix[1 + indices[-1], j]
                    for k in indices[:-1]:
                        product_term *= y[1 + k]
                    sum_b_y = jnp.dot(b[i, j], y)
                    diffusion_value = product_term + sum_b_y
                    diffusion_matrix = diffusion_matrix.at[i, j].set(diffusion_value)
        return diffusion_matrix

    return diffusion

def compute_iterated_integrals_iisignature(primary_variables: np.ndarray, q: int) -> np.ndarray:
    """
    Computes iterated integrals up to level q using iisignature.

    Parameters:
    primary_variables (np.ndarray): Array of shape (T, d) containing primary variable trajectories.
    q (int): Level of iterated integrals to compute.

    Returns:
    np.ndarray: Computed iterated integrals (signature) up to level q.
    """
    # iisignature expects the path to be an array of shape (T, d)
    # Ensure primary_variables includes the time variable as the first column
    path = primary_variables
    sig = iisignature.sig(path, q)
    return sig

def solve_sde(
    drift_function: Callable,
    diffusion_function: Callable,
    initial_condition: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    key: jax.random.PRNGKey  # Added key parameter
):
    # Define Brownian motion
    n = diffusion_function(0, initial_condition, None).shape[1]  # Dimension of Brownian motion
    bm = VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=1e-3,
        shape=(n,),
        key=key
    )


    # Define the SDE terms
    terms = MultiTerm(
        ODETerm(drift_function),
        ControlTerm(diffusion_function, bm)
    )

    # Set the solver
    solver = Heun()  # Heun solver, suitable for Stratonovich SDEs with non-commutative noise

    # Solve the SDE
    num_steps = int((t1 - t0) / dt) + 1  # Ensure the endpoint t1 is included
    ts = jnp.linspace(t0, t1, num_steps)
    saveat = SaveAt(ts=ts)
    sol = diffeqsolve(
        terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=initial_condition,
        saveat=saveat
    )

    return sol, solver

def estimate_A(X, dt, pinv=False):
    """
    Calculate the approximate closed form estimator A_hat for time homogeneous linear drift from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each slice corresponds to a single trajectory.
        dt (float): Discretization time step.
        pinv: whether to use pseudo-inverse. Otherwise, we use left_Var_Equation

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = X.shape
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))
    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for n in range(num_trajectories):
            xt = X[n, t, :]
            dxt = X[n, t + 1, :] - X[n, t, :]
            sum_dxt_xt += np.outer(dxt, xt)
            sum_xt_xt += np.outer(xt, xt)
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)


def estimate_GGT(trajectories, T, est_A=None):
    """
    Estimate the observational diffusion GG^T for a multidimensional linear
    additive noise SDE from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): pre-estimated drift A.
        If none provided, est_A = 0, modeling a pure diffusion process

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = trajectories.shape
    dt = T / (num_steps - 1)

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(trajectories, axis=1)
    else:
        # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
        increments = np.diff(trajectories, axis=1) - dt * np.einsum('ij,nkj->nki', est_A, trajectories[:, :-1, :])

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T * num_trajectories
    return GGT



def run_experiment(config: dict, experiment_name: str):
    t_1 = time.time()

    N = config["N"]
    m = config["m"]
    n = config["n"]
    q = config["q"]
    q_iterated = config["q_iterated"]
    t1 = config["t1"]
    dt = config["dt"]
    coefficients = config["coefficients"]
    base_key_val = config["base_key"]

    t0 = 0.0

    M, level_sizes = compute_total_variables(m, q)
    X0 = jnp.zeros(M)
    X0 = X0.at[0].set(1.0)
    X0 = X0.at[1].set(0.0)

    a = jnp.zeros((M, M))
    b = jnp.zeros((M, n, M))

    variable_indices_full = generate_all_variable_indices(m, q)
    a, b = set_coefficients_from_dict(a, b, variable_indices_full, coefficients, n)

    drift = drift_function_with_time(m, q, a)
    diffusion = diffusion_function_with_time(m, n, q, b)

    base_key = jax.random.PRNGKey(base_key_val)
    keys = jax.random.split(base_key, N)

    solutions = []
    for i in range(N):
        sol, solver_used = solve_sde(
            drift_function=drift,
            diffusion_function=diffusion,
            initial_condition=X0,
            t0=t0,
            t1=t1,
            dt=dt,
            key=keys[i]
        )
        solutions.append(sol)

    ys_array = jnp.stack([sol.ys for sol in solutions], axis=0)
    ys_array = np.array(ys_array)

    primary_data = ys_array[:, :, 2:m + 2]
    primary_data_with_time = ys_array[:, :, 1:m + 2]

    A_hat = estimate_A(primary_data, dt, pinv=True)
    H_hat = estimate_GGT(primary_data, t1, A_hat)
    print('estimated A:', A_hat)
    print('estimated H:', H_hat)

    t_2 = time.time()
    print('Total computation time:', t_2 - t_1)

    sig_length = iisignature.siglength(m + 1, q_iterated)
    signatures = np.zeros((N, sig_length))
    for n_ in range(N):
        path = primary_data_with_time[n_, :, :]
        sig = iisignature.sig(path, q_iterated)
        signatures[n_, :] = sig

    expected_signature = np.mean(signatures, axis=0)

    variable_names = ['t'] + [f'Y_{i}' for i in range(1, m + 1)]
    multi_indices = generate_signature_multi_indices(m+1, q_iterated)
    variable_indices = generate_variable_indices(m, q_iterated)

    multi_index_strings = []
    multi_index_to_variable_index = {}
    for idx_tuple in multi_indices:
        variable_name = '_'.join(variable_names[idx_j] for idx_j in idx_tuple)
        variable_index = variable_indices.get(variable_name)
        multi_index_strings.append(variable_name)
        multi_index_to_variable_index[variable_name] = variable_index

    expected_value_sde = []
    for multi_idx_str in multi_index_strings:
        variable_index = multi_index_to_variable_index.get(multi_idx_str)
        if variable_index is not None and variable_index < ys_array.shape[2]:
            variable_values = ys_array[:, -1, 1 + variable_index]
            expected_value = np.mean(variable_values)
        else:
            expected_value = np.nan
        expected_value_sde.append(expected_value)

    df = pd.DataFrame({
        'multi_index': multi_index_strings,
        'expected_value': expected_signature,
        'expected_value_sde': expected_value_sde
    })
    df['level'] = [len(idx_tuple) for idx_tuple in multi_indices]
    df['difference'] = df['expected_value'] - df['expected_value_sde']

    # Use the exact filenames requested by the user
    # For example:
    # For the first experiment: simple_OU_1_expected_signatures_N-1000_seed-0.csv
    # We'll assume `experiment_name` is given as e.g. "simple_OU_1"
    csv_filename = f"{experiment_name}_expected_signatures_N-{N}_seed-{base_key_val}.csv"
    df.to_csv(csv_filename, index=False)

    metadata = {
        'N': N,
        'm': m,
        'n': n,
        'q': q,
        'q_iterated': q_iterated,
        't1': t1,
        'dt': dt,
        'coefficients': coefficients,
        'base_key': int(base_key[0]),
        'solver': solver_used.__class__.__name__
    }

    json_filename = f"metadata_{experiment_name}_expected_signatures_N-{N}_seed-{base_key_val}.json"
    with open(json_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Plotting
    solution = solutions[0]
    # plot_primary_variables(solution, m)
    # plot_all_variables(solution)
    # plot_differences(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--experiment_name", required=True, help="Base name for experiment.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    run_experiment(config, args.experiment_name)


#
# if __name__ == "__main__":
#     t_1 = time.time()
#     experiment_name = 'proper_level_2_drift_and_diffusion_1'
#     # Parameters
#     N = 1000  # Number of trajectories
#     m = 2  # Number of primary variables (excluding time)
#     n = 1  # Dimension of Brownian motion W_t
#     q = 5  # Level of iterated integrals for simulation
#     q_iterated = 5 # Level of iterated integrals for iisignature computation
#     t0 = 0.0
#     t1 = 0.5
#     dt = 0.01
#
#     M, level_sizes = compute_total_variables(m, q)  # Total number of variables including higher-order terms
#
#     # Initial condition X0, shape (M,)
#     X0 = jnp.zeros(M)
#     X0 = X0.at[0].set(1.0)  # Zero-order term initialized to 1
#     X0 = X0.at[1].set(0.0)  # Time variable starts at t = 0
#
#
#
#     # Initialize the drift coefficient matrix a (M x M)
#     a = jnp.zeros((M, M))
#
#     # Initialize the diffusion coefficient tensor b (M x n x M)
#     b = jnp.zeros((M, n, M))
#
#     # Generate variable indices
#     variable_indices_full = generate_all_variable_indices(m, q)
#
#     # Define your coefficients
#     coefficients = {
#         'drift': {
#             'Y_1': {'Y_1': -1, 'Y_1_1': 0.1, 'Y_2': 0.1},
#             'Y_2': {'Y_0': 1, 'Y_2': -1, 'Y_1': 0.1, 'Y_2_1': 0.1},
#         },
#         'diffusion': {
#             'Y_1': {'Y_0': {0: 5}, 'Y_1': {0: 0.7}, 'Y_1_1': {0: 0.5}},
#             'Y_2': {'Y_0': {0: 0.5}, 'Y_2_2': {0: 0.5}},
#         }
#     }
#
#     # Set the coefficients in 'a' and 'b'
#     a, b = set_coefficients_from_dict(a, b, variable_indices_full, coefficients, n)
#
#     # Create the drift and diffusion functions
#     drift = drift_function_with_time(m, q, a)
#     diffusion = diffusion_function_with_time(m, n, q, b)
#
#     # Generate multiple trajectories
#     base_key = jax.random.PRNGKey(0)  # Base key
#     keys = jax.random.split(base_key, N)
#
#     solutions = []
#     for i in range(N):
#         sol, solver = solve_sde(
#             drift_function=drift,
#             diffusion_function=diffusion,
#             initial_condition=X0,
#             t0=t0,
#             t1=t1,
#             dt=dt,
#             key=keys[i]
#         )
#         solutions.append(sol)
#
#     ys_array = jnp.stack([sol.ys for sol in solutions], axis=0)  # Shape: (N, T, M)
#     ys_array = np.array(ys_array)
#     # Extract primary variables
#     primary_data = ys_array[:, :, 2:m + 2]  # Shape: (N, T, m)
#     primary_data_with_time = ys_array[:, :, 1:m + 2]  # Shape: (N, T, m)
#     print('sanity', primary_data.shape)
#     # Estimate the drift matrix
#     A_hat = estimate_A(primary_data, dt, pinv=True)
#     print("Estimated Drift Matrix A:")
#     print(A_hat)
#     H_hat = estimate_GGT(primary_data, t1, A_hat)
#     print("Estimated Diffusion Matrix H:")
#     print(H_hat)
#
#
#
#     t_2 = time.time()
#     print('Total computation time:', t_2 - t_1)
#
#     # Prepare the signature computation
#     sig_length = iisignature.siglength(m + 1, q_iterated)
#     # Initialize an array to hold the signatures
#     signatures = np.zeros((N, sig_length))
#
#     for n_ in range(N):
#         path = primary_data_with_time[n_, :, :]  # Shape: (T, m+1)
#         sig = iisignature.sig(path, q_iterated)
#         signatures[n_, :] = sig
#
#     print(signatures.shape)
#
#     # Compute the empirical expected signature
#     expected_signature = np.mean(signatures, axis=0)
#
#     # Generate multi-indices for the signature terms
#     variable_names = ['t'] + [f'Y_{i}' for i in range(1, m + 1)]
#     # For m=2, variable_names = ['t', 'Y_1', 'Y_2']
#
#     multi_indices = generate_signature_multi_indices(m+1, q_iterated)
#     # multi_index_strings = ['_'.join(variable_names[i] for i in idx_tuple) for idx_tuple in multi_indices]
#
#     # Create mapping from multi_index strings to variable indices
#     multi_index_to_variable_index = {}
#     multi_index_strings = []
#
#     # Generate variable indices without 0 order term
#     variable_indices = generate_variable_indices(m, q_iterated)
#
#     for idx_tuple in multi_indices:
#         variable_name = '_'.join(variable_names[idx_j] for idx_j in idx_tuple)
#         variable_index = variable_indices.get(variable_name)
#         multi_index_strings.append(variable_name)
#         multi_index_to_variable_index[variable_name] = variable_index
#
#     # Initialize list to hold expected values from SDE
#     expected_value_sde = []
#
#     for multi_idx_str in multi_index_strings:
#         variable_index = multi_index_to_variable_index.get(multi_idx_str)
#         if variable_index is not None and variable_index < ys_array.shape[2]:
#             # Extract variable values at variable_index from ys_array
#             variable_values = ys_array[:, -1, 1+variable_index]  # At final time point
#             expected_value = np.mean(variable_values)
#         else:
#             expected_value = np.nan  # Variable not present in SDE solution
#         expected_value_sde.append(expected_value)
#
#     # Create a DataFrame
#     df = pd.DataFrame({
#         'multi_index': multi_index_strings,
#         'expected_value': expected_signature,
#         'expected_value_sde': expected_value_sde
#     })
#
#     # Add a 'level' column
#     df['level'] = [len(idx_tuple) for idx_tuple in multi_indices]
#
#     # Compute the difference
#     df['difference'] = df['expected_value'] - df['expected_value_sde']
#
#
#     print(df.head(sig_length))
#
#     # Save the DataFrame to CSV
#     base_filename = f'{experiment_name}_expected_signatures_N-{N}_seed-{int(base_key[0])}'
#     csv_filename = base_filename + '.csv'
#     df.to_csv(csv_filename, index=False)
#     # Prepare metadata
#     metadata = {
#         'N': N,
#         'm': m,
#         'n': n,
#         'q': q,
#         'q_iterated': q_iterated,
#         't1': t1,
#         'dt': dt,
#         'coefficients': coefficients,
#         'base_key': int(base_key[0]),
#         'solver': solver.__class__.__name__,  # Include solver name
#     }
#
#
#     # Save metadata to JSON file
#     with open(f'metadata_{base_filename}.json', 'w') as f:
#         json.dump(metadata, f, indent=4)
#
#     solution = solutions[0]
#     plot_primary_variables(solution, m)
#     # Plot all variables
#     plot_all_variables(solution)
#     plot_differences(df)