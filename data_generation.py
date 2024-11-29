import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Heun, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt
import time
import iisignature
from typing import Callable

def compute_total_variables(m: int, q: int) -> (int, list):
    """
    Computes the total number of variables M and the number of variables at each level.
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
    level_start_indices = [1]  # Start index for level 1 variables
    for size in level_sizes[:-1]:
        level_start_indices.append(level_start_indices[-1] + size)

    def drift(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        drift_vector = jnp.zeros(M)
        drift_vector = drift_vector.at[0].set(0.0)  # Zero-order term remains constant
        drift_vector = drift_vector.at[1].set(1.0)  # Time variable derivative is 1

        # Level 1 (primary variables)
        for i in range(2, m + 2):  # Indices for primary variables
            sum_a_y = jnp.dot(a[i], y)
            drift_vector = drift_vector.at[i].set(sum_a_y)

        # Higher levels
        for level in range(2, q + 1):
            prev_level_start = level_start_indices[level - 2]
            curr_level_start = level_start_indices[level - 1]
            prev_level_size = level_sizes[level - 2]
            curr_level_size = level_sizes[level - 1]

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
            prev_level_start = level_start_indices[level - 2]
            curr_level_start = level_start_indices[level - 1]
            prev_level_size = level_sizes[level - 2]
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


def verify_iterated_integrals(primary_variables: np.ndarray, time_steps: np.ndarray, q: int):
    """
    Verifies iterated integrals up to level q using numerical integration.

    Parameters:
    primary_variables (np.ndarray): Array of shape (T, m+1) containing primary variable trajectories (including time).
    time_steps (np.ndarray): Array of shape (T,) containing time steps.
    q (int): Level of iterated integrals to compute.

    Returns:
    iterated_integrals (np.ndarray): Computed iterated integrals of shape (T, total_integrals).
    integral_indices (list): List of indices used for each iterated integral.
    level_sizes (list): Sizes of each level.
    """
    m_plus_one = primary_variables.shape[1]  # Includes time variable
    T = primary_variables.shape[0]
    level_sizes = [(m_plus_one) ** k for k in range(1, q + 1)]
    total_integrals = sum(level_sizes)
    iterated_integrals = np.zeros((T, total_integrals))

    integral_indices = []
    idx = 0
    for level in range(1, q + 1):
        for i in range((m_plus_one) ** level):
            indices = []
            temp_idx = i
            for _ in range(level):
                indices.append(temp_idx % m_plus_one)
                temp_idx = temp_idx // m_plus_one
            indices = indices[::-1]  # Reverse to get correct order
            integral_indices.append(indices)
            idx += 1

    for idx, indices in enumerate(integral_indices):
        integrand = primary_variables[:, indices[0]]
        for idx_i in indices[1:]:
            deriv = np.gradient(primary_variables[:, idx_i], time_steps)
            integrand *= deriv
        iterated_integrals[:, idx] = cumulative_trapezoid(integrand, time_steps, initial=0)

    return iterated_integrals, integral_indices, level_sizes


def solve_sde(
    drift_function: Callable,
    diffusion_function: Callable,
    initial_condition: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float
):
    # Define Brownian motion
    key = jax.random.PRNGKey(42)  # Ensure reproducibility
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
    solver = Heun()  # Heun solver, suitable for Stratonovich SDEs

    # Solve the SDE
    ts = jnp.arange(t0, t1 + dt, dt)
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

    return sol
#
# if __name__ == "__main__":
#     t_1 = time.time()
#     # Parameters
#     m = 1  # Number of primary variables (excluding time)
#     n = 2  # Dimension of Brownian motion W_t
#     q = 3  # Level of iterated integrals
#     M, level_sizes = compute_total_variables(m, q)  # Total number of variables including higher-order terms
#
#     theta = 1.0  # Adjusted parameter
#     sigma = 1.0  # Volatility for primary variable 1
#     sigma_2 = 1.0  # Volatility for primary variable 2
#
#     # Initialize the drift coefficient matrix a (M x M)
#     a = jnp.zeros((M, M))
#
#     # Explicit matrices for visual interpretation
#     # Index mapping:
#     #   0: zero-order term
#     #   1: time variable (t)
#     #   2: primary variable 1
#     #   3: primary variable 2
#     # Level 1 variables are from index 2 to m + 2
#
#     # Set drift coefficients for primary variables
#     a = a.at[2, 1].set(theta)  # Dependency on time for variable 1
#     a = a.at[3, 3].set(-theta)  # Self-interaction for variable 2
#
#     # You can set higher-order drift coefficients here if needed
#
#     # Initialize the diffusion coefficient tensor b (M x n x M)
#     b = jnp.zeros((M, n, M))
#
#     # Set diffusion coefficients explicitly
#     b = b.at[2, 0, 0].set(sigma)     # Variable 1, additive noise via zero-order term
#     b = b.at[3, 1, 0].set(sigma_2)   # Variable 2, additive noise via zero-order term
#
#     # Initial condition X0, shape (M,)
#     X0 = jnp.zeros(M)
#     X0 = X0.at[0].set(1.0)  # Zero-order term initialized to 1
#     X0 = X0.at[1].set(0.0)  # Time variable starts at t = 0
#
#     # Create the drift and diffusion functions
#     drift = drift_function_with_time(m, q, a)
#     diffusion = diffusion_function_with_time(m, n, q, b)
#
#     # Time parameters
#     t0 = 0.0
#     t1 = 0.1
#     dt = 0.01
#
#     # Solve the SDE
#     solution = solve_sde(
#         drift_function=drift,
#         diffusion_function=diffusion,
#         initial_condition=X0,
#         t0=t0,
#         t1=t1,
#         dt=dt
#     )
#
#     # Evaluate the solution at t = t1
#     evaluated_solution = solution.ys[-1]
#     print(f"Solution at t={t1}:\n", evaluated_solution)
#
#     t_2 = time.time()
#     print('Total computation time:', t_2 - t_1)
#
#     # Function to plot strictly primary variables
#     def plot_primary_variables(solution, m):
#         t_values = np.array(solution.ts)
#         y_values = np.array(solution.ys[:, 2:m+2])  # Extract primary variables (indices 2 to m+1)
#
#         plt.figure(figsize=(10, 6))
#         for i in range(y_values.shape[1]):
#             plt.plot(t_values, y_values[:, i], label=f'Y_{i+1}')
#         plt.xlabel('Time')
#         plt.ylabel('Value')
#         plt.title('Primary Variables Trajectories')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#     # Plot the primary variables
#     plot_primary_variables(solution, m)
#
#     # Extract primary variables and compute iterated integrals
#     primary_variables = np.array(solution.ys[:, 1:m + 2])  # Indices 1 to m+1 (including time variable)
#     time_steps = np.array(solution.ts)
#     iterated_integrals, integral_indices, level_sizes_integrals = verify_iterated_integrals(primary_variables, time_steps, q)
#
#     # Exclude level 1 integrals
#     start_idx_integrals = level_sizes_integrals[0]
#     iterated_integrals_higher = iterated_integrals[:, start_idx_integrals:]
#
#     # Extract higher-order variables from the solution
#     level_start_indices = [1]  # Starting index after zero-order term
#     for size in level_sizes[:-1]:
#         level_start_indices.append(level_start_indices[-1] + size)
#     higher_order_start_idx = level_start_indices[1]  # Starting index of level 2 variables
#     higher_order_variables = np.array(solution.ys[:, higher_order_start_idx:])
#
#     # Reshape higher_order_variables to (T, total_higher_order_variables)
#     total_higher_order_variables = sum(level_sizes[1:])  # Exclude level 1 variables
#     higher_order_variables = higher_order_variables.reshape(-1, total_higher_order_variables)
#
#     # Verify shapes match
#     print(f"Shape of higher_order_variables: {higher_order_variables.shape}")
#     print(f"Shape of iterated_integrals_higher: {iterated_integrals_higher.shape}")
#
#     # Compute the absolute difference
#     difference = np.abs(higher_order_variables - iterated_integrals_higher)
#     max_difference = np.nanmax(difference)  # Use nanmax to ignore NaNs
#     print(f"Maximum difference between expanded system and numerical integration: {max_difference}")
#
#     # Plot raw time series for expanded system and numerical integration
#     plt.figure(figsize=(12, 8))
#     for idx in range(iterated_integrals_higher.shape[1]):
#         plt.plot(time_steps, higher_order_variables[:, idx], linestyle='-', label=f'Expanded Var {idx}')
#         plt.plot(time_steps, iterated_integrals_higher[:, idx], linestyle='--', label=f'Numerical Var {idx}')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.title('Comparison of Expanded System and Numerical Integration')
#     plt.legend(ncol=2, loc='upper left', fontsize='small')
#     plt.grid(True)
#     plt.show()
#
#     # Function to plot all variables
#     def plot_all_variables(solution):
#         t_values = np.array(solution.ts)
#         y_values = np.array(solution.ys)  # All variables
#
#         plt.figure(figsize=(10, 6))
#         for i in range(y_values.shape[1]):
#             plt.plot(t_values, y_values[:, i], label=f'Y_{i}')
#         plt.xlabel('Time')
#         plt.ylabel('Value')
#         plt.title('All Variables Trajectories')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#     # Plot all variables
#     plot_all_variables(solution)
if __name__ == "__main__":
    t_1 = time.time()
    # Parameters
    m = 1  # Number of primary variables (excluding time)
    n = 1  # Dimension of Brownian motion W_t
    q = 2  # Level of iterated integrals
    M, level_sizes = compute_total_variables(m, q)  # Total number of variables including higher-order terms

    theta = 1.0  # Adjusted parameter
    sigma = 1.0  # Volatility for primary variable 1
    sigma_2 = 1.0  # Volatility for primary variable 2

    # Initialize the drift coefficient matrix a (M x M)
    a = jnp.zeros((M, M))

    # Explicit matrices for visual interpretation
    # Index mapping:
    #   0: zero-order term
    #   1: time variable (t)
    #   2: primary variable 1
    #   3: primary variable 2
    # Level 1 variables are from index 2 to m + 2

    # Set drift coefficients for primary variables
    a = a.at[2, 1].set(theta)  # Dependency on time for variable 1
    a = a.at[3, 3].set(-theta)  # Self-interaction for variable 2

    # Initialize the diffusion coefficient tensor b (M x n x M)
    b = jnp.zeros((M, n, M))

    # Set diffusion coefficients explicitly
    b = b.at[2, 0, 0].set(sigma)     # Variable 1, additive noise via zero-order term
    b = b.at[3, 1, 0].set(sigma_2)   # Variable 2, additive noise via zero-order term

    # Initial condition X0, shape (M,)
    X0 = jnp.zeros(M)
    X0 = X0.at[0].set(1.0)  # Zero-order term initialized to 1
    X0 = X0.at[1].set(0.0)  # Time variable starts at t = 0

    # Create the drift and diffusion functions
    drift = drift_function_with_time(m, q, a)
    diffusion = diffusion_function_with_time(m, n, q, b)

    # Time parameters
    t0 = 0.0
    t1 = 2.0
    dt = 0.01

    # Solve the SDE
    solution = solve_sde(
        drift_function=drift,
        diffusion_function=diffusion,
        initial_condition=X0,
        t0=t0,
        t1=t1,
        dt=dt
    )

    # Evaluate the solution at t = t1
    evaluated_solution = solution.ys[-1]
    print(f"Solution at t={t1}:\n", evaluated_solution)

    t_2 = time.time()
    print('Total computation time:', t_2 - t_1)

    # Extract primary variables and compute iterated integrals using iisignature
    primary_variables = np.array(solution.ys[:, 1:m + 2])  # Indices 1 to m+1 (including time variable)
    time_steps = np.array(solution.ts)

    # Prepare the signature computation
    sig_length = iisignature.siglength(m + 1, q)
    print(f"Expected signature length: {sig_length}")

    iterated_integrals = np.zeros((len(time_steps), sig_length))

    for i in range(len(time_steps)):
        path_segment = primary_variables[:i+1, :]
        iterated_integrals[i, :] = iisignature.sig(path_segment, q)

    print(f"Computed iterated_integrals shape: {iterated_integrals.shape}")

    # Exclude level 1 integrals from iterated_integrals
    level_1_sig_length = iisignature.siglength(m + 1, 1)
    iterated_integrals_higher = iterated_integrals[:, level_1_sig_length:]

    # Extract higher-order variables from the solution
    level_start_indices = [1]  # Starting index after zero-order term
    for size in level_sizes[:-1]:
        level_start_indices.append(level_start_indices[-1] + size)
    higher_order_start_idx = level_start_indices[1]  # Starting index of level 2 variables
    higher_order_variables = np.array(solution.ys[:, higher_order_start_idx:])
    higher_order_variables = higher_order_variables.reshape(-1, sum(level_sizes[1:]))

    # Verify shapes match
    print(f"Shape of higher_order_variables: {higher_order_variables.shape}")
    print(f"Shape of iterated_integrals_higher: {iterated_integrals_higher.shape}")

    # Ensure shapes match
    min_length = min(higher_order_variables.shape[1], iterated_integrals_higher.shape[1])
    higher_order_variables = higher_order_variables[:, :min_length]
    iterated_integrals_higher = iterated_integrals_higher[:, :min_length]

    # Compute the absolute difference
    difference = np.abs(higher_order_variables - iterated_integrals_higher)
    max_difference = np.nanmax(difference)  # Use nanmax to ignore NaNs
    print(f"Maximum difference between expanded system and numerical integration: {max_difference}")

    # Plot raw time series for expanded system and numerical integration
    plt.figure(figsize=(12, 8))
    for idx in range(min_length):
        plt.plot(time_steps, higher_order_variables[:, idx], linestyle='-', label=f'Expanded Var {idx}')
        plt.plot(time_steps, iterated_integrals_higher[:, idx], linestyle='--', label=f'Numerical Var {idx}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Comparison of Expanded System and Numerical Integration')
    plt.legend(ncol=2, loc='upper left', fontsize='small')
    plt.grid(True)
    plt.show()

    # Function to plot all variables
    def plot_all_variables(solution):
        t_values = np.array(solution.ts)
        y_values = np.array(solution.ys)  # All variables

        plt.figure(figsize=(10, 6))
        for i in range(y_values.shape[1]):
            plt.plot(t_values, y_values[:, i], label=f'Y_{i}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('All Variables Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot all variables
    plot_all_variables(solution)
