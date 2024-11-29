import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Heun, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable

def drift_function_with_time(m: int, a: jnp.ndarray) -> Callable:
    M = 1 + (m + 1) + (m + 1) ** 2  # Total variables: zero-order + (m+1) primary + (m+1)^2 second-order

    def drift(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        drift_vector = jnp.zeros(M)
        drift_vector = drift_vector.at[0].set(0.0)  # Zero-order term remains constant
        drift_vector = drift_vector.at[1].set(1.0)  # Time variable derivative is 1 (dt/dt = 1)

        # Compute drift for primary variables (indices 2 to m + 1)
        for i in range(2, m + 2):  # m + 2 because range is exclusive at the end
            sum_a_y = jnp.dot(a[i], y)
            drift_vector = drift_vector.at[i].set(sum_a_y)

        # Compute drift for second-order variables (indices m + 2 to M - 1)
        for idx in range((m + 1) ** 2):
            x_idx, y_idx = divmod(idx, m + 1)
            i = m + 2 + idx  # Index of the second-order variable

            # Adjust indices for y: indices already include zero-order and time variable
            product_term = y[1 + x_idx] * drift_vector[1 + y_idx]
            sum_a_y = jnp.dot(a[i], y)
            drift_vector = drift_vector.at[i].set(product_term + sum_a_y)

        return drift_vector

    return drift

def diffusion_function_with_time(m: int, n: int, b: jnp.ndarray) -> Callable:
    M = 1 + (m + 1) + (m + 1) ** 2  # Total variables

    def diffusion(t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        diffusion_matrix = jnp.zeros((M, n))
        diffusion_matrix = diffusion_matrix.at[0, :].set(0.0)  # Zero-order term
        diffusion_matrix = diffusion_matrix.at[1, :].set(0.0)  # Time variable

        # Compute diffusion for primary variables (indices 2 to m + 1)
        for i in range(2, m + 2):
            for j in range(n):
                sum_b_y = jnp.dot(b[i, j], y)
                diffusion_matrix = diffusion_matrix.at[i, j].set(sum_b_y)

        # Compute diffusion for second-order variables (indices m + 2 to M - 1)
        for idx in range((m + 1) ** 2):
            x_idx, y_idx = divmod(idx, m + 1)
            i = m + 2 + idx  # Index of the second-order variable

            for j in range(n):
                product_term = y[1 + x_idx] * diffusion_matrix[1 + y_idx, j]
                sum_b_y = jnp.dot(b[i, j], y)
                diffusion_value = product_term + sum_b_y
                diffusion_matrix = diffusion_matrix.at[i, j].set(diffusion_value)

        return diffusion_matrix

    return diffusion
#
def verify_iterated_integrals(primary_variables: np.ndarray, time_steps: np.ndarray) -> np.ndarray:
    """
    Verifies level-2 iterated integrals using numerical integration.

    Parameters:
    primary_variables (np.ndarray): Array of shape (T, m+1) containing primary variable trajectories (including time).
    time_steps (np.ndarray): Array of shape (T,) containing time steps.

    Returns:
    np.ndarray: Computed iterated integrals of shape (T, (m+1)*(m+1)).
    """
    m_plus_one = primary_variables.shape[1]  # Includes time variable
    T = primary_variables.shape[0]
    iterated_integrals = np.zeros((T, (m_plus_one) ** 2))

    for idx in range((m_plus_one) ** 2):
        i, j = divmod(idx, m_plus_one)
        product_path = primary_variables[:, i] * np.gradient(primary_variables[:, j], time_steps)
        iterated_integrals[:, idx] = cumulative_trapezoid(product_path, time_steps, initial=0)

    return iterated_integrals

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
        # Removed stepsize_controller=None
    )

    return sol

if __name__ == "__main__":
    t_1 = time.time()
    # Parameters
    m = 2  # Number of primary variables (excluding time)
    n = 2  # Dimension of Brownian motion W_t
    M = 1 + (m + 1) + (m + 1)**2  # Total number of variables including time and second-order terms

    theta = 0.5  # Mean reversion rate
    sigma = 1   # Volatility for primary variable 1
    sigma_2 = 1  # Volatility for primary variable 2

    # Initialize the drift coefficient matrix a (M x M)
    a = jnp.zeros((M, M))

    # Set drift coefficients explicitly for visual interpretation
    # For m = 2, (m + 1) = 3, so M = 1 + 3 + 9 = 13
    # Let's construct 'a' as an explicit matrix
    a = jnp.array([
        # k=0    k=1     k=2       k=3       k=4  ... up to k=12 (M=13)
        [0.0] * M,  # i=0 (zero-order term)
        [0.0] * M,  # i=1 (time variable, drift is set directly in the function)
        [0.0] * M,  # i=2 (primary variable 1)
        [0.0] * M,  # i=3 (primary variable 2)
    ] + [
        [0.0] * M for _ in range(M - 4)  # Second-order variables
    ])

    # Set drift coefficients for primary variables
    a = a.at[2, 2].set(-theta)  # a[2, 2] = -theta for variable 1
    a = a.at[3, 3].set(-theta)  # a[3, 3] = -theta for variable 2

    # Similarly, you can set drift coefficients for second-order variables if needed

    a = a.at[2, 4].set(2)  # a[2, 2] = -theta for variable 1
    a = a.at[3, 5].set(-theta)  # a[3, 3] = -theta for variable 2

    # Initialize the diffusion coefficient tensor b (M x n x M)
    b = jnp.zeros((M, n, M))

    # Set diffusion coefficients explicitly for visual interpretation
    # Variable 1 (i = 2)
    b = b.at[2, 0, 0].set(sigma)  # b[2, 0, 0] = sigma (additive noise via zero-order term)

    # Variable 2 (i = 3)
    b = b.at[3, 1, 0].set(sigma_2)  # b[3, 1, 0] = sigma_2 (additive noise via zero-order term)

    # Initial condition X0, shape (M,)
    X0 = jnp.zeros(M)
    X0 = X0.at[0].set(1.0)  # Zero-order term initialized to 1
    X0 = X0.at[1].set(0.0)  # Time variable starts at t = 0

    # Create the drift and diffusion functions
    drift = drift_function_with_time(m, a)
    diffusion = diffusion_function_with_time(m, n, b)

    # Time parameters
    t0 = 0.0
    t1 = 2
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

    # Verify the shapes
    print(f"Number of time steps: {len(solution.ts)}")
    print(f"Shape of solution.ys: {solution.ys.shape}")

    # Function to plot strictly primary variables
    def plot_primary_variables(solution, m):
        t_values = np.array(solution.ts)
        y_values = np.array(solution.ys[:, 2:m+2])  # Extract primary variables (indices 2 to m+1)

        plt.figure(figsize=(10, 6))
        for i in range(y_values.shape[1]):
            plt.plot(t_values, y_values[:, i], label=f'Y_{i+1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Primary Variables Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot the primary variables
    plot_primary_variables(solution, m)

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

    # Extract primary variables and compute iterated integrals
    primary_variables = np.array(solution.ys[:, 1:m + 2])  # Indices 1 to m+1
    time_steps = np.array(solution.ts)
    iterated_integrals = verify_iterated_integrals(primary_variables, time_steps)

    # Extract second-order variables and reshape
    second_order_start_idx = m + 2  # Starting index for second-order terms
    second_order_variables = np.array(solution.ys[:, second_order_start_idx:])
    second_order_variables = second_order_variables.reshape(-1, (m + 1)**2)

    # Compute the absolute difference
    difference = np.abs(second_order_variables - iterated_integrals)
    max_difference = np.max(difference)
    print(f"Maximum difference between expanded system and numerical integration: {max_difference}")

    # Plot raw time series for expanded system and numerical integration
    plt.figure(figsize=(12, 8))
    for idx in range((m + 1)**2):
        i, j = divmod(idx, m + 1)
        plt.plot(time_steps, second_order_variables[:, idx], linestyle='-', label=f'Expanded Y_{i},{j}')
        plt.plot(time_steps, iterated_integrals[:, idx], linestyle='--', label=f'Numerical Y_{i},{j}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Comparison of Expanded System and Numerical Integration')
    plt.legend(ncol=2, loc='upper left', fontsize='small')
    plt.grid(True)
    plt.show()
