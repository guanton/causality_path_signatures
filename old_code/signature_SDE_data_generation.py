import numpy as np
import matplotlib.pyplot as plt
import pickle


def simulate_sde(alpha_1_1, alpha_1_2, alpha_1_11, alpha_1_22, alpha_1_12, alpha_1_21,
                 alpha_2_1, alpha_2_2, alpha_2_11, alpha_2_22, alpha_2_12, alpha_2_21, T, N, integral_convention='ito'):
    dt = T / N  # Time step size
    t = np.linspace(0, T, N + 1)  # Time grid

    # Initialize X^(1) and X^(2)
    X1 = np.zeros(N + 1)
    X2 = np.zeros(N + 1)

    # Generate random increments for the Wiener process
    dW1 = np.random.normal(0, np.sqrt(dt), N)
    dW2 = np.random.normal(0, np.sqrt(dt), N)

    for i in range(N):
        # Calculate level 1 and level 2 integral terms via EM
        integral_1 = X1[i] * dt  # Integral \int_0^t X^(1)_s ds
        integral_2 = X2[i] * dt  # Integral \int_0^t X^(2)_s ds
        integral_12 = X1[i] * X2[i] * dt  # (1,2) multi-index
        integral_21 = X2[i] * X1[i] * dt  # (2,1) multi-index
        integral_11 = X1[i] ** 2 * dt  # (1,1) multi-index
        integral_22 = X2[i] ** 2 * dt  # (2,2) multi-index
        if integral_convention == 'stratonovich':
            # Implement Stratonovich to Ito correction
            correction_1 = 0.5 * X1[i] * dt
            integral_12 = integral_12 + correction_1
            integral_11 = integral_11 + correction_1
            correction_2 = 0.5 * X2[i] * dt
            integral_21 = integral_21 + correction_2
            integral_22 = integral_22 + correction_2
            X1[i + 1] = X1[i] + alpha_1_1 * integral_1 + alpha_1_2 * integral_2 \
                        + alpha_1_11 * integral_11 + alpha_1_22 * integral_22 \
                        + alpha_1_12 * integral_12 + alpha_1_21 * integral_21 + dW1[i]

            X2[i + 1] = X2[i] + alpha_2_1 * integral_1 + alpha_2_2 * integral_2 \
                        + alpha_2_11 * integral_11 + alpha_2_22 * integral_22 \
                        + alpha_2_12 * integral_12 + alpha_2_21 * integral_21 + dW2[i]
        else:
            X1[i + 1] = X1[i] + alpha_1_1 * integral_1 + alpha_1_2 * integral_2 \
                        + alpha_1_11 * integral_11 + alpha_1_22 * integral_22 \
                        + alpha_1_12 * integral_12 + alpha_1_21 * integral_21 + dW1[i]

            X2[i + 1] = X2[i] + alpha_2_1 * integral_1 + alpha_2_2 * integral_2 \
                        + alpha_2_11 * integral_11 + alpha_2_22 * integral_22 \
                        + alpha_2_12 * integral_12 + alpha_2_21 * integral_21 + dW2[i]

    return t, X1, X2

def save_simulation_data(filename, t, X1, X2, params):
    """
    Save the simulation data and parameters to a file.

    Parameters:
    - filename: Name of the file to save the data.
    - t: Array of time points.
    - X1: Simulated data for X^(1).
    - X2: Simulated data for X^(2).
    - params: Dictionary of parameters (coefficients).
    """
    data = {
        'time': t,
        'X1': X1,
        'X2': X2,
        'parameters': params
    }

    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

# Parameters (can be adjusted or made dynamic)
params = {
    'alpha_1_1': 0,  # Coefficient for multi-index (1) for variable 1
    'alpha_1_2': 0,  # Coefficient for multi-index (2) for variable 1
    'alpha_1_11': 0,  # Coefficient for multi-index (1,1) for variable 1
    'alpha_1_22': 0,  # Coefficient for multi-index (2,2) for variable 1
    'alpha_1_12': 1,  # Coefficient for multi-index (1,2) for variable 1
    'alpha_1_21': 0,  # Coefficient for multi-index (2,1) for variable 1
    'alpha_2_1': 0,  # Coefficient for multi-index (1) for variable 2
    'alpha_2_2': -2,  # Coefficient for multi-index (2) for variable 2
    'alpha_2_11': 0,  # Coefficient for multi-index (1,1) for variable 2
    'alpha_2_22': 0,  # Coefficient for multi-index (2,2) for variable 2
    'alpha_2_12': 0,  # Coefficient for multi-index (1,2) for variable 2
    'alpha_2_21': 0  # Coefficient for multi-index (2,1) for variable 2
}
n_seed = 0
np.random.seed(n_seed)
T = 2.0  # Total time
N = 1000  # Number of time steps

# Simulate the system with Ito convention
t_ito, X1_ito, X2_ito = simulate_sde(**params, T=T, N=N, integral_convention='ito')

# Simulate the system with Stratonovich convention
t_strat, X1_strat, X2_strat = simulate_sde(**params, T=T, N=N, integral_convention='stratonovich')

# Plotting the results to compare Ito and Stratonovich
plt.figure(figsize=(12, 6))
plt.plot(t_ito, X1_ito, label='Ito $X^{(1)}_t$', linestyle='-', color='b')
plt.plot(t_ito, X2_ito, label='Ito $X^{(2)}_t$', linestyle='--', color='b')
plt.plot(t_strat, X1_strat, label='Stratonovich $X^{(1)}_t$', linestyle='-', color='r')
plt.plot(t_strat, X2_strat, label='Stratonovich $X^{(2)}_t$', linestyle='--', color='r')
plt.xlabel('Time $t$')
plt.ylabel('$X_t$')
plt.title('Comparison of Ito and Stratonovich Integrals for $X^{(1)}_t$ and $X^{(2)}_t$')
plt.legend()
plt.grid(True)
plt.show()
