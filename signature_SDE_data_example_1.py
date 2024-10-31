import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha_21 = 0.5  # Change this value as needed
alpha_1 = 0.3   # Change this value as needed
T = 1.0         # Total time
N = 1000        # Number of time steps
dt = T / N      # Time step size

# Time grid
t = np.linspace(0, T, N+1)

# Initialize X^(1) and X^(2)
X1 = np.zeros(N+1)
X2 = np.zeros(N+1)

# Generate random increments for the Wiener process
dW1 = np.random.normal(0, np.sqrt(dt), N)
dW2 = np.random.normal(0, np.sqrt(dt), N)

# Simulate X^(1) using the Euler-Maruyama method
for i in range(N):
    X1[i+1] = X1[i] + alpha_1 * X1[i] * dt + dW1[i]

# Simulate X^(2) using the Euler-Maruyama method
for i in range(N):
    integral_term = np.sum(X2[:i+1] * (X1[1:i+2] - X1[:i+1])) * dt
    X2[i+1] = X2[i] + alpha_21 * integral_term + dW2[i]

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(t, X1, label='$X^{(1)}_t$')
plt.plot(t, X2, label='$X^{(2)}_t$', linestyle='--')
plt.xlabel('Time $t$')
plt.ylabel('$X_t$')
plt.title('Simulation of $X^{(1)}_t$ and $X^{(2)}_t$')
plt.legend()
plt.grid(True)
plt.show()
