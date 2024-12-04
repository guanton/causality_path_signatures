import numpy as np
import matplotlib.pyplot as plt

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

def plot_differences(df):
    levels = sorted(df['level'].unique())
    for level in levels:
        df_level = df[df['level'] == level]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df_level)), df_level['difference'].abs(), tick_label=df_level['multi_index'])
        plt.title(f'Absolute Differences at Level {level}')
        plt.xlabel('Multi-Index')
        plt.ylabel('Absolute Difference')
        plt.xticks(rotation=90)
        plt.tight_layout()
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

def plot_iterated_integrals(iterated_integrals_higher, higher_order_variables, min_length, time_steps):
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