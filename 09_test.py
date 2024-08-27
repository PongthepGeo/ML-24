#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../dataset/ch_04/rock_density.csv')

# Generate random densities
np.random.seed(42)
random_densities = [np.random.uniform(row['Density (min)'], row['Density (max)']) for _, row in df.iterrows()]
df['Random Density'] = random_densities

# Define the cost function (MSE)
def cost_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Prepare data for modeling
x = np.linspace(1200, 4000, len(df))  # Rock model feature
y = df['Random Density']  # Target values

# Manually guess parameters
guesses = [
    {'omega_0': 2000, 'omega_1': 0.02},
    {'omega_0': 2200, 'omega_1': 0.04},
    {'omega_0': 2400, 'omega_1': 0.06},
    {'omega_0': 2600, 'omega_1': 0.08},
    {'omega_0': 2800, 'omega_1': 0.10},
]

# Evaluate each guess
results = []

for guess in guesses:
    omega_0 = guess['omega_0']
    omega_1 = guess['omega_1']
    
    # Calculate predicted density using the guessed parameters
    predicted_density = omega_0 + omega_1 * x
    
    # Calculate the mean squared error
    mse = cost_function(y, predicted_density)
    
    # Store the results
    results.append({
        'omega_0': omega_0,
        'omega_1': omega_1,
        'MSE': mse
    })

# Display the results
result_df = pd.DataFrame(results)
print(result_df)

# Plotting the results to visualize
plt.figure(figsize=(10, 6))
for result in results:
    omega_0 = result['omega_0']
    omega_1 = result['omega_1']
    predicted_density = omega_0 + omega_1 * x
    plt.plot(x, predicted_density, label=f"omega_0={omega_0}, omega_1={omega_1}, MSE={result['MSE']:.2f}")

# Plot actual random densities for comparison
plt.scatter(x, y, color='black', label='Actual Random Density', alpha=0.5)
plt.xlabel('Rock Feature (x)')
plt.ylabel('Density')
plt.title('Manual Parameter Guessing and MSE Calculation')
plt.legend()
plt.show()