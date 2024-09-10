#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('../dataset/ch_04/rock_density.csv')
bias_range = np.linspace(1500, 3500, 1000)
weight_range = np.linspace(-100, 100, 1000)
save_weight = 'data_out/weight_range.npy'
save_bias = 'data_out/bias_range.npy'

#-----------------------------------------------------------------------------------------#

np.random.seed(42)
random_densities = []
for index, row in df.iterrows():
    min_density = row['Density (min)']
    max_density = row['Density (max)']
    random_density = np.random.uniform(min_density, max_density)
    random_densities.append(random_density)
df['Random Density'] = random_densities

#-----------------------------------------------------------------------------------------#

rmse_matrix = np.zeros((len(bias_range), len(weight_range)))
x_numeric = np.arange(len(df))
y_actual = df['Random Density']

for i, omega_0 in enumerate(bias_range):
    for j, omega_1 in enumerate(weight_range):
        predicted_density = omega_0 + omega_1 * x_numeric
        rmse = U.cost_function(y_actual, predicted_density)
        rmse_matrix[i, j] = rmse
np.save(save_weight, rmse_matrix)
np.save(save_bias, bias_range)

#-----------------------------------------------------------------------------------------#

U.plot_cross_sections(rmse_matrix, bias_range, weight_range)

#-----------------------------------------------------------------------------------------#
