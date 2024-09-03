#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('../dataset/ch_04/rock_density.csv')
print(df.columns) # Check column names
guesses = [
    {'omega_0': 2000, 'omega_1': 0.02, 'color': 'blue'},
    {'omega_0': 2200, 'omega_1': 0.04, 'color': 'green'},
    {'omega_0': 2400, 'omega_1': 0.06, 'color': 'red'},
    {'omega_0': 2600, 'omega_1': 0.08, 'color': 'purple'},
    {'omega_0': 3200, 'omega_1': -50., 'color': 'black'}
]

#-----------------------------------------------------------------------------------------#

np.random.seed(42)  # For reproducibility
random_densities = []
for index, row in df.iterrows():
    min_density = row['Density (min)']
    max_density = row['Density (max)']
    random_density = np.random.uniform(min_density, max_density)
    random_densities.append(random_density)
df['Random Density'] = random_densities

#-----------------------------------------------------------------------------------------#

U.plot_cost_function(df, guesses)

#-----------------------------------------------------------------------------------------#