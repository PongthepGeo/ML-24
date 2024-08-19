#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
#-----------------------------------------------------------------------------------------#

df = pd.read_csv('../dataset/ch_04/rock_density.csv')

#-----------------------------------------------------------------------------------------#

# Generate random samples between min and max density for each rock type
np.random.seed(42)  # For reproducibility
random_densities = [] # Prepare an empty list to store the random densities

for index, row in df.iterrows():
    min_density = row['Density (min)']
    max_density = row['Density (max)']
    random_density = np.random.uniform(min_density, max_density)
    random_densities.append(random_density)
df['Random Density'] = random_densities
# print(f'QC the random densities \n {df.head()}')

#-----------------------------------------------------------------------------------------#

# U.plot_rock(df)

# #-----------------------------------------------------------------------------------------#

omega_0 = 2500  # Intercept
omega_1 = 0.05  # Slope 
x = np.linspace(1200, 4000, len(df)) # Rock model
predicted_density = omega_0 + omega_1 * x
# print(f'QC the predicted density \n {predicted_density}')

#-----------------------------------------------------------------------------------------#

U.plot_rock(df, predicted_density=predicted_density)

#-----------------------------------------------------------------------------------------#