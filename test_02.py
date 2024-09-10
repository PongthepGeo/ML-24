import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from matplotlib.colors import ListedColormap

# Read the CSV file
df = pd.read_csv('/home/pongthep/Desktop/feature_soilnutrients_VI.csv')

# Use LabelEncoder to convert string labels to integers
label_encoder = LabelEncoder()
df['feature_soil nutrients'] = label_encoder.fit_transform(df['feature_soil nutrients'])

# Find the number of unique labels
unique_labels = df['feature_soil nutrients'].nunique()

# Extract data for different features
ndvi = df['NDVI'].values
ndre = df['NDRE'].values
gndvi = df['GNDVI'].values
osavi = df['OSAVI'].values
labels = df['feature_soil nutrients'].values

# Define a discrete colormap using colormaps.get_cmap()
base_cmap = plt.colormaps.get_cmap('tab10')  # Updated method
colors = base_cmap(np.linspace(0, 1, unique_labels))  # Generate colors
cmap = ListedColormap(colors)  # Create a ListedColormap

# Concatenate all feature values and corresponding labels
x_values = np.concatenate([ndvi, ndre, gndvi, osavi])
y_values = np.concatenate([np.full(len(ndvi), 0), np.full(len(ndre), 1), np.full(len(gndvi), 2), np.full(len(osavi), 3)])
soil_labels = np.concatenate([labels, labels, labels, labels])

# Plot setup
plt.figure(figsize=(12, 6))
scatter = plt.scatter(x_values, y_values, c=soil_labels, cmap=cmap, edgecolor='k', s=50)

# Define y-axis labels to feature names
plt.yticks(ticks=[0, 1, 2, 3], labels=['NDVI', 'NDRE', 'GNDVI', 'OSAVI'])
plt.xlabel('Feature Values')
plt.ylabel('Features')
plt.title('Concatenated Feature Values with Soil Nutrient Labels')

# Add a color bar with labels for soil nutrient categories
cbar = plt.colorbar(scatter)
cbar.set_label('Soil Nutrient Categories')
cbar.set_ticks(np.arange(unique_labels))
cbar.set_ticklabels(label_encoder.inverse_transform(np.arange(unique_labels)))

# Show grid for better visualization
plt.grid(True)
plt.show()
