import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

# Load image and convert to NumPy array
img = Image.open('data/gabbro.jpg')
img_np = np.array(img)

# Reshape the image to a 2D array of pixels for clustering
pixels = img_np.reshape(-1, 3)  # Shape: (H*W, 3)

# Apply K-means clustering to classify pixels into 3 clusters
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

# Assign each pixel to a cluster with labels starting from 0 (0, 1, or 2)
flattened_labels = kmeans.labels_

# Flatten each RGB channel for concatenation
red_channel = img_np[:, :, 0].reshape(-1)
green_channel = img_np[:, :, 1].reshape(-1)
blue_channel = img_np[:, :, 2].reshape(-1)

# Create a DataFrame to hold the flattened data
df = pd.DataFrame({
    'Cluster_Label': flattened_labels,
    'Red_Channel': red_channel,
    'Green_Channel': green_channel,
    'Blue_Channel': blue_channel
})

# Save the DataFrame to a CSV file
csv_path = 'tabular/clustered_image_data.csv'
df.to_csv(csv_path, index=False)

print(f"CSV file saved at {csv_path}")