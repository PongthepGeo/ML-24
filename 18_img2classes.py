import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image and convert to NumPy array
img = Image.open('data/gabbro.jpg')
img_np = np.array(img)

# Reshape the image to a 2D array of pixels for clustering
pixels = img_np.reshape(-1, 3)  # Shape: (H*W, 3)

# Apply K-means clustering with scikit-learn to classify pixels into 3 clusters
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

# Assign each pixel to a cluster (1, 2, or 3)
class_labels = kmeans.labels_ + 1  # Shift labels to 1-based indexing (1, 2, 3)
labeled_image = class_labels.reshape(img_np.shape[0], img_np.shape[1])

# Define colors corresponding to each class
class_colors = {
    1: [255, 0, 0],   # Red
    2: [0, 255, 0],   # Green
    3: [0, 0, 255]    # Blue
}

# Create an RGB image based on class labels
colored_image = np.zeros((*labeled_image.shape, 3), dtype=np.uint8)
for label, color in class_colors.items():
    colored_image[labeled_image == label] = color

# Display the single-channel labeled image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Single-Channel Class Labels")
plt.imshow(labeled_image, cmap='viridis', vmin=1, vmax=3)
plt.axis('off')
plt.colorbar(ticks=[1, 2, 3], label="Class Label (1=Red, 2=Green, 3=Blue)")

# Display the color-mapped image
plt.subplot(1, 2, 2)
plt.title("Color-Mapped Image")
plt.imshow(colored_image)
plt.axis('off')
plt.show()