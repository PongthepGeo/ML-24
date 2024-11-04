import numpy as np
import pandas as pd
from PIL import Image
import os

# Load the images
img = Image.open('data/channel/raw_data/image.png')
label = Image.open('data/channel/raw_data/label.png')

# Check and match image sizes
if img.size != label.size:
    print(f"Resizing label from {label.size} to {img.size}")
    label = label.resize(img.size, Image.NEAREST)

# Convert to numpy arrays and remove the alpha channel
img = np.array(img)[:, :, :3]  # Remove alpha channel for RGB
label = np.array(label)[:, :, :3]  # Remove alpha channel for RGB in label as well

# Define color mappings for the labels
background = [255, 255, 255]  # RGB for background
waterbody = [0, 0, 255]       # RGB for waterbody

# Create a function to map color to label
def map_color_to_label(pixel):
    if np.array_equal(pixel, background):
        return 0  # Label for background
    elif np.array_equal(pixel, waterbody):
        return 1  # Label for waterbody
    else:
        return -1  # Unknown or ignore other colors if any

# Map the label image to numeric labels
label_mapped = np.apply_along_axis(map_color_to_label, 2, label)  # Only use RGB for mapping

# Flatten image and label arrays
img_flat = img.reshape(-1, 3)
label_flat = label_mapped.flatten()

# Filter out unknown labels if necessary
valid_indices = label_flat >= 0
img_flat = img_flat[valid_indices]
label_flat = label_flat[valid_indices]

# Create DataFrame
df = pd.DataFrame(img_flat, columns=['Red_Channel', 'Green_Channel', 'Blue_Channel'])
df['Label'] = label_flat

# Create the 'tabular' directory if it doesn't exist
os.makedirs('tabular', exist_ok=True)

# Save to CSV in the 'tabular' folder
df.to_csv('tabular/channel_data_with_labels.csv', index=False)

print("CSV file saved to 'tabular/channel_data_with_labels.csv'")

