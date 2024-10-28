#-----------------------------------------------------------------------------------------#
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

# Load image and convert to NumPy array
img = Image.open('data/gabbro.jpg')
img_np = np.array(img)
# Convert NumPy array to PyTorch tensor and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
image_tensor = torch.from_numpy(img_np).to(device)
# Rearrange dimensions from (H, W, C) to (C, H, W) for PyTorch and verify shape
image_tensor = image_tensor.permute(2, 0, 1)
print(f"Image tensor shape after permutation: {image_tensor.shape}")
# Convert tensor back to (H, W, C) for plotting with matplotlib and move to CPU
image_for_plot = image_tensor.permute(1, 2, 0).cpu().numpy()

# Plot the image
plt.imshow(image_for_plot)
plt.axis('off')
plt.show()

#-----------------------------------------------------------------------------------------#