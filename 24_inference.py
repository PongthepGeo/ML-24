#-----------------------------------------------------------------------------------------#
from Libs.MLPs import MLPs
from Libs.utilities import preprocess_image, predict_image_direct
#-----------------------------------------------------------------------------------------#
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

test_image_path = 'data/channel/test_data/test.png'

#-----------------------------------------------------------------------------------------#

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
n_classes = 2  # Assuming binary classification
model = MLPs(n_classes)
model.load_state_dict(torch.load('best_model/best_model_iou_0.6244.pth'))
model = model.to(device)
model.eval()  # Set to evaluation mode

#-----------------------------------------------------------------------------------------#

image_tensor, original_shape = preprocess_image(test_image_path)
predictions = predict_image_direct(device, model, image_tensor, original_shape)

#-----------------------------------------------------------------------------------------#

plt.imshow(predictions, cmap='gray')
plt.title("Predicted Segmentation (Grayscale)")
plt.show()

#-----------------------------------------------------------------------------------------#