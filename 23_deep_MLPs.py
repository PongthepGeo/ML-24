#-----------------------------------------------------------------------------------------#
from Libs.MLPs import MLPs
#-----------------------------------------------------------------------------------------#
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
#-----------------------------------------------------------------------------------------#

data = pd.read_csv('tabular/channel_data_with_labels.csv')
batch_size = 128
lr = 1e-4
epochs = 20

#-----------------------------------------------------------------------------------------#

X = data[['Red_Channel', 'Green_Channel', 'Blue_Channel']].values
# Normalize RGB channels to [0, 1] range
X = X / 255.0
y = data['Label'].values
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Ensure class labels are of type LongTensor
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#-----------------------------------------------------------------------------------------#

# Initialize model, loss function, and optimizer
n_classes = len(data['Label'].unique())  # Number of classes
# print(n_classes)
model = MLPs(n_classes)
# Specify device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
model = model.to(device)

# Use CrossEntropyLoss for multi-class (or binary) classification
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=lr)

# Ensure save directory exists
save_dir = 'best_model'
os.makedirs(save_dir, exist_ok=True)
best_val_iou = 0.0  # Initialize the best validation IoU

# IoU calculation function
def calculate_iou(preds, labels, threshold=0.5):
    preds = (preds > threshold).float()  # Convert logits/probabilities to binary predictions
    intersection = torch.sum(preds * labels)
    union = torch.sum(preds) + torch.sum(labels) - intersection
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    return iou

# Training and validation loop
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)  # Raw logits without softmax
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation with IoU calculation
    model.eval()
    val_loss = 0.0
    total_iou = 0.0  # Track total IoU across batches
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            # Calculate IoU for this batch
            predictions = torch.argmax(outputs, dim=1)  # Get class predictions
            iou = calculate_iou(predictions, batch_y)
            total_iou += iou.item()

    # Calculate the average validation loss and IoU
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = total_iou / len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")

    # Save the model if the validation IoU is the highest we've seen
    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        best_model_path = os.path.join(save_dir, f'best_model_iou_{best_val_iou:.4f}.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with IoU: {best_val_iou:.4f}")

#-----------------------------------------------------------------------------------------#
