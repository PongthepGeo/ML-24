#-----------------------------------------------------------------------------------------#
from Libs.MLPs import MLPs
#-----------------------------------------------------------------------------------------#
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
#-----------------------------------------------------------------------------------------#

data = pd.read_csv('tabular/clustered_image_data.csv')
batch_size = 32

#-----------------------------------------------------------------------------------------#

X = data[['Red_Channel', 'Green_Channel', 'Blue_Channel']].values
y = data['Cluster_Label'].values
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Ensure class labels are of type LongTensor
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#-----------------------------------------------------------------------------------------#

# Initialize model, loss function, and optimizer
n_classes = len(data['Cluster_Label'].unique())  # Number of classes
model = MLPs(n_classes)
# Move model to GPU if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
model = model.to(device)
# CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#-----------------------------------------------------------------------------------------#

# Training and validation loop
for epoch in range(10):  # Specify the number of epochs as needed
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Ensure both are on the same device
        # Forward pass
        outputs = model(batch_X)  # No softmax during training
        loss = criterion(outputs, batch_y)  # Ensure batch_y is LongTensor
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation (no softmax here as well)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Again, ensure LongTensor and correct device
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    print(f"Epoch [{epoch+1}/10], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

#-----------------------------------------------------------------------------------------#