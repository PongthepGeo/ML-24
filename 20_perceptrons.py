import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load the data
data = pd.read_csv('tabular/clustered_image_data.csv')

# Separate features and labels
X = data[['Red_Channel', 'Green_Channel', 'Blue_Channel']].values
y = data['Cluster_Label'].values

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # Use float32 for RMSE

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader objects
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the Simple Perceptron with a final step function at inference
class SimplePerceptron(nn.Module):
    def __init__(self):
        super(SimplePerceptron, self).__init__()
        # Initialize weights and biases manually for 3 input features to 1 output
        self.weights = nn.Parameter(torch.randn(3, 1))  # 3 inputs, 1 output
        self.bias = nn.Parameter(torch.randn(1))        # Bias for the output

    def forward(self, x):
        # Continuous linear output for backpropagation
        return torch.matmul(x, self.weights).squeeze() + self.bias
    
    def predict(self, x):
        # Apply step function only for binary output interpretation
        continuous_output = self.forward(x)
        return torch.where(continuous_output > 0, torch.tensor(1.0), torch.tensor(0.0))

# Instantiate the model
model = SimplePerceptron()

# RMSE as the loss function
def rmse_loss(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean())

# Training parameters
learning_rate = 0.01
epochs = 10

# Training loop with gradient descent
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # Forward pass with continuous output
        predictions = model(X_batch)
        loss = rmse_loss(predictions, y_batch)
        
        # Backward pass and gradient descent
        loss.backward()  # Compute gradients
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad  # Update parameters
                param.grad.zero_()  # Zero gradients after each update
        
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            # Use predict method to apply step function
            predictions = model.predict(X_batch)
            val_loss += rmse_loss(predictions, y_batch).item()
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")