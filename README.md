Explanation:
Porous Structure Generation: Generates random porous structures as binary arrays saved as .npy files.
Heat Transfer Simulation: Uses the fipy library to simulate heat transfer across each structure and calculates an average temperature.
CNN Model: Defines and trains a CNN model to predict heat transfer efficiency based on the porous structure.
Training and Model Saving: The trained model is saved as optimized_cnn_model.pth for later use.

# porous_material_optimization.py
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from fipy import CellVariable, Grid2D, DiffusionTerm

# Configuration
num_samples = 100  # Number of porous structures to generate
grid_size = (50, 50)  # Size of each porous structure
output_dir = "project_data"
thermal_diffusivity = 0.1  # Diffusion coefficient for heat transfer simulation
L = 10.0  # Domain size for simulation
nx, ny = 50, 50  # Grid resolution
epochs = 200  # Number of training epochs
learning_rate = 0.001  # Learning rate for optimizer

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Generate Porous Structures
print("Generating porous structures...")
porous_structures = []
for i in range(num_samples):
    porous_structure = np.random.choice([0, 1], size=grid_size, p=[0.7, 0.3])
    porous_structures.append(porous_structure)
    np.save(os.path.join(output_dir, f"structure_{i}.npy"), porous_structure)
    plt.imshow(porous_structure, cmap="gray")
    plt.title(f"Porous Structure {i}")
    plt.savefig(os.path.join(output_dir, f"structure_{i}.png"))
    plt.close()
print(f"{num_samples} porous structures saved in '{output_dir}'.")

# 2. Simulate Heat Transfer for Each Structure
print("Simulating heat transfer...")
heat_transfer_data = []
for idx, porous_structure in enumerate(porous_structures):
    mesh = Grid2D(dx=L/nx, dy=L/ny, nx=nx, ny=ny)
    temperature = CellVariable(name="temperature", mesh=mesh, value=300.0)
    temperature.constrain(500.0, mesh.facesLeft)
    temperature.constrain(300.0, mesh.facesRight)
    
    heat_eq = DiffusionTerm(coeff=thermal_diffusivity)
    for step in range(100):
        heat_eq.solve(var=temperature, dt=1.0)
    
    avg_temperature = temperature.value.mean()
    heat_transfer_data.append(avg_temperature)
    np.save(os.path.join(output_dir, f"structure_{idx}_temp.npy"), temperature.value)

print("Heat transfer simulation completed.")

# 3. Define the CNN Model for Training
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate the output size after two Conv2d layers for a 50x50 input.
        # With kernel_size=3 and padding=1, 50x50 input remains 50x50 after each layer.
        # Downsample by 2 using pooling or reduce based on expected flattened size.
        self.fc1 = nn.Linear(32 * 50 * 50, 100)  # Updated to match correct output size
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 50 * 50)  # Updated based on new flattened size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare data for training
print("Preparing data for training...")
train_data = []
train_labels = []

for idx in range(num_samples):
    structure = np.load(os.path.join(output_dir, f"structure_{idx}.npy"))
    heat_transfer = heat_transfer_data[idx]
    train_data.append(structure)
    train_labels.append(heat_transfer)

train_data = torch.FloatTensor(train_data).unsqueeze(1)
train_labels = torch.FloatTensor(train_labels).unsqueeze(1)

# 4. Train the CNN Model
model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print("Training the CNN model...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the model
model_path = os.path.join(output_dir, "optimized_cnn_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model training completed. Model saved as '{model_path}'.")
print ("The End")

