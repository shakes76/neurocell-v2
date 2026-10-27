'''
Solves XOR learning problem comparing traditional MLP to VNN MLP
'''
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time
import pickle

import data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Parameters matching VNN arithmetic pipeline
epochs = 100
n = 8
length = 2 * n
samples = 32000
batch_size = 128
delta = 0.2
hidden_size = 128

# Data Pipeline
print(f"Generating data: samples={samples}, n={n}, delta={delta}")
X1f, X2f, Y_raw = data.generate_arithmetic(samples, "xor", n, batch_size, delta)
X_preproc = np.concatenate([X1f, X2f], axis=-1)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)

# Packbits for exact match accuracy evaluation (numpy)
Y_test_packed = np.packbits(Y_test, axis=2)

# Convert to PyTorch Tensors
X = torch.Tensor(X)
Y = torch.Tensor(Y)
X_test = torch.Tensor(X_test)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("X_test shape:", X_test.shape)

# Model: Extended to 4 layers
class XORNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XORNet, self).__init__()
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Using tanh to match the VNN internal activations
        x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

model = XORNet(length, hidden_size, n)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

#--------------
# Train the model
model.train()
print("> Training")
train_start = time.time()
for epoch in range(epochs):
    loss_value = 0
    for i, (x, labels) in enumerate(zip(X, Y)):
        x = x.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Matching the metric scaling used in the VNN script logging
        loss_value += loss.item() / batch_size

    if (epoch + 1) % 10 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss_value:.4f}")

train_end = time.time()
print(f"Training took {train_end - train_start:.2f} seconds.")

# Test the model
print("> Testing")
model.eval()
with torch.no_grad():
    eval_start = time.time()
    
    Y_pred = []
    # X_test has shape (num_batches_test, batch_size, length)
    for x in X_test:
        x = x.to(device)
        outputs = model(x)
        
        predicted = torch.round(outputs).cpu().numpy().astype(np.uint8)
        predicted_packed = np.packbits(predicted, axis=1)
        Y_pred.append(predicted_packed)
        
    eval_end = time.time()
    eval_elapsed = eval_end - eval_start

    Y_pred = np.array(Y_pred)
    accuracy = np.mean(Y_pred == Y_test_packed)
    print(f"Final Accuracy: {accuracy:.4f}")

# Performance metrics
param_count = sum(p.numel() for p in model.parameters())
memory_bytes = len(pickle.dumps(model))
# Total samples in test set
total_samples = X_test.shape[0] * X_test.shape[1]
throughput = total_samples / eval_elapsed

print(f"\n--- Performance Metrics ---")
print(f"Total Parameters: {param_count}")
print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
print(f"Inference Time: {eval_elapsed:.4f} secs")
print(f"Throughput: {throughput:.2f} samples/sec")

print("END")
