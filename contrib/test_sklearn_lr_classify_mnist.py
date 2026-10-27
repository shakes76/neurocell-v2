'''
Logistic Regression (Linear) baseline for MNIST classification
'''
import numpy as np
import time
import pickle
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

#parameters
batch_size = 64
path = 'F:/'

#===========
#transform preproc
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)), transforms.Lambda(lambda x: x.permute(1, 2, 0))])

##data
print("Loading data via PyTorch...")
trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root=path+'data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

num_classes = 10
length = 28*28

# Extract all batches into full numpy arrays for scikit-learn
print("Extracting train set...")
X_train_list, Y_train_list = [], []
for x, y in tqdm(train_loader):
    x = x.numpy().reshape((x.shape[0], length))
    X_train_list.append(x)
    Y_train_list.append(y.numpy())
X_train = np.vstack(X_train_list)
Y_train = np.concatenate(Y_train_list)

print("Extracting test set...")
X_test_list, Y_test_list = [], []
for x, y in tqdm(test_loader):
    x = x.numpy().reshape((x.shape[0], length))
    X_test_list.append(x)
    Y_test_list.append(y.numpy())
X_test = np.vstack(X_test_list)
Y_test = np.concatenate(Y_test_list)

print("Data Shape:", X_train.shape, "Samples:", X_train.shape[0], "Input Length:", length, "Classes:", num_classes)

#===========
##create model
print("\nTraining Logistic Regression...")
# Using multinomial LogisticRegression which acts as a linear classifier
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, n_jobs=-1, random_state=42)

start = time.time()
model.fit(X_train, Y_train)
end = time.time()
elapsed = end - start
print(f"Training took {elapsed:.2f} secs or {elapsed/60:.2f} mins in total")

# Evaluation
print("\nEvaluating...")
eval_start = time.time()
predictions = model.predict(X_test)
eval_end = time.time()
eval_elapsed = eval_end - eval_start

accuracy = np.mean(predictions == Y_test)
print(f"[Final Test Accuracy {accuracy * 100:.2f}%]")

# Performance metrics
num_params = model.coef_.size + model.intercept_.size
memory_bytes = len(pickle.dumps(model))
throughput = X_test.shape[0] / eval_elapsed

print(f"\n--- Performance Metrics ---")
print(f"Total Parameters: {num_params}")
print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
print(f"Inference Time: {eval_elapsed:.4f} secs")
print(f"Throughput: {throughput:.2f} samples/sec")

print("\nEND")
