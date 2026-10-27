'''
Random Forest baseline for arithmetic
'''
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

import data
import utils #local

#parameters
n = 8
length = 2*n
samples = 32000
batch_size = 128
delta = 0.2

#===========
#data
X1f, X2f, Y_raw = data.generate_arithmetic(samples, "add", n, batch_size, delta)

#stack
X_preproc = np.concatenate([X1f, X2f], axis=-1)

X, X_test, Y, Y_test = train_test_split(X_preproc, Y_raw, test_size=0.2, random_state=42)
print("X shape", X.shape, "Y shape:", Y.shape, "Y test shape:", Y_test.shape)

Y_packed = np.packbits(Y,axis=2)
Y_test_packed = np.packbits(Y_test,axis=2)
print("Y_packed shape", Y_packed.shape)

# Flatten for scikit-learn
X_train_flat = X.reshape(-1, X.shape[-1])
Y_train_flat = Y.reshape(-1, Y.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

#===========
##create models
base_rf_params = dict(
    n_estimators=500, 
    criterion='entropy',
    random_state=42, 
    n_jobs=-1, 
    bootstrap=True, 
    max_features='sqrt',
    min_samples_split=5
)

models = {
    "Pure Random Forest": RandomForestClassifier(**base_rf_params),
    "MultiOutput RF": MultiOutputClassifier(RandomForestClassifier(**base_rf_params)),
    "Classifier Chain RF": ClassifierChain(RandomForestClassifier(**base_rf_params), random_state=42)
}

for name, model in models.items():
    print(f"\n=========================================")
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train_flat, Y_train_flat)
    end = time.time()
    elapsed = end - start
    print(f"Training took {elapsed:.2f} secs or {elapsed/60:.2f} mins in total")

    # Evaluation
    print(f"Evaluating {name}...")
    eval_start = time.time()
    predictions_flat = model.predict(X_test_flat)
    eval_end = time.time()
    eval_elapsed = eval_end - eval_start
    predictions_flat = predictions_flat.astype(np.uint8)

    # Reshape back to batches for evaluation like the original script
    predictions = predictions_flat.reshape(X_test.shape[0], X_test.shape[1], -1)
    Y_pred = np.packbits(predictions, axis=2)

    accuracy = np.mean(Y_pred == Y_test_packed)
    print(f"[{name} Final Accuracy {accuracy}]")

    # Performance metrics
    if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'estimators_'):
        # MultiOutput or ClassifierChain
        num_nodes = sum(tree.tree_.node_count for sub_model in model.estimators_ for tree in sub_model.estimators_)
    else:
        # Pure RF
        num_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        
    memory_bytes = len(pickle.dumps(model))
    throughput = X_test_flat.shape[0] / eval_elapsed

    print(f"--- {name} Performance Metrics ---")
    print(f"Total Tree Nodes: {num_nodes} (Proxy for parameters)")
    print(f"Memory Footprint: {memory_bytes / 1024 / 1024:.4f} MB")
    print(f"Inference Time: {eval_elapsed:.4f} secs")
    print(f"Throughput: {throughput:.2f} samples/sec")

print("\nEND")
