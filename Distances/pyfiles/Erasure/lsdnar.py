"""
leace_synt_dist.py

This script performs LEACE (Linear Concept Erasure) on GPT-2 embeddings to erase the concept of syntactic (tree) distances between words.
It loads precomputed embeddings and their corresponding pairwise syntactic distance matrices (as produced by encode_synt_distances.py),
flattens all word pairs into a regression dataset, and applies LEACE to remove information about syntactic distances from the embeddings.

The script then evaluates a linear regression probe's ability to predict syntactic distances from the embeddings, both before and after erasure,
and reports the mean squared error (MSE) on a held-out test set. The erased embeddings and the LEACE eraser object are saved for further analysis.

Inputs:
    - GPT-2 embeddings with syntactic distance matrices for train and test sets

Outputs:
    - Erased embeddings (train and test)
    - LEACE eraser object
    - JSON file with MSE before and after erasure, probed with a linear regression model
"""

import pickle
import torch
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from concept_erasure import LeaceEraser
import os
import json

# --------------------------
# Settings: Erasure of syntactic distances
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
RESULTS_FILE = "Distances/Results/Narratives/leace_results_synt_dist.json"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

BATCH_SIZE = 10000  # You can adjust this

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

# Flatten all word embeddings and all pairwise distances
def flatten_embeddings_and_distances(data):
    X = []
    Y = []
    for sent in data:
        emb = sent["embeddings_by_layer"][LAYER]  # [num_words, hidden_dim]
        dist = sent["distance_matrix"]            # [num_words, num_words]
        n = emb.shape[0]
        # For each pair (i, j), use (emb_i, emb_j) as input, dist[i, j] as label
        for i in range(n):
            for j in range(n):
                X.append(torch.cat([emb[i], emb[j]]).numpy())  # Concatenate embeddings
                Y.append(dist[i, j].item())
    return np.stack(X), np.array(Y)

print("Flattening training data...")
X_train, y_train = flatten_embeddings_and_distances(data)
print("Flattening test data...")
X_test, y_test = flatten_embeddings_and_distances(test_data)

print(f"Train pairs: {X_train.shape[0]}, Test pairs: {X_test.shape[0]}")
print(f"Embedding dim: {X_train.shape[1]}")

# --------------------------
# Batch generator
# --------------------------
def batch_generator(data, layer, batch_size=BATCH_SIZE):
    for sent in data:
        emb = sent["embeddings_by_layer"][layer]  # [num_words, hidden_dim]
        dist = sent["distance_matrix"]            # [num_words, num_words]
        n = emb.shape[0]
        pairs = [(i, j) for i in range(n) for j in range(n)]
        X_batch, Y_batch = [], []
        for i, j in pairs:
            val = dist[i, j].item()
            if np.isfinite(val):
                X_batch.append(torch.cat([emb[i], emb[j]]).numpy())
                Y_batch.append(val)
            # else: skip this pair
            if len(X_batch) == batch_size:
                yield np.stack(X_batch), np.array(Y_batch)
                X_batch, Y_batch = [], []
        if X_batch:
            yield np.stack(X_batch), np.array(Y_batch)

# --------------------------
# Standardize (fit on batches)
# --------------------------
scaler = StandardScaler()
for X_batch, _ in batch_generator(data, LAYER):
    scaler.partial_fit(X_batch)
print("Scaler fitted.")

# --------------------------
# Fit probe before LEACE (SGDRegressor, batch)
# --------------------------
probe = SGDRegressor(max_iter=1, tol=None, warm_start=True)
first = True
for X_batch, y_batch in batch_generator(data, LAYER):
    X_batch_scaled = scaler.transform(X_batch)
    if first:
        probe.partial_fit(X_batch_scaled, y_batch)
        first = False
    else:
        probe.partial_fit(X_batch_scaled, y_batch)
print("SGDRegressor probe fitted.")

# Evaluate on test set in batches
y_test_true = []
y_test_pred = []
for X_batch, y_batch in batch_generator(test_data, LAYER):
    X_batch_scaled = scaler.transform(X_batch)
    y_pred = probe.predict(X_batch_scaled)
    y_test_true.append(y_batch)
    y_test_pred.append(y_pred)
y_test_true = np.concatenate(y_test_true)
y_test_pred = np.concatenate(y_test_pred)
mse_before = mean_squared_error(y_test_true, y_test_pred)
print(f"MSE on test set BEFORE erasure: {mse_before:.4f}")

# --------------------------
# LEACE: fit on batches
# --------------------------
# Collect all Z (distance labels) for LEACE
Z_train = []
X_train_scaled = []
for X_batch, y_batch in batch_generator(data, LAYER):
    X_train_scaled.append(scaler.transform(X_batch))
    Z_train.append(y_batch.reshape(-1, 1))
X_train_scaled = np.concatenate(X_train_scaled)
Z_train = np.concatenate(Z_train)

eraser = LeaceEraser.fit(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(Z_train, dtype=torch.float32))

# Apply erasure and probe again in batches
probe_erased = SGDRegressor(max_iter=1, tol=None, warm_start=True)
first = True
for X_batch, y_batch in batch_generator(data, LAYER):
    X_batch_scaled = scaler.transform(X_batch)
    X_batch_erased = eraser(torch.tensor(X_batch_scaled, dtype=torch.float32)).numpy()
    if first:
        probe_erased.partial_fit(X_batch_erased, y_batch)
        first = False
    else:
        probe_erased.partial_fit(X_batch_erased, y_batch)
print("SGDRegressor probe (after erasure) fitted.")

# Evaluate on test set in batches (after erasure)
y_test_true = []
y_test_pred_erased = []
for X_batch, y_batch in batch_generator(test_data, LAYER):
    X_batch_scaled = scaler.transform(X_batch)
    X_batch_erased = eraser(torch.tensor(X_batch_scaled, dtype=torch.float32)).numpy()
    y_pred = probe_erased.predict(X_batch_erased)
    y_test_true.append(y_batch)
    y_test_pred_erased.append(y_pred)
y_test_true = np.concatenate(y_test_true)
y_test_pred_erased = np.concatenate(y_test_pred_erased)
mse_after = mean_squared_error(y_test_true, y_test_pred_erased)
print(f"MSE on test set AFTER erasure: {mse_after:.4f}")

# --------------------------
# Save eraser object and results (no need to save all erased embeddings)
# --------------------------
os.makedirs(os.path.dirname("Distances/Eraser_objects/Narratives/leace_eraser_synt_dist.pkl"), exist_ok=True)
with open("Distances/Eraser_objects/Narratives/leace_eraser_synt_dist.pkl", "wb") as f:
    pickle.dump(eraser, f)

results = {
    "mse_before": mse_before,
    "mse_after": mse_after,
}
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE}")
