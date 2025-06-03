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
    - JSON file with MSE before and after erasure
"""

import pickle
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from concept_erasure import LeaceEraser
import os
import json

# --------------------------
# Settings: Erasure of syntactic distances
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Distances/Embeddings/UD/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Distances/Embeddings/UD/gpt2_embeddings_test_synt_dist.pt"
RESULTS_FILE = "Distances/Results/leace_results_synt_dist.json"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

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
# Standardize
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Fit probe before LEACE (linear regression)
# --------------------------
probe = LinearRegression()
probe.fit(X_train_scaled, y_train)
y_pred_test = probe.predict(X_test_scaled)
mse_before = mean_squared_error(y_test, y_pred_test)
print(f"MSE on test set BEFORE erasure: {mse_before:.4f}")

# --------------------------
# Prepare labels for LEACE: use one-hot binning of distances (optional)
# For full erasure, you can use the raw distances as the concept matrix Z
# --------------------------
# Here, we use the raw distances as the concept to erase
Z_train = y_train.reshape(-1, 1)
Z_test = y_test.reshape(-1, 1)

# --------------------------
# Apply LEACE
# --------------------------
eraser = LeaceEraser.fit(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(Z_train, dtype=torch.float32))
X_train_erased = eraser(torch.tensor(X_train_scaled, dtype=torch.float32)).numpy()
X_test_erased = eraser(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

# --------------------------
# Fit probe after LEACE
# --------------------------
probe_erased = LinearRegression()
probe_erased.fit(X_train_erased, y_train)
y_pred_test_erased = probe_erased.predict(X_test_erased)
mse_after = mean_squared_error(y_test, y_pred_test_erased)
print(f"MSE on test set AFTER erasure: {mse_after:.4f}")

# --------------------------
# Save erased embeddings and eraser object
# --------------------------
with open("Distances/Erased_embeddings/leace_embeddings_synt_dist.pkl", "wb") as f:
    pickle.dump({
        "train_erased": X_train_erased,
        "test_erased": X_test_erased,
        "train_labels": y_train,
        "test_labels": y_test,
    }, f)
with open("Distances/Eraser_objects/leace_eraser_synt_dist.pkl", "wb") as f:
    pickle.dump(eraser, f)

# --------------------------
# Save results
# --------------------------
results = {
    "mse_before": mse_before,
    "mse_after": mse_after,
    "train_pairs": int(X_train.shape[0]),
    "test_pairs": int(X_test.shape[0]),
}
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE}")
