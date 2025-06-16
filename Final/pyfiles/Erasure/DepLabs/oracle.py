"""
This script performs Oracle LEACE on GPT-2 embeddings to erase the concept of dependency labels (deprel) at the word level.
For each word, the concept to erase is its dependency label (one-hot encoded).
The script loads precomputed embeddings and their corresponding dependency labels,
fits Oracle LEACE to remove information about the dependency label concept from the embeddings, and saves the erased embeddings.

Inputs:
    - GPT-2 embeddings with aligned dependency labels for all sentences (train+test or train+val+test) as a single file

Outputs:
    - Erased embeddings (all sentences)
"""

import argparse
import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Oracle LEACE (manual version)
# --------------------------
def oracle_leace(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Z_centered = Z - np.mean(Z, axis=0, keepdims=True)
    Σ_XZ = X_centered.T @ Z_centered / X.shape[0]
    Σ_ZZ = Z_centered.T @ Z_centered / X.shape[0]
    Σ_ZZ_inv = np.linalg.pinv(Σ_ZZ)
    W = Σ_XZ @ Σ_ZZ_inv
    X_proj = Z_centered @ W.T
    X_erased = X_centered - X_proj
    # Add back the mean of X so the scale is preserved
    X_erased = X_erased + np.mean(X, axis=0, keepdims=True)
    return X_erased

# --------------------------
# Argument parsing
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["narratives", "ud"], required=True,
                    help="Which dataset to erase: 'narratives' or 'ud'")
args = parser.parse_args()

if args.dataset == "narratives":
    LAYER = 8
    EMBEDDING_FILE = "Final/Embeddings/Original/Narratives/Embed_nar_deplabs.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/Narratives/oracle_nar_deplabs_vec.pkl"
elif args.dataset == "ud":
    LAYER = 8
    EMBEDDING_FILE = "Final/Embeddings/Original/UD/Embed_ud_deplabs.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/UD/oracle_ud_deplabs_vec.pkl"
else:
    raise ValueError("Unknown dataset")

os.makedirs(os.path.dirname(ERASED_EMB_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

# --------------------------
# Collect all dependency labels in the dataset
# --------------------------
all_deplabs = set()
for sent in data:
    all_deplabs.update(sent["deplabs"])
all_deplabs = sorted(list(all_deplabs))
deplab_to_idx = {dep: i for i, dep in enumerate(all_deplabs)}
print(f"Dependency labels found: {all_deplabs}")

# --------------------------
# Prepare word-level features and dependency label concepts
# --------------------------
def get_word_level_features_and_deplab_concepts(dataset, layer, deplab_to_idx, num_deplabs):
    X = []
    Z = []
    for sent in dataset:
        emb = sent["embeddings_by_layer"][layer]  # [num_words, hidden_dim]
        deplabs = sent["deplabs"]                 # [num_words]
        n = emb.shape[0]
        for i in range(n):
            X.append(emb[i].numpy())
            onehot = np.zeros(num_deplabs, dtype=np.float32)
            onehot[deplab_to_idx[deplabs[i]]] = 1.0
            Z.append(onehot)
    return np.stack(X), np.stack(Z)

print("Preparing word-level features and dependency label concepts for all sentences...")
num_deplabs = len(all_deplabs)
X_all, Z_all = get_word_level_features_and_deplab_concepts(data, LAYER, deplab_to_idx, num_deplabs)

# --------------------------
# Standardize features
# --------------------------
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
print("Scaler fitted.")

# Check and filter non-finite values
mask = np.isfinite(X_all_scaled).all(axis=1) & np.isfinite(Z_all).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values.")
    X_all_scaled = X_all_scaled[mask]
    Z_all = Z_all[mask]

# --------------------------
# Apply Oracle LEACE erasure to all embeddings
# --------------------------
print("Applying Oracle LEACE to all embeddings...")
X_all_erased_scaled = oracle_leace(X_all_scaled, Z_all)

# Inverse transform to return to original embedding space
X_all_erased = scaler.inverse_transform(X_all_erased_scaled)

# --------------------------
# Reconstruct sentence-wise structure for saving
# --------------------------
def reconstruct_sentencewise(erased_flat, dataset, layer):
    out = []
    idx = 0
    for sent in dataset:
        n = sent["embeddings_by_layer"][layer].shape[0]
        out.append(erased_flat[idx:idx+n])
        idx += n
    return out

all_erased = reconstruct_sentencewise(X_all_erased, data, LAYER)

# --------------------------
# Save erased embeddings
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "all_erased": all_erased,
    }, f)

print(f"\nDone. Oracle LEACE erased embeddings saved to {ERASED_EMB_FILE}")