"""
This script performs Oracle LEACE on GPT-2 embeddings to erase the concept of syntactic (tree) distances at the word level.
For each word, the concept to erase is its vector of syntactic distances to all other words in the sentence.
The script loads precomputed embeddings and their corresponding pairwise syntactic distance matrices,
fits Oracle LEACE to remove information about this word-level syntactic distance vector concept from the embeddings, and saves the erased embeddings.

Inputs:
    - GPT-2 embeddings with syntactic distance matrices for all sentences (train+test or train+val+test) as a single file

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
    EMBEDDING_FILE = "Final/Embeddings/Original/Narratives/Embed_nar_sd.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/Narratives/oracle_ar_sd_vec.pkl"
elif args.dataset == "ud":
    LAYER = 8
    EMBEDDING_FILE = "Final/Embeddings/Original/UD/Embed_ud_sd.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/UD/oracle_ud_sd_vec.pkl"
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
# Find maximum sentence length in the dataset
# --------------------------
def get_max_sentence_length(dataset, layer=LAYER):
    max_len = 0
    for sent in dataset:
        n = sent["embeddings_by_layer"][layer].shape[0]
        if n > max_len:
            max_len = n
    return max_len

max_sentence_length = get_max_sentence_length(data, layer=LAYER)
pad_value = max_sentence_length - 1
print(f"Max sentence length: {max_sentence_length}, pad value: {pad_value}")

# --------------------------
# Prepare word-level features and vector concepts
# --------------------------
def get_word_level_features_and_vector_concepts(dataset, layer, max_len, pad_value):
    X = []
    Z = []
    for sent in dataset:
        emb = sent["embeddings_by_layer"][layer]  # [num_words, hidden_dim]
        dist = sent["distance_matrix"]            # [num_words, num_words]
        n = emb.shape[0]
        for i in range(n):
            X.append(emb[i].numpy())
            # Concept: syntactic distance vector from word i to all other words, padded
            dist_vec = dist[i].tolist()
            if len(dist_vec) < max_len:
                dist_vec += [pad_value] * (max_len - len(dist_vec))
            Z.append(dist_vec)
    return np.stack(X), np.stack(Z)

print("Preparing word-level features and vector concepts for all sentences...")
X_all, Z_all = get_word_level_features_and_vector_concepts(data, LAYER, max_sentence_length, pad_value)

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
