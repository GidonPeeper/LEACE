"""
leace_synt_dist_oracle.py

This script performs **Oracle LEACE** (manual version) on GPT-2 embeddings to erase the concept of syntactic (tree) distances at the word level.
For each word, the concept to erase is its vector of syntactic distances to all other words in the sentence.
The script loads precomputed embeddings and their corresponding pairwise syntactic distance matrices,
fits Oracle LEACE to remove information about this word-level syntactic distance vector concept from the embeddings, and saves the erased embeddings.

Inputs:
    - GPT-2 embeddings with syntactic distance matrices for train and test sets

Outputs:
    - Erased embeddings (train and test)
"""

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
    projection = Σ_XZ @ Σ_ZZ_inv @ Z_centered.T
    X_erased = X - projection.T
    return X_erased

# --------------------------
# Settings: Erasure of word-level syntactic distance vectors
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
ERASED_EMB_FILE = "Distances/Embeddings/Erased_embeddings/Narratives/oracle_leace_embeddings_synt_dist_vec.pkl"

os.makedirs(os.path.dirname(ERASED_EMB_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

# --------------------------
# Find maximum sentence length in the dataset
# --------------------------
def get_max_sentence_length(*datasets, layer=LAYER):
    max_len = 0
    for dataset in datasets:
        for sent in dataset:
            n = sent["embeddings_by_layer"][layer].shape[0]
            if n > max_len:
                max_len = n
    return max_len

max_sentence_length = get_max_sentence_length(data, test_data, layer=LAYER)
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

print("Preparing word-level features and vector concepts for train set...")
X_train, Z_train = get_word_level_features_and_vector_concepts(data, LAYER, max_sentence_length, pad_value)
print("Preparing word-level features and vector concepts for test set...")
X_test, Z_test = get_word_level_features_and_vector_concepts(test_data, LAYER, max_sentence_length, pad_value)

# --------------------------
# Standardize features
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaler fitted.")

# Check and filter non-finite values
mask = np.isfinite(X_train_scaled).all(axis=1) & np.isfinite(Z_train).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values in train set.")
    X_train_scaled = X_train_scaled[mask]
    Z_train = Z_train[mask]

mask_test = np.isfinite(X_test_scaled).all(axis=1) & np.isfinite(Z_test).all(axis=1)
if not mask_test.all():
    print(f"Filtered out {np.sum(~mask_test)} rows with non-finite values in test set.")
    X_test_scaled = X_test_scaled[mask_test]
    Z_test = Z_test[mask_test]

# --------------------------
# Apply Oracle LEACE erasure to all embeddings (train and test)
# --------------------------
print("Applying Oracle LEACE to train embeddings...")
X_train_erased_scaled = oracle_leace(X_train_scaled, Z_train)
print("Applying Oracle LEACE to test embeddings...")
X_test_erased_scaled = oracle_leace(X_test_scaled, Z_test)

# Inverse transform to return to original embedding space
X_train_erased = scaler.inverse_transform(X_train_erased_scaled)
X_test_erased = scaler.inverse_transform(X_test_erased_scaled)

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

train_erased = reconstruct_sentencewise(X_train_erased, data, LAYER)
test_erased = reconstruct_sentencewise(X_test_erased, test_data, LAYER)

# --------------------------
# Save erased embeddings
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "train_erased": train_erased,
        "test_erased": test_erased,
    }, f)

print(f"\nDone. Oracle LEACE erased embeddings saved to {ERASED_EMB_FILE}")
