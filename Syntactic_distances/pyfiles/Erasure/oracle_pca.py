"""
oracle_pca.py

This script performs Oracle LEACE (manual version) on GPT-2 embeddings to erase the concept of syntactic (tree) distances at the word level,
**after first projecting the concept vectors to a lower-dimensional space using PCA** (as in leace_pca.py).
For each word, the concept to erase is its vector of syntactic distances to all other words in the sentence, padded and then PCA-reduced.

Inputs:
    - GPT-2 embeddings with syntactic distance matrices for train and test sets

Outputs:
    - Erased embeddings (train and test)
    - Fitted PCA projection object
"""

import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --------------------------
# Settings
# --------------------------
# Narratives or UD
LAYER = 8
N_PCA_COMPONENTS = 20  # Number of PCA components to keep

EMBEDDING_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_test_synt_dist.pt"
ERASED_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec_pca.pkl"
PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/UD/oracle_pca_synt_dist_vec.pkl"

os.makedirs(os.path.dirname(ERASED_EMB_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PCA_OBJ_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

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
    X_erased = X_erased + np.mean(X, axis=0, keepdims=True)
    return X_erased

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
pad_value = max_sentence_length  # Safe: cannot be a real distance
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
            dist_vec = dist[i].tolist()
            dist_vec += [pad_value] * (max_len - len(dist_vec))
            Z.append(dist_vec)
    return np.stack(X), np.stack(Z)

print("Preparing word-level features and vector concepts for train set...")
X_train, Z_train = get_word_level_features_and_vector_concepts(data, LAYER, max_sentence_length, pad_value)
print("Preparing word-level features and vector concepts for test set...")
X_test, Z_test = get_word_level_features_and_vector_concepts(test_data, LAYER, max_sentence_length, pad_value)

# --------------------------
# Standardize features and apply PCA to Z
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaler fitted.")

# Remove non-finite rows
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

# Apply PCA to concept vectors
pca = PCA(n_components=N_PCA_COMPONENTS)
Z_train_pca = pca.fit_transform(Z_train)
Z_test_pca = pca.transform(Z_test)
print(f"PCA fitted. Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")

# --------------------------
# Apply Oracle LEACE erasure to all embeddings (train and test)
# --------------------------
print("Applying Oracle LEACE to train embeddings...")
X_train_erased_scaled = oracle_leace(X_train_scaled, Z_train_pca)
print("Applying Oracle LEACE to test embeddings...")
X_test_erased_scaled = oracle_leace(X_test_scaled, Z_test_pca)

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
# Save erased embeddings and PCA object
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "train_erased": train_erased,
        "test_erased": test_erased,
    }, f)

with open(PCA_OBJ_FILE, "wb") as f:
    pickle.dump(pca, f)

print(f"\nDone. Oracle LEACE erased embeddings (with PCA) saved to {ERASED_EMB_FILE}")
print(f"PCA object saved to {PCA_OBJ_FILE}")