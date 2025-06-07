"""
leace_synt_dist.py

This script performs LEACE (Linear Concept Erasure) on GPT-2 embeddings to erase the concept of 
syntactic (tree) distances at the word level. Each word's concept is defined as its vector of 
syntactic distances to all other words in the sentence, padded to the length of the longest sentence 
in the dataset. To reduce noise introduced by padding and emphasize the true structure of the 
syntactic distance vectors, the concept vectors are first projected to a lower-dimensional space 
using PCA.

The script proceeds as follows:
- Loads precomputed GPT-2 embeddings and syntactic distance matrices (e.g., from CoNLL-U parses).
- Pads all distance vectors to a uniform length using a safe constant value.
- Applies PCA to reduce the dimensionality of the concept vectors (Z).
- Standardizes the input embeddings (X) and fits LEACE to remove the PCA-transformed concept.
- Applies the fitted LEACE transformation to both train and test embeddings.
- Saves the erased embeddings, the fitted LEACE eraser, and the PCA projection object.

Inputs:
    - Pickled GPT-2 embeddings with syntactic distance matrices for train and test sets

Outputs:
    - Pickled erased embeddings (train and test)
    - Pickled LEACE eraser object
    - Pickled PCA projection object
"""

import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from concept_erasure import LeaceEraser
import os

# --------------------------
# Settings
# --------------------------
LAYER = 8
N_PCA_COMPONENTS = 20  # Number of PCA components to keep

EMBEDDING_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
ERASED_EMB_FILE = "Distances/Embeddings/Erased_embeddings/Narratives/leace_embeddings_synt_dist_vec_pca.pkl"
ERASER_OBJ_FILE = "Distances/Eraser_objects/Narratives/leace_eraser_synt_dist_vec_pca.pkl"
PCA_OBJ_FILE = "Distances/Eraser_objects/Narratives/leace_pca_synt_dist_vec.pkl"

os.makedirs(os.path.dirname(ERASED_EMB_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ERASER_OBJ_FILE), exist_ok=True)
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
# Determine padding value
# --------------------------
def get_max_sentence_length(*datasets, layer=LAYER):
    return max(
        sent["embeddings_by_layer"][layer].shape[0]
        for dataset in datasets for sent in dataset
    )

max_sentence_length = get_max_sentence_length(data, test_data, layer=LAYER)
pad_value = max_sentence_length  # Safe: cannot be a real distance
print(f"Max sentence length: {max_sentence_length}, pad value: {pad_value}")

# --------------------------
# Extract features and concepts
# --------------------------
def get_word_level_features_and_vector_concepts(dataset, layer, max_len, pad_value):
    X, Z = [], []
    for sent in dataset:
        emb = sent["embeddings_by_layer"][layer]
        dist = sent["distance_matrix"]
        for i in range(emb.shape[0]):
            X.append(emb[i].numpy())
            row = dist[i].tolist()
            row += [pad_value] * (max_len - len(row))
            Z.append(row)
    return np.stack(X), np.stack(Z)

print("Preparing word-level features and concepts...")
X_train, Z_train = get_word_level_features_and_vector_concepts(data, LAYER, max_sentence_length, pad_value)
X_test, Z_test = get_word_level_features_and_vector_concepts(test_data, LAYER, max_sentence_length, pad_value)

# --------------------------
# Standardize X and apply PCA to Z
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("X scaler fitted.")

mask = np.isfinite(X_train_scaled).all(axis=1) & np.isfinite(Z_train).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values.")
    X_train_scaled = X_train_scaled[mask]
    Z_train = Z_train[mask]

# Apply PCA to concept vectors
pca = PCA(n_components=N_PCA_COMPONENTS)
Z_train_pca = pca.fit_transform(Z_train)
print(f"PCA fitted. Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")

# --------------------------
# Filter and transform Z_test safely
# --------------------------
if not np.isfinite(Z_test).all():
    print(f"Warning: Z_test contains non-finite values. Filtering...")
    mask = np.isfinite(Z_test).all(axis=1)
    Z_test = Z_test[mask]
    X_test = X_test[mask]

Z_test_pca = pca.transform(Z_test)

# --------------------------
# Fit LEACE
# --------------------------
eraser = LeaceEraser.fit(
    torch.tensor(X_train_scaled, dtype=torch.float64),
    torch.tensor(Z_train_pca, dtype=torch.float64)
)
print("LEACE eraser fitted.")

# --------------------------
# Apply erasure to full datasets
# --------------------------
def erase_embeddings(dataset, layer, scaler, eraser):
    erased = []
    for sent in dataset:
        emb = sent["embeddings_by_layer"][layer]
        emb_scaled = scaler.transform(emb.numpy())
        emb_erased_scaled = eraser(torch.tensor(emb_scaled, dtype=torch.float64)).numpy()
        emb_erased = scaler.inverse_transform(emb_erased_scaled)
        erased.append(emb_erased)
    return erased

print("Erasing train embeddings...")
train_erased = erase_embeddings(data, LAYER, scaler, eraser)
print("Erasing test embeddings...")
test_erased = erase_embeddings(test_data, LAYER, scaler, eraser)

# --------------------------
# Save outputs
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "train_erased": train_erased,
        "test_erased": test_erased,
    }, f)

with open(ERASER_OBJ_FILE, "wb") as f:
    pickle.dump(eraser, f)

with open(PCA_OBJ_FILE, "wb") as f:
    pickle.dump(pca, f)

print(f"\nDone. Erased embeddings saved to {ERASED_EMB_FILE}")
print(f"LEACE eraser object saved to {ERASER_OBJ_FILE}")
print(f"PCA object saved to {PCA_OBJ_FILE}")
