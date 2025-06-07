"""
leace_synt_dist.py

This script performs LEACE (Linear Concept Erasure) on GPT-2 embeddings to erase the concept of syntactic (tree) distances at the **word level**.
For each word, the concept to erase is its vector of syntactic distances to all other words in the sentence.
To make this possible, each word's concept vector is padded to the length of the longest sentence in the dataset, using the value (max_sentence_length - 1) for padding.
The script loads precomputed embeddings and their corresponding pairwise syntactic distance matrices (as produced by encode_synt_distances.py),
fits LEACE to remove information about this word-level syntactic distance vector concept from the embeddings, and saves the eraser object and the erased embeddings.

Inputs:
    - GPT-2 embeddings with syntactic distance matrices for train and test sets

Outputs:
    - Erased embeddings (train and test)
    - LEACE eraser object
"""

import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from concept_erasure import LeaceEraser
import os

# --------------------------
# Settings: Erasure of word-level syntactic distance vectors
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
ERASED_EMB_FILE = "Distances/Embeddings/Erased_embeddings/Narratives/leace_embeddings_synt_dist_vec.pkl"
ERASER_OBJ_FILE = "Distances/Eraser_objects/Narratives/leace_eraser_synt_dist_vec.pkl"

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
print("Scaler fitted.")


# Check and filter non-finite values
mask = np.isfinite(X_train_scaled).all(axis=1) & np.isfinite(Z_train).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values.")
    X_train_scaled = X_train_scaled[mask]
    Z_train = Z_train[mask]

# --------------------------
# Fit LEACE eraser on word-level features/vector concepts
# --------------------------
eraser = LeaceEraser.fit(
    torch.tensor(X_train_scaled, dtype=torch.float64),
    torch.tensor(Z_train, dtype=torch.float64)
)
print("LEACE eraser fitted.")

# --------------------------
# Apply erasure to all embeddings (train and test)
# --------------------------
def erase_embeddings(dataset, layer, scaler, eraser):
    erased = []
    for sent in dataset:
        emb = sent["embeddings_by_layer"][layer]  # [num_words, hidden_dim]
        emb_scaled = scaler.transform(emb.numpy())
        emb_erased_scaled = eraser(torch.tensor(emb_scaled, dtype=torch.float64)).numpy()
        # Inverse transform to return to original embedding space
        emb_erased = scaler.inverse_transform(emb_erased_scaled)
        erased.append(emb_erased)
    return erased

print("Erasing train embeddings...")
train_erased = erase_embeddings(data, LAYER, scaler, eraser)
print("Erasing test embeddings...")
test_erased = erase_embeddings(test_data, LAYER, scaler, eraser)

# --------------------------
# Save erased embeddings and eraser object
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "train_erased": train_erased,
        "test_erased": test_erased,
    }, f)
with open(ERASER_OBJ_FILE, "wb") as f:
    pickle.dump(eraser, f)

print(f"\nDone. Erased embeddings saved to {ERASED_EMB_FILE}")
print(f"LEACE eraser object saved to {ERASER_OBJ_FILE}")
