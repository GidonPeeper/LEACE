"""
This script performs LEACE (Linear Concept Erasure) on GPT-2 embeddings to erase the concept of POS tags at the word level.
For each word, the concept to erase is its POS tag (one-hot encoded).
The script loads precomputed embeddings and their corresponding POS tags,
fits LEACE to remove information about the POS concept from the embeddings, and saves the eraser object and the erased embeddings.

Inputs:
    - GPT-2 embeddings with aligned POS tags for all sentences (train+test or train+val+test) as a single file

Outputs:
    - Erased embeddings (all sentences)
    - LEACE eraser object
"""

import argparse
import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from concept_erasure import LeaceEraser
import os

# --------------------------
# Argument parsing
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["narratives", "ud"], required=True,
                    help="Which dataset to erase: 'narratives' or 'ud'")
args = parser.parse_args()

if args.dataset == "narratives":
    LAYER = 8
    EMBEDDING_FILE = "Final/Embeddings/Original/Narratives/Embed_nar_pos.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/Narratives/leace_nar_pos_vec.pkl"
    ERASER_OBJ_FILE = "Final/Eraser_objects/Narratives/eraser_obj_nar_pos.pkl"
elif args.dataset == "ud":
    LAYER = 8
    EMBEDDING_FILE = "Final/Embeddings/Original/UD/Embed_ud_pos.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/UD/leace_ud_pos_vec.pkl"
    ERASER_OBJ_FILE = "Final/Eraser_objects/UD/eraser_obj_ud_pos.pkl"
else:
    raise ValueError("Unknown dataset")

os.makedirs(os.path.dirname(ERASED_EMB_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ERASER_OBJ_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

# --------------------------
# Collect all POS tags in the dataset
# --------------------------
all_pos_tags = set()
for sent in data:
    all_pos_tags.update(sent["pos_tags"])
all_pos_tags = sorted(list(all_pos_tags))
pos_to_idx = {pos: i for i, pos in enumerate(all_pos_tags)}
print(f"POS tags found: {all_pos_tags}")

# --------------------------
# Prepare word-level features and POS concepts
# --------------------------
def get_word_level_features_and_pos_concepts(dataset, layer, pos_to_idx, num_pos):
    X = []
    Z = []
    for sent in dataset:
        emb = sent["embeddings_by_layer"][layer]  # [num_words, hidden_dim]
        pos_tags = sent["pos_tags"]               # [num_words]
        n = emb.shape[0]
        for i in range(n):
            X.append(emb[i].numpy())
            onehot = np.zeros(num_pos, dtype=np.float32)
            onehot[pos_to_idx[pos_tags[i]]] = 1.0
            Z.append(onehot)
    return np.stack(X), np.stack(Z)

print("Preparing word-level features and POS concepts for all sentences...")
num_pos = len(all_pos_tags)
X_all, Z_all = get_word_level_features_and_pos_concepts(data, LAYER, pos_to_idx, num_pos)

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
# Fit LEACE eraser on word-level features/POS concepts
# --------------------------
eraser = LeaceEraser.fit(
    torch.tensor(X_all_scaled, dtype=torch.float64),
    torch.tensor(Z_all, dtype=torch.float64)
)
print("LEACE eraser fitted.")

# --------------------------
# Apply erasure to all embeddings
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

print("Erasing all embeddings...")
all_erased = erase_embeddings(data, LAYER, scaler, eraser)

# --------------------------
# Save erased embeddings and eraser object
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "all_erased": all_erased,
    }, f)
with open(ERASER_OBJ_FILE, "wb") as f:
    pickle.dump(eraser, f)

print(f"\nDone. Erased embeddings saved to {ERASED_EMB_FILE}")
print(f"LEACE eraser object saved to {ERASER_OBJ_FILE}")
