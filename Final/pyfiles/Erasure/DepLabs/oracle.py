"""
This script performs Oracle LEACE on GPT-2 embeddings to erase the concept of dependency labels.
The script loads precomputed embeddings and their corresponding dependency labels,
fits Oracle LEACE to remove information about the dependency label concept from the embeddings, and saves the erased embeddings.

This implementation follows the full concept erasure approach using multiclass label vectors.

Inputs:
    - GPT-2 embeddings with aligned dependency labels for all sentences as a single file

Outputs:
    - Erased embeddings (all sentences)
"""

import argparse
import pickle
import torch
import numpy as np
from collections import Counter
import os
import json

# --------------------------
# Oracle LEACE (following full concept erasure implementation)
# --------------------------
def oracle_leace(X: np.ndarray, Z: np.ndarray):
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Z_centered = Z - np.mean(Z, axis=0, keepdims=True)
    Σ_XZ = X_centered.T @ Z_centered / X.shape[0]
    Σ_ZZ = Z_centered.T @ Z_centered / X.shape[0]
    Σ_ZZ_inv = np.linalg.pinv(Σ_ZZ)
    projection = Σ_XZ @ Σ_ZZ_inv @ Z_centered.T
    X_erased = X - projection.T
    return X_erased, projection

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
    RESULTS_FILE = "Final/Results/Narratives/oracle_deplabs_results.json"
elif args.dataset == "ud":
    LAYER = 8
    EMBEDDING_FILE = "Final/Embeddings/Original/UD/Embed_ud_deplabs.pkl"
    ERASED_EMB_FILE = "Final/Embeddings/Erased/UD/oracle_ud_deplabs_vec.pkl"
    RESULTS_FILE = "Final/Results/UD/oracle_deplabs_results.json"
else:
    raise ValueError("Unknown dataset")

os.makedirs(os.path.dirname(ERASED_EMB_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
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
print(f"Dependency labels found: {all_deplabs}")

# --------------------------
# Calculate most frequent dependency label (chance accuracy)
# --------------------------
all_deplabs_flat = []
for sent in data:
    all_deplabs_flat.extend(sent["deplabs"])
deplab_counts = Counter(all_deplabs_flat)
most_frequent_deplab = deplab_counts.most_common(1)[0][0]
chance_accuracy = deplab_counts[most_frequent_deplab] / len(all_deplabs_flat)
print(f"Most frequent dependency label: {most_frequent_deplab} ({chance_accuracy:.4f} chance accuracy)")

# --------------------------
# Prepare embeddings and labels following full concept erasure approach
# --------------------------
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)

# Create labels_by_feature dictionary
labels_by_feature = {}
for feat in all_deplabs:
    labels = []
    for sent in data:
        for deplab in sent["deplabs"]:
            labels.append(1 if deplab == feat else 0)
    labels_by_feature[feat] = torch.tensor(labels).long()

# --------------------------
# Create multiclass label vector for Oracle LEACE (following full concept erasure)
# --------------------------
deplab_to_idx = {dep: i for i, dep in enumerate(all_deplabs)}
y_deplab = torch.stack([labels_by_feature[feat] for feat in all_deplabs], dim=1).argmax(dim=1)

num_deplabs = len(all_deplabs)
y_onehot = torch.nn.functional.one_hot(y_deplab, num_classes=num_deplabs).float()
Z_all = y_onehot[:, :-1]  # Drop last column for full rank

# Check the rank of Z_all
Z_all_np = Z_all.numpy()
rank = np.linalg.matrix_rank(Z_all_np)
print(f"Rank of Z_all: {rank} / {Z_all_np.shape[1]} (num dependency labels minus one)")

# --------------------------
# Apply Oracle LEACE
# --------------------------
X_np = X.numpy()
X_erased_np, projection = oracle_leace(X_np, Z_all_np)
X_erased = torch.from_numpy(X_erased_np)

# --------------------------
# Calculate L2 distance
# --------------------------
l2_distance = torch.norm(X - X_erased, dim=1).mean().item()
print(f"L2 distance: {l2_distance:.4f}")

# --------------------------
# Reconstruct sentence-wise structure for saving
# --------------------------
def reconstruct_sentencewise(erased_flat, dataset, layer):
    out = []
    idx = 0
    for sent in dataset:
        n = sent["embeddings_by_layer"][layer].shape[0]
        out.append(erased_flat[idx:idx+n].cpu().numpy())
        idx += n
    return out

all_erased = reconstruct_sentencewise(X_erased, data, LAYER)

# --------------------------
# Save erased embeddings
# --------------------------
with open(ERASED_EMB_FILE, "wb") as f:
    pickle.dump({
        "all_erased": all_erased,
    }, f)

# --------------------------
# Save results
# --------------------------
results = {
    "dataset": args.dataset,
    "concept": "deplabs",
    "method": "oracle",
    "layer": LAYER,
    "most_frequent_deplab": most_frequent_deplab,
    "chance_accuracy": chance_accuracy,
    "l2_distance": l2_distance,
    "z_rank": int(rank),
    "num_deplabs": num_deplabs,
    "deplabs": all_deplabs
}

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Oracle LEACE erased embeddings saved to {ERASED_EMB_FILE}")
print(f"Results saved to {RESULTS_FILE}")