import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from concept_erasure import LeaceEraser
from collections import Counter
import os
import json

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Stage3/Embeddings/UD/AllPOS/Original_embeddings/gpt2_embeddings.pt" 
TEST_FILE = "Stage3/Embeddings/UD/AllPOS/Original_embeddings/gpt2_embeddings_test.pt"
SEED = 42

torch.manual_seed(SEED)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)

all_pos_tags = sorted(list(data[0]["word_labels"].keys()))
labels_by_feature = {}
for feat in all_pos_tags:
    labels_by_feature[feat] = torch.tensor(
        [label for sent in data for label in sent["word_labels"][feat]]
    ).long()

with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)
X_test_all = [sent["embeddings_by_layer"][LAYER] for sent in test_data]
X_test = torch.cat(X_test_all, dim=0)
labels_by_feature_test = {}
for feat in all_pos_tags:
    labels_by_feature_test[feat] = torch.tensor(
        [label for sent in test_data for label in sent["word_labels"][feat]]
    ).long()

# --------------------------
# Erase all POS tags at once
# --------------------------
# Stack all POS tag labels as columns to form the concept matrix Z_all
Z_all = torch.stack([labels_by_feature[feat] for feat in all_pos_tags], dim=1).float()

# Fit LEACE eraser on all POS tags jointly
eraser_all = LeaceEraser.fit(X, Z_all)
X_all_erased = eraser_all(X)
X_test_all_erased = eraser_all(X_test)

# Save the eraser and erased embeddings
with open("Stage4/Embeddings/UD/leace_embeddings_ALL_POS.pkl", "wb") as f:
    pickle.dump({
        "train_erased": X_all_erased.cpu(),
        "test_erased": X_test_all_erased.cpu(),
        "train_labels": {feat: labels_by_feature[feat].cpu() for feat in all_pos_tags},
        "test_labels": {feat: labels_by_feature_test[feat].cpu() for feat in all_pos_tags},
    }, f)

# --- Save the eraser object ---
with open(f"Stage4/Eraser_objects/UD/leace_eraser__ALL_POS.pkl", "wb") as f:
    pickle.dump(eraser_all, f)
# ----------------------------------------

# Compute L2 norm between original and erased embeddings
l2_train_all = torch.norm(X - X_all_erased, dim=1).mean().item()
l2_test_all = torch.norm(X_test - X_test_all_erased, dim=1).mean().item()
print(f"\nL2 norm (train, ALL_POS): {l2_train_all:.4f}")
print(f"L2 norm (test, ALL_POS): {l2_test_all:.4f}")

# Evaluate probe accuracy for each POS tag after erasing all POS tags
scaler_all = StandardScaler()
X_train_all_np = scaler_all.fit_transform(X_all_erased.numpy())
X_test_all_np = scaler_all.transform(X_test_all_erased.numpy())

probe_results = {}
print("\nProbe accuracy after erasing ALL POS tags:")
for feat in all_pos_tags:
    y = labels_by_feature[feat]
    y_test = labels_by_feature_test[feat]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_all_np, y.numpy())
    acc = clf.score(X_test_all_np, y_test.numpy())
    probe_results[feat] = acc
    print(f"{feat:10}: {acc:.4f}")

# Save probe results and L2 norms
with open("Stage4/Results/leace_results_ALL_POS.json", "wb") as f:
    pickle.dump({
        "probe_accuracy": probe_results,
        "l2_train": l2_train_all,
        "l2_test": l2_test_all,
    }, f)