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
EMBEDDING_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings.pt"
TEST_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings_test.pt"
RESULTS_FILE = "Stage4/Results/UD/Synt_deps/leace.json"
BALANCE_CLASSES = False
SEED = 42

torch.manual_seed(SEED)

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs("Stage4/Eraser_objects/UD/", exist_ok=True)
os.makedirs("Stage4/Embeddings/UD/Erased_embeddings/", exist_ok=True)

# --------------------------
# Load data once
# --------------------------
# Load train data
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)

# Get all dependency labels as features
all_dep_labels = sorted(list(data[0]["word_labels"].keys()))

labels_by_feature = {}
for feat in all_dep_labels:
    labels_by_feature[feat] = torch.tensor(
        [label for sent in data for label in sent["word_labels"][feat]]
    ).long()

# Load test data
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)
X_test_all = [sent["embeddings_by_layer"][LAYER] for sent in test_data]
X_test = torch.cat(X_test_all, dim=0)
labels_by_feature_test = {}
for feat in all_dep_labels:
    labels_by_feature_test[feat] = torch.tensor(
        [label for sent in test_data for label in sent["word_labels"][feat]]
    ).long()

# --------------------------
# Evaluate all source → target combinations
# --------------------------
results = []

for source_feat in all_dep_labels:
    print(f"\n=== LEACE erasing: {source_feat} ===")
    # Prepare LEACE
    X_train = torch.cat([sent["embeddings_by_layer"][LAYER] for sent in data], dim=0)
    X_test = torch.cat([sent["embeddings_by_layer"][LAYER] for sent in test_data], dim=0)
    y_train = torch.tensor([label for sent in data for label in sent["word_labels"][source_feat]]).long()
    y_test = torch.tensor([label for sent in test_data for label in sent["word_labels"][source_feat]]).long()

    # Fit LEACE
    Z = y_train.unsqueeze(1).float()
    eraser = LeaceEraser.fit(X_train, Z)
    X_train_erased = eraser(X_train)
    X_test_erased = eraser(X_test)

    # Save both train and test erased embeddings for this feature
    with open(f"Stage4/Embeddings/UD/Synt_deps/Erased_embeddings/s4_leace_embeddings_{source_feat}.pkl", "wb") as f:
        pickle.dump({
            "train_erased": X_train_erased.cpu(),
            "test_erased": X_test_erased.cpu(),
            "train_labels": y_train.cpu(),
            "test_labels": y_test.cpu(),
        }, f)

    # --- Save the eraser for this feature ---
    with open(f"Stage4/Eraser_objects/UD/s4_leace_eraser_{source_feat}.pkl", "wb") as f:
        pickle.dump(eraser, f)
    # ----------------------------------------

    # --- Compute L2 norm between original and erased embeddings ---
    l2_train = torch.norm(X_train - X_train_erased, dim=1).mean().item()
    l2_test = torch.norm(X_test - X_test_erased, dim=1).mean().item()
    print(f"L2 norm (train, {source_feat}): {l2_train:.4f}")
    print(f"L2 norm (test, {source_feat}): {l2_test:.4f}")
    # -------------------------------------------------------------

    # Scale
    scaler = StandardScaler()
    X_train_e_np = scaler.fit_transform(X_train_erased.numpy())
    X_test_e_np = scaler.transform(X_test_erased.numpy())

    # Refit probe after LEACE on original concept
    clf_leace = LogisticRegression(max_iter=2000)
    clf_leace.fit(X_train_e_np, y_train.numpy())
    acc_erased = clf_leace.score(X_test_e_np, y_test.numpy())

    # Compute baseline accuracy for this feature
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_e_np, y_train.numpy())
    baseline_acc = dummy.score(X_test_e_np, y_test.numpy())

    for target_feat in all_dep_labels:
        y2 = labels_by_feature[target_feat]
        y2_test = labels_by_feature_test[target_feat]

        # Before LEACE
        clf2_orig = LogisticRegression(max_iter=2000)
        clf2_orig.fit(X_train_e_np, y2.numpy())
        acc2_orig = clf2_orig.score(X_test_e_np, y2_test.numpy())

        # After LEACE
        clf2_leace = LogisticRegression(max_iter=2000)
        clf2_leace.fit(X_train_e_np, y2.numpy())
        acc2_leace = clf2_leace.score(X_test_e_np, y2_test.numpy())

        results.append({
            "erased": source_feat,
            "probed": target_feat,
            "acc_before": acc2_orig,
            "acc_after": acc2_leace,
            "erasure_accuracy": acc_erased if source_feat == target_feat else None,
            "baseline": baseline_acc if source_feat == target_feat else None,
            "l2_train": l2_train if source_feat == target_feat else None,
            "l2_test": l2_test if source_feat == target_feat else None
        })

# --------------------------
# Save Results
# --------------------------
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

# --------------------------
# Display Results
# --------------------------
print("\n\nLEACE Concept Erasure Matrix (Dependency labels)")
print("Erased → Probed | Acc (before) → Acc (after)")
for source_feat in all_dep_labels:
    for target_feat in all_dep_labels:
        row = next(r for r in results if r["erased"] == source_feat and r["probed"] == target_feat)
        print(f"{source_feat:10} → {target_feat:10} | {row['acc_before']:.4f} → {row['acc_after']:.4f}", end="")
        if source_feat == target_feat:
            print(f" (baseline: {row['baseline']:.4f}, L2 train: {row['l2_train']:.4f}, L2 test: {row['l2_test']:.4f})", end="")
        print()