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
EMBEDDING_FILE = "Stage3/Embeddings/Narratives/4conc/Original_embeddings/gpt2_embeddings.pt"
TEST_FILE = "Stage3/Embeddings/Narratives/4conc/Original_embeddings/gpt2_embeddings_test.pt"
RESULTS_FILE = "Stage3/Results/Narratives/4conc/leace_probe_results.json"
BALANCE_CLASSES = False
SEED = 42

torch.manual_seed(SEED)

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# --------------------------
# Load data once
# --------------------------
# Load train data
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)

# Get all POS tags as features
all_pos_tags = sorted(list(data[0]["word_labels"].keys()))

labels_by_feature = {}
for feat in all_pos_tags:
    labels_by_feature[feat] = torch.tensor(
        [label for sent in data for label in sent["word_labels"][feat]]
    ).long()

# Load test data
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
# Evaluate all source → target combinations
# --------------------------
results = []

for source_feat in all_pos_tags:
    y = labels_by_feature[source_feat]
    y_test = labels_by_feature_test[source_feat]

    # Optional balancing (only on train)
    if BALANCE_CLASSES:
        idx_class_0 = (y == 0).nonzero(as_tuple=True)[0]
        idx_class_1 = (y == 1).nonzero(as_tuple=True)[0]
        min_class_size = min(len(idx_class_0), len(idx_class_1))
        idx_0_sampled = idx_class_0[torch.randperm(len(idx_class_0))[:min_class_size]]
        idx_1_sampled = idx_class_1[torch.randperm(len(idx_class_1))[:min_class_size]]
        selected_indices = torch.cat([idx_0_sampled, idx_1_sampled])
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        X_bal = X[selected_indices]
        y = y[selected_indices]
    else:
        X_bal = X

    # Scale
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_bal.numpy())
    X_test_np = scaler.transform(X_test.numpy())

    # Fit probe before LEACE
    clf_orig = LogisticRegression(max_iter=2000)
    clf_orig.fit(X_train_np, y.numpy())
    acc_orig = clf_orig.score(X_test_np, y_test.numpy())

    # Dummy
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_np, y.numpy())
    baseline_acc = baseline.score(X_test_np, y_test.numpy())

    # Apply LEACE
    Z = y.unsqueeze(1).float()
    eraser = LeaceEraser.fit(X_bal, Z)
    X_erased = eraser(X_bal)
    X_test_erased = eraser(X_test)

    # --- Save the eraser for this feature ---
    with open(f"Stage3/pyfiles/s3_leace_eraser_{source_feat}.pkl", "wb") as f:
        pickle.dump(eraser, f)
    # ----------------------------------------

    # --- Save the LEACE-erased embeddings for this feature ---
    with open(f"Stage3/pyfiles/s3_leace_embeddings_{source_feat}.pkl", "wb") as f:
        pickle.dump({
            "train_erased": X_erased.cpu(),
            "train_labels": y.cpu(),
        }, f)
    # ---------------------------------------------------------

    # --- Compute L2 norm between original and erased embeddings ---
    l2_train = torch.norm(X_bal - X_erased, dim=1).mean().item()
    l2_test = torch.norm(X_test - X_test_erased, dim=1).mean().item()
    print(f"L2 norm (train, {source_feat}): {l2_train:.4f}")
    print(f"L2 norm (test, {source_feat}): {l2_test:.4f}")
    # -------------------------------------------------------------

    X_train_e_np = scaler.fit_transform(X_erased.numpy())
    X_test_e_np = scaler.transform(X_test_erased.numpy())

    # Refit probe after LEACE on original concept
    clf_leace = LogisticRegression(max_iter=2000)
    clf_leace.fit(X_train_e_np, y.numpy())
    acc_erased = clf_leace.score(X_test_e_np, y_test.numpy())

    for target_feat in all_pos_tags:
        y2 = labels_by_feature[target_feat]
        y2_test = labels_by_feature_test[target_feat]

        # Before LEACE
        clf2_orig = LogisticRegression(max_iter=2000)
        clf2_orig.fit(X_train_np, y2.numpy())
        acc2_orig = clf2_orig.score(X_test_np, y2_test.numpy())

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
print("\n\nLEACE Concept Erasure Matrix (POS tags)")
print("Erased → Probed | Acc (before) → Acc (after)")
for source_feat in all_pos_tags:
    for target_feat in all_pos_tags:
        row = next(r for r in results if r["erased"] == source_feat and r["probed"] == target_feat)
        print(f"{source_feat:10} → {target_feat:10} | {row['acc_before']:.4f} → {row['acc_after']:.4f}", end="")
        if source_feat == target_feat:
            print(f" (baseline: {row['baseline']:.4f}, L2 train: {row['l2_train']:.4f}, L2 test: {row['l2_test']:.4f})", end="")
        print()