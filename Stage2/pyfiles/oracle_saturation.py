import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from collections import Counter
import os
import json

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
# Settings
# --------------------------
EMBEDDING_FILE = "Stage2/Embeddings/gpt2_embeddings.pt"
TEST_FILE = "Stage2/Embeddings/gpt2_embeddings_test.pt"
FEATURES = ["function_content", "noun_nonnoun", "verb_nonverb", "closed_open"]
LAYER = 8
BALANCE_CLASSES = False
SEED = 42

torch.manual_seed(SEED)

# --------------------------
# Load train data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)
labels_by_feature = {}
for feat in FEATURES:
    labels_by_feature[feat] = torch.tensor(
        [label for sent in data for label in sent["word_labels"][feat]]
    ).long()


# --------------------------
# Load test data
# --------------------------
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)
X_test_all = [sent["embeddings_by_layer"][LAYER] for sent in test_data]
X_test = torch.cat(X_test_all, dim=0)
labels_by_feature_test = {}
for feat in FEATURES:
    labels_by_feature_test[feat] = torch.tensor(
        [label for sent in test_data for label in sent["word_labels"][feat]]
    ).long()

# --------------------------
# Evaluate all source → target combinations
# --------------------------
FRACTIONS = [0.01, 0.1, 1.0]

for frac in FRACTIONS:
    print(f"\n\n=== Running Oracle LEACE for {int(frac*100)}% of the data ===")
    results = []

    # Subsample train data
    n_train = X.shape[0]
    n_sub = max(1, int(n_train * frac))
    idx_train = torch.randperm(n_train)[:n_sub]
    X_frac = X[idx_train]
    labels_by_feature_frac = {}
    for feat in FEATURES:
        labels_by_feature_frac[feat] = labels_by_feature[feat][idx_train]

    for source_feat in FEATURES:
        y = labels_by_feature_frac[source_feat]
        y_test = labels_by_feature_test[source_feat]

        if BALANCE_CLASSES:
            idx_class_0 = (y == 0).nonzero(as_tuple=True)[0]
            idx_class_1 = (y == 1).nonzero(as_tuple=True)[0]
            min_class_size = min(len(idx_class_0), len(idx_class_1))
            idx_0_sampled = idx_class_0[torch.randperm(len(idx_class_0))[:min_class_size]]
            idx_1_sampled = idx_class_1[torch.randperm(len(idx_class_1))[:min_class_size]]
            selected_indices = torch.cat([idx_0_sampled, idx_1_sampled])
            selected_indices = selected_indices[torch.randperm(len(selected_indices))]
            X_bal = X_frac[selected_indices]
            y = y[selected_indices]
        else:
            X_bal = X_frac

        # Scale
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_bal.numpy())
        X_test_np = scaler.transform(X_test.numpy())

        # Fit probe before LEACE
        clf_orig = LogisticRegression(max_iter=2000)
        clf_orig.fit(X_train_np, y.numpy())
        acc_orig = clf_orig.score(X_test_np, y_test.numpy())

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train_np, y.numpy())
        dummy_acc = dummy.score(X_test_np, y_test.numpy())

        # Oracle LEACE
        Z_np = y.unsqueeze(1).float().numpy()
        X_bal_np = X_bal.numpy()
        X_erased_np = oracle_leace(X_bal_np, Z_np)
        X_erased = torch.from_numpy(X_erased_np)
        Z_test_np = y_test.unsqueeze(1).float().numpy()
        X_test_erased_np = oracle_leace(X_test.numpy(), Z_test_np)
        X_test_erased = torch.from_numpy(X_test_erased_np)

        # --- Compute L2 norm between original and erased embeddings ---
        l2_train = torch.norm(X_bal - X_erased, dim=1).mean().item()
        l2_test = torch.norm(X_test - X_test_erased, dim=1).mean().item()
        print(f"L2 norm (train, {source_feat}, frac={frac}): {l2_train:.4f}")
        print(f"L2 norm (test, {source_feat}, frac={frac}): {l2_test:.4f}")
        # -------------------------------------------------------------

        X_train_e_np = scaler.fit_transform(X_erased.numpy())
        X_test_e_np = scaler.transform(X_test_erased.numpy())

        clf_leace = LogisticRegression(max_iter=2000)
        clf_leace.fit(X_train_e_np, y.numpy())
        acc_erased = clf_leace.score(X_test_e_np, y_test.numpy())

        for target_feat in FEATURES:
            y2 = labels_by_feature_frac[target_feat]
            y2_test = labels_by_feature_test[target_feat]

            clf2_orig = LogisticRegression(max_iter=2000)
            clf2_orig.fit(X_train_np, y2.numpy())
            acc2_orig = clf2_orig.score(X_test_np, y2_test.numpy())

            clf2_leace = LogisticRegression(max_iter=2000)
            clf2_leace.fit(X_train_e_np, y2.numpy())
            acc2_leace = clf2_leace.score(X_test_e_np, y2_test.numpy())

            results.append({
                "erased": source_feat,
                "probed": target_feat,
                "acc_before": acc2_orig,
                "acc_after": acc2_leace,
                "erasure_accuracy": acc_erased if source_feat == target_feat else None,
                "dummy": dummy_acc if source_feat == target_feat else None,
                "l2_train": l2_train if source_feat == target_feat else None,
                "l2_test": l2_test if source_feat == target_feat else None,
                "fraction": frac
            })

    # Save results for this fraction
    out_file = f"Stage2/Results/Oracle_saturation/oracle_saturation_{int(frac*100)}pct.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Display Results
    print(f"\n\nOracle LEACE Concept Erasure Matrix (fraction={frac})")
    print("Erased → Probed | Acc (before) → Acc (after)")
    for source_feat in FEATURES:
        for target_feat in FEATURES:
            row = next(r for r in results if r["erased"] == source_feat and r["probed"] == target_feat)
            print(f"{source_feat:16} → {target_feat:16} | {row['acc_before']:.4f} → {row['acc_after']:.4f}", end="")
            if source_feat == target_feat:
                print(f" (dummy: {row['dummy']:.4f}, L2 train: {row['l2_train']:.2f}, L2 test: {row['l2_test']:.2f})", end="")
            print()
