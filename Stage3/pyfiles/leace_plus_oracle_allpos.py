# oracle leace is only done on the training set of course
# We can't save the eraser object, but we can save the transformation matrix
import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import os
import json
import pandas as pd

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
LAYER = 8
EMBEDDING_FILE = "Stage3/Embeddings/UD/leace_embeddings_ALL_POS.pkl"
RESULTS_FILE = "Stage3/Results/UD/leace_plus_oracle_results_ALL_POS.json"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
SEED = 42

torch.manual_seed(SEED)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X = data["train_erased"]
labels_by_feature = data["train_labels"]
all_pos_tags = sorted(labels_by_feature.keys())

# --------------------------
# Oracle LEACE: Remove all POS information at once using train labels for train set
# --------------------------
# Stack all POS tag labels as columns to form the concept matrix Z_all
Z_all = torch.stack([labels_by_feature[feat] for feat in all_pos_tags], dim=1).float()

# Check the rank of Z_all
Z_all_np = Z_all.numpy()
rank = np.linalg.matrix_rank(Z_all_np)
print(f"Rank of Z_all: {rank} / {Z_all_np.shape[1]} (num POS tags)")

# Check correlation between columns of Z_all
corr = pd.DataFrame(Z_all_np).corr()
print("Correlation matrix of Z_all columns (first 5 rows):")
print(corr.iloc[:5, :5])

# Oracle LEACE: Fit eraser on train set using train labels, apply to train set
X_np = X.numpy()
Z_all_np = Z_all.numpy()
X_all_erased_np = oracle_leace(X_np, Z_all_np)
X_all_erased = torch.from_numpy(X_all_erased_np)

# Save the oracle-erased train embeddings and labels
with open("Stage3/Embeddings/UD/oracle_leace_embeddings_ALL_POS.pkl", "wb") as f:
    pickle.dump({
        "train_erased": X_all_erased.cpu(),
        "train_labels": {feat: labels_by_feature[feat].cpu() for feat in all_pos_tags},
    }, f)

# --- Save the eraser object ---
with open(f"Stage3/Eraser_objects/UD/leace_plus_oracle_eraser_ALL_POS.pkl", "wb") as f:
    pickle.dump({"eraser_matrix": None}, f)  # You can save the matrix if you want

# ----------------------------------------
# Compute L2 norm between original and oracle-erased train embeddings
l2_train_all = torch.norm(X - X_all_erased, dim=1).mean().item()
print(f"\nL2 norm (train, ALL_POS, oracle): {l2_train_all:.4f}")

# Evaluate probe accuracy for each POS tag after oracle erasure (on train set)
scaler_all = StandardScaler()
X_all_np = scaler_all.fit_transform(X_all_erased.numpy())

probe_results = {}
print("\nProbe accuracy after ORACLE erasure of ALL POS tags (on train set):")
for feat in all_pos_tags:
    y = labels_by_feature[feat]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_all_np, y.numpy())  # Oracle: fit and test on train set
    acc = clf.score(X_all_np, y.numpy())
    probe_results[feat] = acc
    print(f"{feat:10}: {acc:.4f}")

# --------------------------
# Compute overall POS accuracy (before and after oracle erasure)
# --------------------------
y_train_pos = torch.stack([labels_by_feature[feat] for feat in all_pos_tags], dim=1).argmax(dim=1)

# Before erasure (fit and eval on train set)
scaler_orig = StandardScaler()
X_orig_np = scaler_orig.fit_transform(X.numpy())
clf_pos = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf_pos.fit(X_orig_np, y_train_pos.numpy())
overall_acc_before = clf_pos.score(X_orig_np, y_train_pos.numpy())
print(f"\nOverall POS accuracy BEFORE erasure (train set): {overall_acc_before:.4f}")

# After oracle erasure (fit and eval on train set)
clf_pos_erased = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf_pos_erased.fit(X_all_np, y_train_pos.numpy())
overall_acc_after = clf_pos_erased.score(X_all_np, y_train_pos.numpy())
print(f"Overall POS accuracy AFTER oracle erasure (train set): {overall_acc_after:.4f}")

# Compute baseline: accuracy of always predicting the most frequent class
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_orig_np, y_train_pos.numpy())
baseline_acc = dummy.score(X_orig_np, y_train_pos.numpy())
print(f"Baseline (most frequent POS) accuracy: {baseline_acc:.4f}")

# Save results
results = []
results.append({
    "probe_accuracy": probe_results,
    "l2_train": l2_train_all,
    "overall_pos_accuracy_before": overall_acc_before,
    "overall_pos_accuracy_after": overall_acc_after,
    "baseline_accuracy": baseline_acc,
    "z_all_rank": int(rank),
})

# Save as JSON
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE}")