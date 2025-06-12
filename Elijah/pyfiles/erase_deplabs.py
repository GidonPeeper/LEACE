# This file erases dependency labels from Narratives sentences with Oracle LEACE
import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import os
import json

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
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Elijah/Embeddings/original_embeddings_deplabs.pkl"
RESULTS_FILE = "Elijah/Results/remove_deplabs.json"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)

all_deplabs = sorted(list(data[0]["word_labels"].keys()))
labels_by_feature = {
    feat: torch.tensor([label for sent in data for label in sent["word_labels"][feat]]).long()
    for feat in all_deplabs
}

# --------------------------
# Oracle LEACE: Remove all dependency label information at once
# --------------------------
# Create multiclass label vector and one-hot encode
y_train_dep = torch.stack([labels_by_feature[feat] for feat in all_deplabs], dim=1).argmax(dim=1)
num_dep = len(all_deplabs)
y_train_onehot = torch.nn.functional.one_hot(y_train_dep, num_classes=num_dep).float()
Z_all = y_train_onehot[:, :-1]  # Drop last column for full rank

# Check the rank of Z_all
Z_all_np = Z_all.numpy()
rank = np.linalg.matrix_rank(Z_all_np)
print(f"Rank of Z_all: {rank} / {Z_all_np.shape[1]} (num dependency labels minus one)")

# Oracle LEACE: Fit eraser on train set using train labels, apply to train set
X_np = X.numpy()
X_all_erased_np, projection = oracle_leace(X_np, Z_all_np)
X_all_erased = torch.from_numpy(X_all_erased_np)

# Save the oracle-erased train embeddings and labels
with open("Elijah/Embeddings/narratives_without_deplabs.pkl", "wb") as f:
    pickle.dump({
        "train_erased": X_all_erased.cpu(),
        "train_labels": {feat: labels_by_feature[feat].cpu() for feat in all_deplabs},
    }, f)

# Ensure the eraser objects directory exists
os.makedirs("Elijah/Eraser_objects", exist_ok=True)

# Save the projection matrix as a separate file
with open("Elijah/Eraser_objects/oracle_transf_matrix_deplabs.pkl", "wb") as f:
    pickle.dump(projection, f)

# ----------------------------------------
# Compute L2 norm between original and oracle-erased train embeddings
l2_train_all = torch.norm(X - X_all_erased, dim=1).mean().item()
print(f"\nL2 norm (train, ALL_DEPLABS, oracle): {l2_train_all:.4f}")

# Evaluate probe accuracy for each dependency label after oracle erasure (on train set)
scaler_all = StandardScaler()
X_all_np_scaled = scaler_all.fit_transform(X_all_erased.numpy())

probe_results = {}
print("\nProbe accuracy after ORACLE erasure of ALL dependency labels (on train set):")
for feat in all_deplabs:
    y = labels_by_feature[feat]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_all_np_scaled, y.numpy())  # Oracle: fit and test on train set
    acc = clf.score(X_all_np_scaled, y.numpy())
    probe_results[feat] = acc
    print(f"{feat:15}: {acc:.4f}")

# --------------------------
# Compute overall dependency label accuracy (before and after oracle erasure)
# --------------------------
# Multiclass probe
scaler_orig = StandardScaler()
X_orig_np = scaler_orig.fit_transform(X.numpy())
clf_dep = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf_dep.fit(X_orig_np, y_train_dep.numpy())
overall_acc_before = clf_dep.score(X_orig_np, y_train_dep.numpy())
print(f"\nOverall dependency label accuracy BEFORE erasure (train set): {overall_acc_before:.4f}")

clf_dep_erased = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf_dep_erased.fit(X_all_np_scaled, y_train_dep.numpy())
overall_acc_after = clf_dep_erased.score(X_all_np_scaled, y_train_dep.numpy())
print(f"Overall dependency label accuracy AFTER oracle erasure (train set): {overall_acc_after:.4f}")

# Compute baseline: accuracy of always predicting the most frequent class
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_orig_np, y_train_dep.numpy())
baseline_acc = dummy.score(X_orig_np, y_train_dep.numpy())
print(f"Baseline (most frequent dependency label) accuracy: {baseline_acc:.4f}")

labels = list(range(len(all_deplabs)))
y_pred = clf_dep_erased.predict(X_all_np_scaled)
print(classification_report(y_train_dep.numpy(), y_pred, labels=labels, target_names=all_deplabs))

# Save results
results = [{
    "probe_accuracy": probe_results,
    "l2_train": l2_train_all,
    "overall_deplab_accuracy_before": overall_acc_before,
    "overall_deplab_accuracy_after": overall_acc_after,
    "baseline_accuracy": baseline_acc,
    "z_all_rank": int(rank)
}]

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE}")