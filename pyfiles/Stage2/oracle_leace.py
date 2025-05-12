import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from collections import Counter

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
EMBEDDING_FILE = "gpt2_embeddings.pt"
FEATURES = ["function_content", "noun_nonnoun", "verb_nonverb", "closed_open"]
BALANCE_CLASSES = False
SEED = 42

torch.manual_seed(SEED)

# --------------------------
# Load data once
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)  # shape: [N, hidden_dim]

# Load all label sets
labels_by_feature = {}
for feat in FEATURES:
    labels_by_feature[feat] = torch.tensor(
        [label for sent in data for label in sent["word_labels"][feat]]
    ).long()

# --------------------------
# Evaluate all source → target combinations
# --------------------------
results = []

for source_feat in FEATURES:
    y = labels_by_feature[source_feat]

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
        selected_indices = torch.arange(len(y))

    N = X_bal.shape[0]
    train_size = int(0.8 * N)
    val_size = N - train_size
    dataset = TensorDataset(X_bal, y)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    X_train, y_train = train_set[:][0], train_set[:][1]
    X_val, y_val = val_set[:][0], val_set[:][1]

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.numpy())
    X_val_np = scaler.transform(X_val.numpy())

    clf_orig = LogisticRegression(max_iter=2000)
    clf_orig.fit(X_train_np, y_train.numpy())
    acc_orig = clf_orig.score(X_val_np, y_val.numpy())

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_np, y_train.numpy())
    dummy_acc = dummy.score(X_val_np, y_val.numpy())

    # Oracle LEACE
    Z_np = y.unsqueeze(1).float().numpy()
    X_bal_np = X_bal.numpy()
    X_erased_np = oracle_leace(X_bal_np, Z_np)
    X_erased = torch.from_numpy(X_erased_np)

    X_train_e = X_erased[train_set.indices]
    X_val_e = X_erased[val_set.indices]
    X_train_e_np = scaler.fit_transform(X_train_e.numpy())
    X_val_e_np = scaler.transform(X_val_e.numpy())

    clf_leace = LogisticRegression(max_iter=2000)
    clf_leace.fit(X_train_e_np, y_train.numpy())
    acc_erased = clf_leace.score(X_val_e_np, y_val.numpy())

    for target_feat in FEATURES:
        y2 = labels_by_feature[target_feat][selected_indices]
        y2_train = y2[train_set.indices]
        y2_val = y2[val_set.indices]

        clf2_orig = LogisticRegression(max_iter=2000)
        clf2_orig.fit(X_train_np, y2_train.numpy())
        acc2_orig = clf2_orig.score(X_val_np, y2_val.numpy())

        clf2_leace = LogisticRegression(max_iter=2000)
        clf2_leace.fit(X_train_e_np, y2_train.numpy())
        acc2_leace = clf2_leace.score(X_val_e_np, y2_val.numpy())

        results.append({
            "erased": source_feat,
            "probed": target_feat,
            "acc_before": acc2_orig,
            "acc_after": acc2_leace,
            "erasure_accuracy": acc_erased if source_feat == target_feat else None,
            "dummy": dummy_acc if source_feat == target_feat else None
        })

# --------------------------
# Display Results
# --------------------------
print("\n\nOracle LEACE Concept Erasure Matrix")
print("Erased → Probed | Acc (before) → Acc (after)")
for source_feat in FEATURES:
    for target_feat in FEATURES:
        row = next(r for r in results if r["erased"] == source_feat and r["probed"] == target_feat)
        print(f"{source_feat:16} → {target_feat:16} | {row['acc_before']:.4f} → {row['acc_after']:.4f}", end="")
        if source_feat == target_feat:
            print(f" (dummy: {row['dummy']:.4f})", end="")
        print()
