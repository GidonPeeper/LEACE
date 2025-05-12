import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from concept_erasure import LeaceEraser
from collections import Counter

# --------------------------
# Settings
# --------------------------
LAYER = 4
EMBEDDING_FILE = "gpt2_embeddings.pt"
FEATURES = ["function_content", "noun_nonnoun", "verb_nonverb", "closed_open"]
BALANCE_CLASSES = False
SEED = 42
FRACTIONS = [0.1, 0.5, 1.0]

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
# Evaluate for each concept at each data size
# --------------------------
results = []

for source_feat in FEATURES:
    for frac in FRACTIONS:
        print(f"\n--- {source_feat.upper()} | Using {int(frac * 100)}% of data ---")

        y = labels_by_feature[source_feat]
        selected_indices = torch.arange(len(y))

        if BALANCE_CLASSES:
            idx_class_0 = (y == 0).nonzero(as_tuple=True)[0]
            idx_class_1 = (y == 1).nonzero(as_tuple=True)[0]
            min_class_size = min(len(idx_class_0), len(idx_class_1))
            idx_0_sampled = idx_class_0[torch.randperm(len(idx_class_0))[:min_class_size]]
            idx_1_sampled = idx_class_1[torch.randperm(len(idx_class_1))[:min_class_size]]
            selected_indices = torch.cat([idx_0_sampled, idx_1_sampled])
            selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        if frac < 1.0:
            n = int(len(selected_indices) * frac)
            selected_indices = selected_indices[torch.randperm(len(selected_indices))[:n]]

        X_sub = X[selected_indices]
        y_sub = y[selected_indices]

        # Split
        N = X_sub.shape[0]
        train_size = int(0.8 * N)
        val_size = N - train_size
        dataset = TensorDataset(X_sub, y_sub)
        train_set, val_set = random_split(dataset, [train_size, val_size])
        X_train, y_train = train_set[:][0], train_set[:][1]
        X_val, y_val = val_set[:][0], val_set[:][1]

        # Scale
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train.numpy())
        X_val_np = scaler.transform(X_val.numpy())

        # Fit probe before LEACE
        clf_orig = LogisticRegression(max_iter=2000)
        clf_orig.fit(X_train_np, y_train.numpy())
        acc_orig = clf_orig.score(X_val_np, y_val.numpy())

        # Dummy
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train_np, y_train.numpy())
        dummy_acc = dummy.score(X_val_np, y_val.numpy())

        # Apply LEACE
        Z = y_sub.unsqueeze(1).float()
        eraser = LeaceEraser.fit(X_sub, Z)
        X_erased = eraser(X_sub)
        X_train_e = X_erased[train_set.indices]
        X_val_e = X_erased[val_set.indices]
        X_train_e_np = scaler.fit_transform(X_train_e.numpy())
        X_val_e_np = scaler.transform(X_val_e.numpy())

        # Refit probe after LEACE
        clf_leace = LogisticRegression(max_iter=2000)
        clf_leace.fit(X_train_e_np, y_train.numpy())
        acc_erased = clf_leace.score(X_val_e_np, y_val.numpy())

        results.append({
            "feature": source_feat,
            "fraction": frac,
            "acc_before": acc_orig,
            "acc_after": acc_erased,
            "dummy": dummy_acc
        })

# --------------------------
# Print Summary
# --------------------------
print("\n\nLEACE Saturation Test Results")
for row in results:
    print(f"{row['feature']:16} | {int(row['fraction']*100)}% | {row['acc_before']:.4f} → {row['acc_after']:.4f} (dummy: {row['dummy']:.4f})")
