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
# Parameters
# --------------------------
LAYER = 8
FEATURE = "closed_open"
EMBEDDING_FILE = "gpt2_embeddings.pt"
BALANCE_CLASSES = True  # <- SET THIS TO True to balance classes
SEED = 42

torch.manual_seed(SEED)

# --------------------------
# Load embeddings and labels
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

X_list, y_list = [], []

for sent in data:
    X_list.append(sent["embeddings_by_layer"][LAYER])  # Tensor: [num_words, dim]
    y_list.extend(sent["word_labels"][FEATURE])         # List[int]

X = torch.cat(X_list, dim=0)           # shape: [N, hidden_dim]
y = torch.tensor(y_list).long()        # shape: [N]

# --------------------------
# Report class distribution
# --------------------------
counts = Counter(y_list)
total = sum(counts.values())
print(f"\n[CLASS DISTRIBUTION] ({FEATURE}):")
for cls, count in sorted(counts.items()):
    print(f"  Class {cls}: {count} tokens ({count / total:.2%})")

print(f"Loaded {X.shape[0]} tokens, embedding dim = {X.shape[1]}")

# --------------------------
# Optional: balance the classes
# --------------------------
if BALANCE_CLASSES:
    print("\n[INFO] Balancing classes by downsampling majority class...")

    idx_class_0 = (y == 0).nonzero(as_tuple=True)[0]
    idx_class_1 = (y == 1).nonzero(as_tuple=True)[0]

    min_class_size = min(len(idx_class_0), len(idx_class_1))

    idx_0_sampled = idx_class_0[torch.randperm(len(idx_class_0))[:min_class_size]]
    idx_1_sampled = idx_class_1[torch.randperm(len(idx_class_1))[:min_class_size]]
    selected_indices = torch.cat([idx_0_sampled, idx_1_sampled])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]

    X = X[selected_indices]
    y = y[selected_indices]

    print(f"[INFO] After balancing: {X.shape[0]} samples (equal per class)")

# --------------------------
# Train-test split
# --------------------------
N = X.shape[0]
train_size = int(0.8 * N)
val_size = N - train_size

dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [train_size, val_size])

X_train, y_train = train_set[:][0], train_set[:][1]
X_val, y_val = val_set[:][0], val_set[:][1]

# --------------------------
# Scale the data
# --------------------------
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train.numpy())
X_val_np = scaler.transform(X_val.numpy())

# --------------------------
# Fit probe on original embeddings
# --------------------------
clf_orig = LogisticRegression(max_iter=2000)
clf_orig.fit(X_train_np, y_train.numpy())
y_pred_orig = clf_orig.predict(X_val_np)
acc_orig = accuracy_score(y_val.numpy(), y_pred_orig)
print(f"[BEFORE LEACE] Probe accuracy: {acc_orig:.4f}")

# --------------------------
# Dummy baseline
# --------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_np, y_train.numpy())
dummy_acc = dummy.score(X_val_np, y_val.numpy())
print(f"[DUMMY BASELINE] Majority class accuracy: {dummy_acc:.4f}")

# --------------------------
# Apply LEACE
# --------------------------
Z = y.unsqueeze(1).float()
eraser = LeaceEraser.fit(X, Z)
X_erased = eraser(X)

# --------------------------
# Re-split and scale erased data
# --------------------------
X_train_erased = X_erased[train_set.indices]
X_val_erased = X_erased[val_set.indices]

X_train_erased_np = scaler.fit_transform(X_train_erased.numpy())
X_val_erased_np = scaler.transform(X_val_erased.numpy())

# --------------------------
# Fit probe on LEACE-erased embeddings
# --------------------------
clf_leace = LogisticRegression(max_iter=2000)
clf_leace.fit(X_train_erased_np, y_train.numpy())
y_pred_leace = clf_leace.predict(X_val_erased_np)
acc_leace = accuracy_score(y_val.numpy(), y_pred_leace)
print(f"[AFTER LEACE ] Probe accuracy: {acc_leace:.4f}")

