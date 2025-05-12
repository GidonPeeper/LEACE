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
LAYER = 8
EMBEDDING_FILE = "gpt2_embeddings.pt"
FEATURE = "closed_open"
BALANCE_CLASSES = False
SEED = 42
FRACTION = 1.0  # Use 1.0 or 0.5

torch.manual_seed(SEED)

# --------------------------
# Load data once
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)  # shape: [N, hidden_dim]

y_all = torch.tensor([label for sent in data for label in sent["word_labels"][FEATURE]]).long()
selected_indices = torch.arange(len(y_all))

if BALANCE_CLASSES:
    idx_class_0 = (y_all == 0).nonzero(as_tuple=True)[0]
    idx_class_1 = (y_all == 1).nonzero(as_tuple=True)[0]
    min_class_size = min(len(idx_class_0), len(idx_class_1))
    idx_0_sampled = idx_class_0[torch.randperm(len(idx_class_0))[:min_class_size]]
    idx_1_sampled = idx_class_1[torch.randperm(len(idx_class_1))[:min_class_size]]
    selected_indices = torch.cat([idx_0_sampled, idx_1_sampled])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]

if FRACTION < 1.0:
    n = int(len(selected_indices) * FRACTION)
    selected_indices = selected_indices[torch.randperm(len(selected_indices))[:n]]

X_sub = X[selected_indices]
y_sub = y_all[selected_indices]

# --------------------------
# Split
# --------------------------
N = X_sub.shape[0]
train_size = int(0.8 * N)
val_size = N - train_size
dataset = TensorDataset(X_sub, y_sub)
train_set, val_set = random_split(dataset, [train_size, val_size])
X_train, y_train = train_set[:][0], train_set[:][1]
X_val, y_val = val_set[:][0], val_set[:][1]

# --------------------------
# Scale
# --------------------------
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train.numpy())
X_val_np = scaler.transform(X_val.numpy())

# --------------------------
# Probe before LEACE
# --------------------------
clf_orig = LogisticRegression(max_iter=2000)
clf_orig.fit(X_train_np, y_train.numpy())
acc_orig = clf_orig.score(X_val_np, y_val.numpy())

# --------------------------
# Dummy baseline
# --------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_np, y_train.numpy())
dummy_acc = dummy.score(X_val_np, y_val.numpy())

# --------------------------
# Apply LEACE twice
# --------------------------
Z = y_sub.unsqueeze(1).float()

eraser1 = LeaceEraser.fit(X_sub, Z)
X1 = eraser1(X_sub)

eraser2 = LeaceEraser.fit(X1, Z)
X2 = eraser2(X1)

# --------------------------
# Probe after second LEACE
# --------------------------
X_train_e = X2[train_set.indices]
X_val_e = X2[val_set.indices]
X_train_e_np = scaler.fit_transform(X_train_e.numpy())
X_val_e_np = scaler.transform(X_val_e.numpy())

clf_leace = LogisticRegression(max_iter=2000)
clf_leace.fit(X_train_e_np, y_train.numpy())
acc_erased = clf_leace.score(X_val_e_np, y_val.numpy())

# --------------------------
# Print Results
# --------------------------
print("\nLEACE Double Application Result")
print(f"Concept: {FEATURE}")
print(f"Accuracy before LEACE: {acc_orig:.4f}")
print(f"Accuracy after 2x LEACE: {acc_erased:.4f}")
print(f"Dummy baseline accuracy: {dummy_acc:.4f}")
