import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from concept_erasure import LeaceEraser
from collections import Counter
import os
import json

# --------------------------
# Settings, this file can be used for full erasure of both deplabs and all POS tags
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings.pt"
TEST_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings_test.pt"
RESULTS_FILE = "Full_concept_erasure/Results/leace_results_ALL_DEPLABS.json"
# EMBEDDING_FILE = "Stage3/Embeddings/UD/AllPOS/Original_embeddings/gpt2_embeddings.pt"
# TEST_FILE = "Stage3/Embeddings/UD/AllPOS/Original_embeddings/gpt2_embeddings_test.pt"
# RESULTS_FILE = "Full_concept_erasure/Results/leace_results_ALL_POS.json"
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

all_pos_tags = sorted(list(data[0]["word_labels"].keys()))
labels_by_feature = {
    feat: torch.tensor([label for sent in data for label in sent["word_labels"][feat]]).long()
    for feat in all_pos_tags
}

with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)
X_test_all = [sent["embeddings_by_layer"][LAYER] for sent in test_data]
X_test = torch.cat(X_test_all, dim=0)
labels_by_feature_test = {
    feat: torch.tensor([label for sent in test_data for label in sent["word_labels"][feat]]).long()
    for feat in all_pos_tags
}

# --------------------------
# Create multiclass label vector for LEACE
# --------------------------
pos_to_idx = {pos: i for i, pos in enumerate(all_pos_tags)}
y_train_pos = torch.stack([labels_by_feature[feat] for feat in all_pos_tags], dim=1).argmax(dim=1)
y_test_pos = torch.stack([labels_by_feature_test[feat] for feat in all_pos_tags], dim=1).argmax(dim=1)

num_pos = len(all_pos_tags)
y_train_onehot = torch.nn.functional.one_hot(y_train_pos, num_classes=num_pos).float()
Z_all = y_train_onehot[:, :-1]  # Drop last column for full rank

# --------------------------
# Apply LEACE
# --------------------------
Z_all_np = Z_all.numpy()
rank = np.linalg.matrix_rank(Z_all_np)
print(f"Rank of Z_all: {rank} / {Z_all_np.shape[1]}")

eraser_all = LeaceEraser.fit(X, Z_all)
X_all_erased = eraser_all(X)
X_test_all_erased = eraser_all(X_test)

# # Save embeddings and eraser
# with open("Full_concept_erasure/Erased_embeddings/leace_embeddings_ALL_POS.pkl", "wb") as f:
#     pickle.dump({
#         "train_erased": X_all_erased.cpu(),
#         "test_erased": X_test_all_erased.cpu(),
#         "train_labels": y_train_pos.cpu(),
#         "test_labels": y_test_pos.cpu(),
#     }, f)
# with open("Full_concept_erasure/Eraser_objects/leace_eraser_ALL_POS.pkl", "wb") as f:
#     pickle.dump(eraser_all, f)
# Save embeddings and eraser
with open("Full_concept_erasure/Erased_embeddings/leace_embeddings_ALL_DEPLABS.pkl", "wb") as f:
    pickle.dump({
        "train_erased": X_all_erased.cpu(),
        "test_erased": X_test_all_erased.cpu(),
        "train_labels": y_train_pos.cpu(),
        "test_labels": y_test_pos.cpu(),
    }, f)
with open("Full_concept_erasure/Eraser_objects/leace_eraser_ALL_DEPLABS.pkl", "wb") as f:
    pickle.dump(eraser_all, f)

# --------------------------
# L2 distance
# --------------------------
l2_train_all = torch.norm(X - X_all_erased, dim=1).mean().item()
l2_test_all = torch.norm(X_test - X_test_all_erased, dim=1).mean().item()
print(f"\nL2 norm (train): {l2_train_all:.4f}")
print(f"L2 norm (test): {l2_test_all:.4f}")

# --------------------------
# Probe evaluation
# --------------------------
scaler_all = StandardScaler()
X_train_all_np = scaler_all.fit_transform(X_all_erased.numpy())
X_test_all_np = scaler_all.transform(X_test_all_erased.numpy())

print("\nPer-tag probe accuracy AFTER erasure:")
probe_results = {}
for feat in all_pos_tags:
    y = labels_by_feature[feat]
    y_test = labels_by_feature_test[feat]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_all_np, y.numpy())
    acc = clf.score(X_test_all_np, y_test.numpy())
    probe_results[feat] = acc
    print(f"{feat:10}: {acc:.4f}")

# --------------------------
# Multiclass POS accuracy before and after (already on test set)
# --------------------------
scaler_orig = StandardScaler()
X_train_orig_np = scaler_orig.fit_transform(X.numpy())
X_test_orig_np = scaler_orig.transform(X_test.numpy())

clf_pos = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf_pos.fit(X_train_orig_np, y_train_pos.numpy())
overall_acc_before = clf_pos.score(X_test_orig_np, y_test_pos.numpy())
print(f"\nOverall POS accuracy BEFORE erasure (test set): {overall_acc_before:.4f}")

clf_pos_erased = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf_pos_erased.fit(X_train_all_np, y_train_pos.numpy())
overall_acc_after = clf_pos_erased.score(X_test_all_np, y_test_pos.numpy())
print(f"Overall POS accuracy AFTER erasure (test set): {overall_acc_after:.4f}")

# Add: Training set accuracy before and after erasure for direct comparison with Oracle LEACE
train_acc_before = clf_pos.score(X_train_orig_np, y_train_pos.numpy())
print(f"Overall POS accuracy BEFORE erasure (train set): {train_acc_before:.4f}")

train_acc_after = clf_pos_erased.score(X_train_all_np, y_train_pos.numpy())
print(f"Overall POS accuracy AFTER erasure (train set): {train_acc_after:.4f}")

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_orig_np, y_train_pos.numpy())
baseline_acc = dummy.score(X_test_orig_np, y_test_pos.numpy())
print(f"Baseline (most frequent POS) accuracy: {baseline_acc:.4f}")

y_pred = clf_pos.predict(X_test_orig_np)
labels = list(range(len(all_pos_tags)))
print(classification_report(y_test_pos.numpy(), y_pred, labels=labels, target_names=all_pos_tags))

# --------------------------
# Save results
# --------------------------
results = [{
    "probe_accuracy": probe_results,
    "l2_train": l2_train_all,
    "l2_test": l2_test_all,
    "overall_pos_accuracy_before": overall_acc_before,
    "overall_pos_accuracy_after": overall_acc_after,
    "baseline_accuracy": baseline_acc,
    "z_all_rank": int(rank),
    "train_acc_before": train_acc_before,
    "train_acc_after": train_acc_after,
}]
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE}")
