import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import os
import json

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Stage3/Embeddings/Narratives/4conc/Original_embeddings/gpt2_embeddings.pt"
TEST_FILE = "Stage3/Embeddings/Narratives/4conc/Original_embeddings/gpt2_embeddings_test.pt"
RESULTS_FILE = "Stage3/Results/Narratives/4conc/precomputed_UD_leace_results.json"
MATRIX_DIR = "/home/gpeeper/LEACE/Stage3/Leace_transf_matrices/UD/4conc/"
SEED = 42

torch.manual_seed(SEED)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# --------------------------
# Load data
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)
all_pos_tags = sorted(list(data[0]["word_labels"].keys()))
labels_by_feature = {}
for feat in all_pos_tags:
    labels_by_feature[feat] = torch.tensor(
        [label for sent in data for label in sent["word_labels"][feat]]
    ).long()

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

    # Load precomputed LEACE transformation matrix
    matrix_path = os.path.join(MATRIX_DIR, f"s3_leace_eraser_{source_feat}.pkl")
    with open(matrix_path, "rb") as f:
        leace_matrix = pickle.load(f)  # Should be a numpy array or torch tensor

    # Apply transformation: X' = leace_matrix(X)
    if hasattr(leace_matrix, "__call__"):
        X_erased = leace_matrix(X)
        X_test_erased = leace_matrix(X_test)
    else:
        if isinstance(leace_matrix, np.ndarray):
            leace_matrix = torch.from_numpy(leace_matrix)
        X_erased = X @ leace_matrix
        X_test_erased = X_test @ leace_matrix

    # Scale
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X.numpy())
    X_test_np = scaler.transform(X_test.numpy())
    X_train_e_np = scaler.fit_transform(X_erased.numpy())
    X_test_e_np = scaler.transform(X_test_erased.numpy())

    # Fit probe before LEACE
    clf_orig = LogisticRegression(max_iter=2000)
    clf_orig.fit(X_train_np, y.numpy())
    acc_orig = clf_orig.score(X_test_np, y_test.numpy())

    # Dummy
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_np, y.numpy())
    baseline_acc = baseline.score(X_test_np, y_test.numpy())

    # Fit probe after LEACE
    clf_leace = LogisticRegression(max_iter=2000)
    clf_leace.fit(X_train_e_np, y.numpy())
    acc_erased = clf_leace.score(X_test_e_np, y_test.numpy())

    # L2 norm
    l2_train = torch.norm(X - X_erased, dim=1).mean().item()
    l2_test = torch.norm(X_test - X_test_erased, dim=1).mean().item()

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

print("Done. Results saved to", RESULTS_FILE)