import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import os
import json

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
TEST_FILE = "Stage3/Embeddings/Narratives/4conc/Original_embeddings/gpt2_embeddings_test.pt"
RESULTS_FILE = "Stage3/Results/Narratives/4conc/oracle_leace_after_UDleace_Narr_oracle_results.json"
UD_ERASER_DIR = "Stage3/Leace_transf_matrices/UD/4conc/"
SEED = 42

torch.manual_seed(SEED)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# --------------------------
# Load Narratives test data
# --------------------------
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

X_test_orig = torch.cat([sent["embeddings_by_layer"][LAYER] for sent in test_data], dim=0)
labels_by_feature_test = {}
for feat in test_data[0]["word_labels"].keys():
    labels_by_feature_test[feat] = torch.tensor(
        [label for sent in test_data for label in sent["word_labels"][feat]]
    ).long()

all_pos_tags = sorted(list(labels_by_feature_test.keys()))
results = []

for source_feat in all_pos_tags:
    print(f"\n=== Erasing: {source_feat} ===")
    # Load the UD LEACE eraser
    with open(os.path.join(UD_ERASER_DIR, f"s3_leace_eraser_{source_feat}.pkl"), "rb") as f:
        eraser = pickle.load(f)

    # Apply UD LEACE to the Narratives test data
    X_test_leace = eraser(X_test_orig)

    # Now apply Oracle LEACE using the Narratives labels (oracle direction)
    y_test_source = labels_by_feature_test[source_feat]
    Z_test = y_test_source.unsqueeze(1).float().numpy()
    X_test_leace_np = X_test_leace.numpy()
    X_test_leace_oracle_np = oracle_leace(X_test_leace_np, Z_test)
    X_test_leace_oracle = torch.from_numpy(X_test_leace_oracle_np)

    # --- L2 norms for this diagonal ---
    l2_leace_oracle = torch.norm(X_test_leace - X_test_leace_oracle, dim=1).mean().item()
    l2_orig_leace = torch.norm(X_test_orig - X_test_leace, dim=1).mean().item()
    l2_orig_leace_oracle = torch.norm(X_test_orig - X_test_leace_oracle, dim=1).mean().item()
    # ----------------------------------

    scaler = StandardScaler()
    X_test_leace_np_scaled = scaler.fit_transform(X_test_leace.numpy())
    X_test_leace_oracle_np_scaled = scaler.fit_transform(X_test_leace_oracle.numpy())

    for target_feat in all_pos_tags:
        y_test_target = labels_by_feature_test[target_feat].numpy()

        # Probe before Oracle LEACE
        clf_before = LogisticRegression(max_iter=2000)
        clf_before.fit(X_test_leace_np_scaled, y_test_target)
        acc_before = clf_before.score(X_test_leace_np_scaled, y_test_target)

        # Probe after Oracle LEACE
        clf_after = LogisticRegression(max_iter=2000)
        clf_after.fit(X_test_leace_oracle_np_scaled, y_test_target)
        acc_after = clf_after.score(X_test_leace_oracle_np_scaled, y_test_target)

        results.append({
            "erased": source_feat,
            "probed": target_feat,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "baseline": None,  # will be set below for diagonal
            "l2_leace_oracle": l2_leace_oracle if source_feat == target_feat else None,
            "l2_orig_leace": l2_orig_leace if source_feat == target_feat else None,
            "l2_orig_leace_oracle": l2_orig_leace_oracle if source_feat == target_feat else None,
        })

    # Compute and store the majority class baseline for the diagonal
    if source_feat in labels_by_feature_test:
        y_majority = labels_by_feature_test[source_feat].numpy()
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_test_leace_np_scaled, y_majority)
        baseline_acc = dummy.score(X_test_leace_np_scaled, y_majority)
        for r in results:
            if r["erased"] == source_feat and r["probed"] == source_feat:
                r["baseline"] = baseline_acc

# Save Results
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

# Display Results
print("\nOracle LEACE (Narratives oracle) after UD LEACE Concept Erasure Matrix (POS tags)")
print("Erased → Probed | Acc (before) → Acc (after)")
for source_feat in all_pos_tags:
    for target_feat in all_pos_tags:
        row = next(r for r in results if r["erased"] == source_feat and r["probed"] == target_feat)
        print(f"{source_feat:16} → {target_feat:16} | {row['acc_before']:.4f} → {row['acc_after']:.4f}", end="")
        if source_feat == target_feat:
            print(f" (baseline: {row['baseline']:.4f}, L2 leace→oracle: {row['l2_leace_oracle']:.4f}, L2 orig→leace: {row['l2_orig_leace']:.4f}, L2 orig→leace+oracle: {row['l2_orig_leace_oracle']:.4f})", end="")
        print()