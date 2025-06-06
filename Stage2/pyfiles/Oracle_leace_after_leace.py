import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

def oracle_leace(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Z_centered = Z - np.mean(Z, axis=0, keepdims=True)
    Σ_XZ = X_centered.T @ Z_centered / X.shape[0]
    Σ_ZZ = Z_centered.T @ Z_centered / X.shape[0]
    Σ_ZZ_inv = np.linalg.pinv(Σ_ZZ)
    projection = Σ_XZ @ Σ_ZZ_inv @ Z_centered.T
    X_erased = X - projection.T
    return X_erased

FEATURES = ["function_content", "noun_nonnoun", "verb_nonverb", "closed_open"]

# Load original test embeddings and labels
with open("gpt2_embeddings_test.pt", "rb") as f:
    test_data = pickle.load(f)

# Prepare all test embeddings and labels for all features
if isinstance(test_data, list):
    X_test_orig = torch.cat([sent["embeddings_by_layer"][8] for sent in test_data], dim=0)
    labels_by_feature_test = {}
    for feat in FEATURES:
        labels_by_feature_test[feat] = torch.tensor(
            [label for sent in test_data for label in sent["word_labels"][feat]]
        ).long()
else:
    X_test_orig = test_data["embeddings"]
    labels_by_feature_test = {feat: test_data["labels"] for feat in FEATURES}

results = []

for source_feat in FEATURES:
    print(f"\n=== Erasing: {source_feat} ===")
    # Load the LEACE eraser trained on the training set
    with open(f"s2_leace_eraser_{source_feat}.pkl", "rb") as f:
        eraser = pickle.load(f)

    # Apply normal LEACE to the test data
    X_test_leace = eraser(X_test_orig)

    # Apply Oracle LEACE to the LEACE-erased test embeddings
    y_test_source = labels_by_feature_test[source_feat]
    Z_test = y_test_source.unsqueeze(1).float().numpy()
    X_test_leace_np = X_test_leace.numpy()
    X_test_leace_oracle_np = oracle_leace(X_test_leace_np, Z_test)
    X_test_leace_oracle = torch.from_numpy(X_test_leace_oracle_np)

    # --- L2 norms for this diagonal ---
    l2_orig_leace = torch.norm(X_test_orig - X_test_leace, dim=1).mean().item()
    l2_orig_leace_oracle = torch.norm(X_test_orig - X_test_leace_oracle, dim=1).mean().item()
    # ----------------------------------

    scaler = StandardScaler()
    X_test_leace_np_scaled = scaler.fit_transform(X_test_leace.numpy())
    X_test_leace_oracle_np_scaled = scaler.fit_transform(X_test_leace_oracle.numpy())

    for target_feat in FEATURES:
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

# Display Results
print("\nOracle LEACE after LEACE Concept Erasure Matrix")
print("Erased → Probed | Acc (before) → Acc (after)")
for source_feat in FEATURES:
    for target_feat in FEATURES:
        row = next(r for r in results if r["erased"] == source_feat and r["probed"] == target_feat)
        print(f"{source_feat:16} → {target_feat:16} | {row['acc_before']:.4f} → {row['acc_after']:.4f}", end="")
        if source_feat == target_feat:
            print(f" (baseline: {row['baseline']:.4f}, L2 orig→leace: {row['l2_orig_leace']:.4f}, L2 orig→leace+oracle: {row['l2_orig_leace_oracle']:.4f})", end="")
        print()
