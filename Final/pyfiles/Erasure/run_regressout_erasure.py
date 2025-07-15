"""
Performs concept erasure using the "Regressing Out" baseline method.

The method replaces embeddings entirely with the residuals of a linear regression.
The regression is trained on 80% of the data. This script saves both the full
"erased" (residual) dataset for probing, and a results file containing the
L2 distance calculated on the held-out test set.
"""
import argparse
import pickle
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================================
# 1. OLS Fitter for Regression
# =========================================================================
class OLSFitter:
    """A class to fit an OLS regression and compute residuals."""
    def __init__(self):
        self.W = None
        self.mean_x_train = None
        self.mean_z_train = None

    def fit(self, X_train: np.ndarray, Z_train: np.ndarray):
        """Learns the regression from Z to X on the training data."""
        self.mean_x_train = np.mean(X_train, axis=0, keepdims=True)
        self.mean_z_train = np.mean(Z_train, axis=0, keepdims=True)
        
        X_train_c = X_train.astype(np.float64) - self.mean_x_train
        Z_train_c = Z_train.astype(np.float64) - self.mean_z_train
        
        Σ_XZ = X_train_c.T @ Z_train_c
        Σ_ZZ = Z_train_c.T @ Z_train_c
        
        self.W = Σ_XZ @ np.linalg.pinv(Σ_ZZ)
        self.W = self.W.astype(X_train.dtype)

    def get_residuals(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Computes the residuals R = X_centered - X_pred_centered."""
        if self.W is None: raise RuntimeError("Must call .fit() before using.")
        
        X_c = X - self.mean_x_train
        Z_c = Z - self.mean_z_train
        
        X_pred_c = Z_c @ self.W.T
        Residuals = X_c - X_pred_c
        
        return Residuals

# =========================================================================
# 2. Main Script Logic
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Regressing-Out erasure baseline.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--concept", choices=["pos", "deplab", "sd", "ld", "pe"], required=True)
    parser.add_argument("--scaling", choices=["on", "off"], default="off", help="Enable StandardScaler before erasure ('on') or not ('off').")
    parser.add_argument("--data_fraction", type=int, choices=[1, 10, 100], default=100, help="Percentage of the dataset to use, corresponding to the pre-generated embedding file.")
    args = parser.parse_args()

    # --- Setup ---
    LAYER = 8; SEED = 42
    np.random.seed(SEED)
    dataset_name = "nar" if args.dataset == "narratives" else "ud"
    dataset_dir_name_base = "UD" if args.dataset == "ud" else "Narratives"
    
    concept_key_map = { 
        "pos": "pos_tags", "deplab": "deplabs", "sd": "sd", "ld": "ld", "pe": "pe" 
    }
    internal_concept_key = concept_key_map[args.concept]
    
    # --- File Paths ---
    file_suffix = f"_{args.data_fraction}pct" if args.data_fraction < 100 else ""
    scaling_suffix = "_Scaled" if args.scaling == "on" else "_Unscaled"
    dataset_dir_name_probe = dataset_dir_name_base + scaling_suffix
    base_dir = "Final"
    emb_file = f"{base_dir}/Embeddings/Original/{dataset_dir_name_base}/Embed_{dataset_name}_{args.concept}{file_suffix}.pkl"
    erased_dir = f"{base_dir}/Embeddings/Erased/{dataset_dir_name_probe}"
    results_dir = f"{base_dir}/Results/{dataset_dir_name_probe}"
    for d in [erased_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Running 'Regressing Out' Baseline (Scaling: {args.scaling.upper()}) ---")
    print(f"Dataset: {args.dataset}, Concept: {args.concept}, Data Fraction: {args.data_fraction}%")
    print(f"Loading embeddings from: {emb_file}")

    try:
        with open(emb_file, "rb") as f: data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {emb_file}")
        print("Please ensure you have generated the embeddings for this data fraction first.")
        return

    # --- Prepare Full Numpy Arrays ---
    X_full = np.vstack([s["embeddings_by_layer"][LAYER].numpy() for s in data])
    if args.concept in ["pos", "deplab"]:
        labels = sorted(list(set(l for s in data for l in s[internal_concept_key])))
        l2i = {l: i for i, l in enumerate(labels)}; indices = np.array([l2i[l] for s in data for l in s[internal_concept_key]])
        Z_full = np.zeros((len(indices), len(labels))); Z_full[np.arange(len(indices)), indices] = 1.0
        Z_full = Z_full[:, :-1]
    elif args.concept == 'ld':
        Z_full = np.array([val for s in data for val in s[internal_concept_key]]).reshape(-1, 1)
    else: Z_full = np.vstack([s[internal_concept_key].cpu().numpy() for s in data])
    
    # --- Split Data ---
    indices = np.arange(X_full.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
    X_train, Z_train = X_full[train_indices], Z_full[train_indices]
    X_test, Z_test = X_full[test_indices], Z_full[test_indices]
    
    # --- Prepare Data for Fitting ---
    if args.scaling == "on":
        scaler = StandardScaler(); X_train_to_fit = scaler.fit_transform(X_train)
    else:
        scaler = None; X_train_to_fit = X_train

    # --- Fit Regressor on Train Set ---
    fitter = OLSFitter(); fitter.fit(X_train_to_fit, Z_train)

    # --- "Erase" the TEST SET to calculate metrics ---
    if args.scaling == "on":
        X_test_to_erase = scaler.transform(X_test)
    else:
        X_test_to_erase = X_test
    X_test_erased = fitter.get_residuals(X_test_to_erase, Z_test)

    # --- "Erase" the FULL DATASET for saving ---
    if args.scaling == "on":
        X_full_scaled = scaler.transform(X_full)
        X_full_erased = fitter.get_residuals(X_full_scaled, Z_full)
    else:
        X_full_erased = fitter.get_residuals(X_full, Z_full)

    # =====================================================================
    # === ADDED: CALCULATE AND SAVE L2 SCORE AND RESULTS ===
    # =====================================================================

    # Calculate L2 distance between original test vectors and their residuals
    l2_distance_test = np.linalg.norm(X_test_to_erase - X_test_erased, axis=1).mean()
    print(f"\nL2 distance on test set: {l2_distance_test:.4f}")

    # --- Save the "erased" embeddings (residuals) ---
    erased_emb_path = f"{erased_dir}/regressout_{dataset_name}_{args.concept}{file_suffix}.pkl"
    print(f"Saving FULL erased (residual) embeddings array to: {erased_emb_path}")
    with open(erased_emb_path, "wb") as f: pickle.dump(X_full_erased, f)
        
    # --- Save Results Summary ---
    results_file = f"{results_dir}/regressout_{dataset_name}_{args.concept}{file_suffix}_results.json"
    results = {
        "method": "regressout", "dataset": args.dataset, "concept": args.concept, "scaling": args.scaling,
        "data_fraction_used": args.data_fraction,
        "l2_distance_on_test_set": float(l2_distance_test),
        "notes": f"L2 distance is between original test vectors and their computed residuals, on the {args.data_fraction}% data subset."
    }
    print(f"Saving results summary to: {results_file}")
    with open(results_file, "w") as f: json.dump(results, f, indent=4)
    
    print("\nDone.")

if __name__ == "__main__":
    main()