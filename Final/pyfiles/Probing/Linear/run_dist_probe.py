"""
A specialized probing script to compare different distance/positional concepts.
This version is FULLY UPDATED to be compatible with the final erasure pipeline.
"""
import argparse
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.metrics import r2_score

# =========================================================================
# 1. Helper Functions (Unchanged - Correct)
# =========================================================================

def cosine_similarity_score(y_true, y_pred):
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    dot = np.sum(y_true * y_pred, axis=1)
    norms = norm(y_true, axis=1) * norm(y_pred, axis=1)
    return np.mean(dot / (norms + 1e-9))

def train_probe_and_eval(X, y, concept_name, seed=42):
    if y.ndim == 1: y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    
    probe = Ridge(random_state=seed).fit(X_train_s, y_train)
    preds = probe.predict(X_test_s)
    
    if concept_name == 'ld':
        return r2_score(y_test, preds)
    else:
        return cosine_similarity_score(y_test, preds)

# =========================================================================
# 2. Main Script Logic (Updated for New Pipeline)
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run a distance/position probing analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to probe.")
    parser.add_argument("--method", choices=["leace", "oracle"], required=True, help="Erasure method to probe.")
    args = parser.parse_args()

    # --- Setup ---
    method = args.method
    SEED = 42
    CONCEPTS = ["sd", "ld", "pe"]
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    base_dir = "Final"

    concept_map = {
        "sd": {"internal_key": "sd", "file_key": "sd"},
        "ld": {"internal_key": "ld", "file_key": "ld"},
        "pe": {"internal_key": "pe", "file_key": "pe"}
    }

    print(f"\n\n{'='*80}\nDISTANCE PROBING for Method: {method.upper()}, Dataset: {args.dataset.upper()}\n{'='*80}")
    raw_scores = pd.DataFrame(index=CONCEPTS, columns=["original"] + CONCEPTS, dtype=float)

    # --- Data Loading and Probing Loop ---
    for probed_concept in tqdm(CONCEPTS, desc="Probing Concept"):
        p_info = concept_map[probed_concept]
        
        # ======================================================================
        # === START: NEW UNIFIED DATA LOADING AND SPLITTING LOGIC ===
        # ======================================================================
        
        # 1. Load Original Data (X_full and Z_full) for the concept being probed.
        orig_path = f"{base_dir}/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{p_info['file_key']}.pkl"
        try:
            with open(orig_path, "rb") as f: data = pickle.load(f)
            X_full = np.vstack([s["embeddings_by_layer"][8].numpy() for s in data])
            
            # This logic handles ld (scalar) vs. sd/pe (vector) correctly.
            if probed_concept == 'ld':
                Z_full = np.array([val for s in data for val in s[p_info['internal_key']]]).reshape(-1, 1)
            else:
                Z_full = np.vstack([s[p_info['internal_key']].cpu().numpy() for s in data])
        except FileNotFoundError:
            print(f"  - WARNING: Original data for '{probed_concept}' not found at {orig_path}. Skipping row."); continue

        # 2. Determine which subset of data to use for probing based on the method.
        if method == "leace":
            # For LEACE, we must probe on the 20% test set.
            # Re-create the split using the same SEED used in the erasure script.
            indices = np.arange(X_full.shape[0])
            _, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
            X_probe_orig = X_full[test_indices]
            Z_probe_labels = Z_full[test_indices]
        else: # oracle
            # For Oracle, we probe on the full dataset.
            X_probe_orig = X_full
            Z_probe_labels = Z_full
            
        # 3. Probe the original (unerased) embeddings.
        raw_scores.loc[probed_concept, "original"] = train_probe_and_eval(X_probe_orig, Z_probe_labels, probed_concept, seed=SEED)

        # 4. Probe the erased embeddings for each concept.
        for erased_concept in CONCEPTS:
            # Use the new, unified file path.
            erased_path = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}/{method}_{dataset_name_short}_{erased_concept}.pkl"
            try:
                with open(erased_path, "rb") as f: e_data = pickle.load(f)
                
                # Check if the loaded data is a NumPy array, as expected.
                if isinstance(e_data, np.ndarray):
                    X_full_erased = e_data
                else:
                    print(f"\n  - FATAL ERROR: Unknown data format in '{erased_path}'. Expected np.ndarray, got {type(e_data)}.")
                    continue
                
                # Select the correct subset of the erased data based on the method.
                if method == "leace":
                    X_erased_probe = X_full_erased[test_indices]
                else: # oracle
                    X_erased_probe = X_full_erased

                if X_erased_probe.shape[0] == Z_probe_labels.shape[0]:
                    raw_scores.loc[probed_concept, erased_concept] = train_probe_and_eval(X_erased_probe, Z_probe_labels, probed_concept, seed=SEED)
                else:
                    print(f"  - WARNING: Mismatch for Z:'{probed_concept}' ({Z_probe_labels.shape[0]}) and X_erased:'{erased_concept}' ({X_erased_probe.shape[0]}). Skipping cell.")

            except FileNotFoundError:
                print(f"  - INFO: Erased file '{erased_path}' not found. Cell will be NaN.")
                continue
        # ======================================================================
        # === END: NEW UNIFIED DATA LOADING AND SPLITTING LOGIC ===
        # ======================================================================

    # --- RDI Calculation and Display/Save Results (Unchanged - Correct) ---
    rdi_scores = pd.DataFrame(index=CONCEPTS, columns=CONCEPTS, dtype=float)
    for p_concept in CONCEPTS:
        p_before = raw_scores.loc[p_concept, "original"]
        if pd.isna(p_before): continue
        for e_concept in CONCEPTS:
            p_after = raw_scores.loc[p_concept, e_concept]
            if pd.isna(p_after): continue
            denominator = p_before
            rdi = 1.0 - p_after / denominator if abs(denominator) > 1e-9 else 0.0
            rdi_scores.loc[p_concept, e_concept] = rdi

    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "-"*70 + f"\nRESULTS for Method: {method.upper()}\n" + "-" * 70)
    print("\n--- Raw Probe Performance (R2 for ld; Cosine Sim for sd/pe) ---")
    print("Columns: Concept Erased From Embeddings | Rows: Concept Being Probed\n")
    print(raw_scores)
    print("\n--- RDI (Normalized Relative Drop in Information) Scores ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(rdi_scores)
    print("\n" + "="*80)

    results_dir = f"Final/Results/{dataset_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    raw_scores.to_csv(f"{results_dir}/distance_probing_raw_perf_{args.dataset}_{method}.csv")
    rdi_scores.to_csv(f"{results_dir}/distance_probing_rdi_scores_{args.dataset}_{method}.csv")
    print(f"\nResults tables saved to CSV files in: {results_dir}")

if __name__ == "__main__":
    main()