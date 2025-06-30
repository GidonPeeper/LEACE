"""
A specialized probing script to compare different distance/positional concepts:
- sd: Syntactic Distance (structural position via dependency tree)
- ld: Linear Distance (naive token index)
- pe: Positional Encoding (sinusoidal vectors from 'Attention Is All You Need')

This script evaluates how erasing one of these concepts affects the decodability
of the others, using the rigorous LEACE (test-set) and ORACLE (in-sample)
methodologies. All continuous concepts are evaluated using Cosine Similarity,
except for the scalar 'ld' concept which uses R2 score.
"""
import argparse, pickle, numpy as np, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.metrics import r2_score # IMPORT ADDED FOR R2 SCORE

# =========================================================================
# 1. Helper Functions
# =========================================================================

def cosine_similarity_score(y_true, y_pred):
    """Calculates the average cosine similarity for regression evaluation."""
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    dot = np.sum(y_true * y_pred, axis=1)
    norms = norm(y_true, axis=1) * norm(y_pred, axis=1)
    return np.mean(dot / (norms + 1e-9))

# === MODIFIED FUNCTION TO HANDLE 'ld' DIFFERENTLY ===
def train_probe_and_eval(X, y, concept_name, seed=42):
    """
    Trains and evaluates a linear Ridge regression probe on the provided data.
    Uses R2 score for 'ld' and Cosine Similarity for other vector concepts.
    """
    if y.ndim == 1: y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    
    probe = Ridge(random_state=seed).fit(X_train_s, y_train)
    preds = probe.predict(X_test_s)
    
    if concept_name == 'ld':
        # Use R2 score for the scalar 'ld' concept
        return r2_score(y_test, preds)
    else:
        # Use Cosine Similarity for vector concepts 'sd' and 'pe'
        return cosine_similarity_score(y_test, preds)

# =========================================================================
# 2. Main Script Logic
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run a distance/position probing analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to probe.")
    parser.add_argument("--method", choices=["leace", "oracle"], required=True, help="Erasure method to probe.")
    args = parser.parse_args()

    # --- Setup ---
    method = args.method
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
        # --- Part A: Get the ground-truth labels (Z) and baseline data (X_original) ---
        if method == "leace":
            probe_data_path = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}/leace_{dataset_name_short}_{probed_concept}_for_probing.pkl"
            try:
                with open(probe_data_path, "rb") as f: p_data = pickle.load(f)
                X_original, Z_labels = p_data["X_test_original"], p_data["Z_test"]
            except FileNotFoundError:
                print(f"  - WARNING: Probing data file for '{probed_concept}' not found at {probe_data_path}. Skipping row."); continue
        else: # ORACLE METHOD
            internal_key = concept_map[probed_concept]["internal_key"]
            file_key = concept_map[probed_concept]["file_key"]
            orig_path = f"{base_dir}/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{file_key}.pkl"
            try:
                with open(orig_path, "rb") as f: data = pickle.load(f)
                X_original = np.vstack([s["embeddings_by_layer"][8].numpy() for s in data])
                if probed_concept == 'ld':
                    Z_labels = np.array([val for s in data for val in s[internal_key]]).reshape(-1, 1)
                else:
                    Z_labels = np.vstack([s[internal_key].cpu().numpy() for s in data])
            except FileNotFoundError:
                print(f"  - WARNING: Original data for '{probed_concept}' not found at {orig_path}. Skipping row."); continue
        
        # === MODIFIED CALL: Pass concept name to evaluation function ===
        raw_scores.loc[probed_concept, "original"] = train_probe_and_eval(X_original, Z_labels, probed_concept)

        # --- Part B: Probe the erased embeddings ---
        for erased_concept in CONCEPTS:
            if method == "leace":
                erased_path = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}/leace_{dataset_name_short}_{erased_concept}_for_probing.pkl"
                try:
                    with open(erased_path, "rb") as f: e_data = pickle.load(f)
                    X_erased = e_data["X_test_erased"]
                except FileNotFoundError: continue
            else: # oracle
                erased_path = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}/oracle_{dataset_name_short}_{erased_concept}_vec.pkl"
                try:
                    with open(erased_path, "rb") as f: e_data = pickle.load(f)
                    X_erased = np.vstack(e_data['all_erased'])
                except FileNotFoundError: continue

            if X_erased.shape[0] == X_original.shape[0]:
                # === MODIFIED CALL: Pass concept name to evaluation function ===
                raw_scores.loc[probed_concept, erased_concept] = train_probe_and_eval(X_erased, Z_labels, probed_concept)
            else:
                print(f"  - WARNING: Mismatch in data points for erased concept '{erased_concept}'. Skipping cell.")

    # --- Calculate RDI and Display/Save Results ---
    rdi_scores = pd.DataFrame(index=CONCEPTS, columns=CONCEPTS, dtype=float)
    for p_concept in CONCEPTS:
        p_before = raw_scores.loc[p_concept, "original"]
        if pd.isna(p_before): continue
        for e_concept in CONCEPTS:
            p_after = raw_scores.loc[p_concept, e_concept]
            if pd.isna(p_after): continue
            # For Cosine Similarity and R2, the baseline/chance score is 0
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