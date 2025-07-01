"""
A unified probing script to analyze the erasure of five core concepts:
- pos (Part-of-Speech):       Classification, evaluated with Accuracy.
- deplab (Dependency Relation): Classification, evaluated with Accuracy.
- ld (Linear Distance):        Regression, evaluated with R2 Score.
- sd (Syntactic Distance):     Regression, evaluated with R2 Score.
- pe (Positional Encoding):    Regression, evaluated with Cosine Similarity.

This script reads from erasure pipelines that can be run with or without
a StandardScaler preprocessing step, controlled by the --scaling flag.
"""
import argparse
import pickle
import numpy as np
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.metrics import r2_score

# =========================================================================
# 1. Unified Evaluation Functions
# =========================================================================

# === THIS FUNCTION WAS MISSING. IT IS NOW ADDED BACK. ===
def cosine_similarity_score(y_true, y_pred):
    """Calculates the average cosine similarity for regression evaluation."""
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    dot = np.sum(y_true * y_pred, axis=1)
    norms = norm(y_true, axis=1) * norm(y_pred, axis=1)
    # Add a small epsilon to avoid division by zero
    return np.mean(dot / (norms + 1e-9))

def train_probe_and_eval(X, y, task_type, seed=42):
    """
    Trains and evaluates a probe based on the specified task type.
    - 'classification': Uses Logistic Regression and returns Accuracy.
    - 'regression_r2':  Uses Ridge Regression and returns R2 Score.
    - 'regression_cos': Uses Ridge Regression and returns Cosine Similarity.
    """
    # Create the probe's own internal train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    # Scale the features for the probe
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    
    if task_type == 'classification':
        # Convert one-hot labels to class indices for the probe
        if y.ndim == 2:
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)
        
        probe = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1).fit(X_train_s, y_train.ravel())
        return probe.score(X_test_s, y_test.ravel())
    
    else: # All regression tasks
        probe = Ridge(random_state=seed).fit(X_train_s, y_train)
        preds = probe.predict(X_test_s)
        
        if task_type == 'regression_r2':
            return r2_score(y_test, preds)
        elif task_type == 'regression_cos':
            return cosine_similarity_score(y_test, preds)
        else:
            raise ValueError(f"Unknown regression task type: {task_type}")

# =========================================================================
# 2. Main Script Logic (Unchanged from what I provided before)
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run a unified probing analysis for all concepts.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--method", choices=["leace", "oracle"], required=True)
    parser.add_argument("--scaling", choices=["on", "off"], default="on", help="Probe the scaled ('on') or unscaled ('off') erasure results.")
    args = parser.parse_args()

    # --- Setup ---
    method, SEED = args.method, 42
    CONCEPTS = ["pos", "deplab", "ld", "sd", "pe"]
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    # The master map defining how to handle each concept
    concept_map = {
        "pos":    {"ikey": "pos_tags", "fkey": "pos",    "task": "classification"},
        "deplab": {"ikey": "deplabs",  "fkey": "deplab", "task": "classification"},
        "ld":     {"ikey": "ld",       "fkey": "ld",     "task": "regression_r2"},
        "sd":     {"ikey": "sd",       "fkey": "sd",     "task": "regression_r2"},
        "pe":     {"ikey": "pe",       "fkey": "pe",     "task": "regression_cos"}
    }
    
    # Dynamic paths based on the scaling flag
    scaling_suffix = "_Scaled" if args.scaling == "on" else "_Unscaled"
    dataset_dir_name_base = "UD" if args.dataset == "ud" else "Narratives"
    dataset_dir_name_probe = dataset_dir_name_base + scaling_suffix
    base_dir = "Final"

    print(f"\n\n{'='*80}\nUNIFIED PROBE (Method: {method.upper()}, Dataset: {args.dataset.upper()}, Scaling: {args.scaling.upper()})\n{'='*80}")
    raw_scores = pd.DataFrame(index=CONCEPTS, columns=["original"] + CONCEPTS, dtype=float)
    chance_scores = {}

    for p_concept in tqdm(CONCEPTS, desc="Probing Concept"):
        p_info = concept_map[p_concept]
        
        # --- Load Original Data and Ground-Truth Labels (Z_full) ---
        orig_path = f"{base_dir}/Embeddings/Original/{dataset_dir_name_base}/Embed_{dataset_name_short}_{p_info['fkey']}.pkl"
        try:
            with open(orig_path, "rb") as f: data = pickle.load(f)
            X_full = np.vstack([s["embeddings_by_layer"][8].numpy() for s in data])
            
            # Prepare Z_full based on concept type
            if p_info["task"] == 'classification':
                all_str_labels = sorted(list(set(lab for s in data for lab in s[p_info['ikey']])))
                label_to_idx = {label: i for i, label in enumerate(all_str_labels)}
                indices = [label_to_idx[lab] for s in data for lab in s[p_info['ikey']]]
                Z_full = np.zeros((len(indices), len(all_str_labels)), dtype=np.float32)
                Z_full[np.arange(len(indices)), indices] = 1.0
            elif p_concept == 'ld':
                Z_full = np.array([val for s in data for val in s[p_info['ikey']]]).reshape(-1, 1)
            else: # sd, pe
                Z_full = np.vstack([s[p_info['ikey']].cpu().numpy() for s in data])
        except FileNotFoundError:
            print(f"  - WARNING: Original data for '{p_concept}' not found at {orig_path}. Skipping."); continue

        # --- Determine Probing Subset (LEACE vs. Oracle) ---
        if method == "leace":
            indices = np.arange(X_full.shape[0])
            _, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
            X_probe_orig = X_full[test_indices]
            Z_probe_labels = Z_full[test_indices]
        else: # oracle
            X_probe_orig = X_full
            Z_probe_labels = Z_full
        
        # --- Calculate Chance Scores and Probe Original Embeddings ---
        if p_info["task"] == 'classification':
            labels_for_chance = np.argmax(Z_probe_labels, axis=1)
            counts = Counter(labels_for_chance); chance_scores[p_concept] = counts.most_common(1)[0][1] / len(labels_for_chance)
        else: # For R2 and Cosine Sim, meaningful chance is 0
            chance_scores[p_concept] = 0.0
        
        raw_scores.loc[p_concept, "original"] = train_probe_and_eval(X_probe_orig, Z_probe_labels, p_info["task"], seed=SEED)

        # --- Probe the Erased Embeddings ---
        for e_concept in CONCEPTS:
            erased_path = f"{base_dir}/Embeddings/Erased/{dataset_dir_name_probe}/{method}_{dataset_name_short}_{e_concept}.pkl"
            try:
                with open(erased_path, "rb") as f: X_full_erased = pickle.load(f)
                
                if not isinstance(X_full_erased, np.ndarray):
                    print(f"\n  - FATAL ERROR: Unknown data format in '{erased_path}'. Expected np.ndarray.")
                    continue
                
                X_erased_probe = X_full_erased[test_indices] if method == "leace" else X_full_erased

                if X_erased_probe.shape[0] == Z_probe_labels.shape[0]:
                    raw_scores.loc[p_concept, e_concept] = train_probe_and_eval(X_erased_probe, Z_probe_labels, p_info["task"], seed=SEED)
            except FileNotFoundError:
                continue

    # --- RDI Calculation ---
    rdi_scores = pd.DataFrame(index=CONCEPTS, columns=CONCEPTS, dtype=float)
    for p_concept in CONCEPTS:
        p_before = raw_scores.loc[p_concept, "original"]
        chance = chance_scores.get(p_concept, 0.0)
        denominator = p_before - chance
        if pd.isna(p_before) or abs(denominator) < 1e-9: continue
        
        for e_concept in CONCEPTS:
            p_after = raw_scores.loc[p_concept, e_concept]
            if pd.isna(p_after): continue
            rdi = 1.0 - ((p_after - chance) / denominator)
            rdi_scores.loc[p_concept, e_concept] = rdi
    
    # --- Display and Save Results ---
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "-"*70 + f"\nRESULTS for Method: {method.upper()}, Dataset: {args.dataset.upper()}, Scaling: {args.scaling.upper()}\n" + "-"*70)
    print("\n--- Raw Probe Performance (Acc/R2/CosSim) ---")
    print("Columns: Concept Erased From Embeddings | Rows: Concept Being Probed\n")
    print(raw_scores)
    print("\n--- RDI (Normalized Relative Drop in Information) Scores ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(rdi_scores)
    print("\n" + "="*80)

    results_dir = f"Final/Results/{dataset_dir_name_probe}"
    scaling_filename = "scaled" if args.scaling == "on" else "unscaled"
    raw_scores.to_csv(f"{results_dir}/unified_raw_{args.dataset}_{method}_{scaling_filename}.csv")
    rdi_scores.to_csv(f"{results_dir}/unified_rdi_{args.dataset}_{method}_{scaling_filename}.csv")
    print(f"\nResults tables saved to CSV files in: {results_dir}")

if __name__ == "__main__":
    main()