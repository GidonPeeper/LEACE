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

def cosine_similarity_score(y_true, y_pred):
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    dot = np.sum(y_true * y_pred, axis=1)
    norms = norm(y_true, axis=1) * norm(y_pred, axis=1)
    return np.mean(dot / (norms + 1e-9))

def train_probe_and_eval(X, y, task_type, seed=42):
    if task_type == 'classification' and y.ndim == 2:
        y = np.argmax(y, axis=1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    if task_type == 'classification':
        y_train, y_test = y_train.ravel(), y_test.ravel()
        probe = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1).fit(X_train_s, y_train)
        return probe.score(X_test_s, y_test)
    else: # regression
        probe = Ridge(random_state=seed).fit(X_train_s, y_train)
        preds = probe.predict(X_test_s)
        return cosine_similarity_score(y_test, preds)

def main():
    parser = argparse.ArgumentParser(description="Run main probing analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--method", choices=["leace", "oracle"], required=True)
    args = parser.parse_args()

    # --- Setup ---
    method = args.method
    SEED = 42
    CONCEPTS = ["pos", "deplab", "sd"]
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    base_dir = "Final"
    
    concept_map = {
        "pos":    {"internal_key": "pos_tags", "file_key": "pos", "task": "classification"},
        "deplab": {"internal_key": "deplabs",  "file_key": "deplab", "task": "classification"},
        "sd":     {"internal_key": "sd",       "file_key": "sd", "task": "regression"}
    }

    print(f"\n\n{'='*80}\nPROBING ANALYSIS for Method: {method.upper()}, Dataset: {args.dataset.upper()}\n{'='*80}")
    raw_scores = pd.DataFrame(index=CONCEPTS, columns=["original"] + CONCEPTS, dtype=float)
    chance_scores = {}

    for probed_concept in tqdm(CONCEPTS, desc="Probing Concept"):
        p_info = concept_map[probed_concept]
        task_type = p_info["task"]
        
        # --- Unified Data Loading ---
        # 1. Load Original Data (X_full and Z_full)
        orig_path = f"{base_dir}/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{p_info['file_key']}.pkl"
        try:
            with open(orig_path, "rb") as f: data = pickle.load(f)
            X_full = np.vstack([s["embeddings_by_layer"][8].numpy() for s in data])
            
            if task_type == 'classification':
                all_str_labels = sorted(list(set(lab for s in data for lab in s[p_info['internal_key']])))
                label_to_idx = {label: i for i, label in enumerate(all_str_labels)}
                indices = [label_to_idx[lab] for s in data for lab in s[p_info['internal_key']]]
                Z_full = np.zeros((len(indices), len(all_str_labels)), dtype=np.float32)
                Z_full[np.arange(len(indices)), indices] = 1.0
            else: # regression (sd)
                Z_full = np.vstack([s[p_info['internal_key']].cpu().numpy() for s in data])
        except FileNotFoundError:
            print(f"  - WARNING: Original data for '{probed_concept}' not found at {orig_path}. Skipping."); continue

        # 2. Determine which subset of data to use for probing based on method
        if method == "leace":
            indices = np.arange(X_full.shape[0])
            _, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
            X_probe_orig = X_full[test_indices]
            Z_probe_labels = Z_full[test_indices]
        else: # oracle
            X_probe_orig = X_full
            Z_probe_labels = Z_full
        
        # --- Calculate Chance and Probe Original Embeddings ---
        if task_type == 'classification':
            labels_for_chance = np.argmax(Z_probe_labels, axis=1)
            counts = Counter(labels_for_chance); chance_scores[probed_concept] = counts.most_common(1)[0][1] / len(labels_for_chance)
        else: chance_scores[probed_concept] = 0.0

        raw_scores.loc[probed_concept, "original"] = train_probe_and_eval(X_probe_orig, Z_probe_labels, task_type, seed=SEED)

        # --- Probe the Erased Embeddings ---
        for erased_concept in CONCEPTS:
            erased_path = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}/{method}_{dataset_name_short}_{erased_concept}.pkl"
            try:
                with open(erased_path, "rb") as f: e_data = pickle.load(f)
                
                # --- SIMPLIFIED AND ROBUST LOADING LOGIC ---
                if isinstance(e_data, np.ndarray):
                    X_full_erased = e_data
                else:
                    print(f"\n  - FATAL ERROR: Unknown data format in '{erased_path}'. Expected np.ndarray, got {type(e_data)}.")
                    continue
                
                # Select the correct subset of the erased data based on the method
                if method == "leace":
                    X_erased_probe = X_full_erased[test_indices]
                else: # oracle
                    X_erased_probe = X_full_erased

                if X_erased_probe.shape[0] == Z_probe_labels.shape[0]:
                    raw_scores.loc[probed_concept, erased_concept] = train_probe_and_eval(X_erased_probe, Z_probe_labels, task_type, seed=SEED)
                else:
                    print(f"  - WARNING: Mismatch for Z:'{probed_concept}' ({Z_probe_labels.shape[0]}) and X_erased:'{erased_concept}' ({X_erased_probe.shape[0]}). Skipping cell.")

            except FileNotFoundError:
                print(f"  - INFO: Erased file '{erased_path}' not found. Cell will be NaN.")
                continue

    # --- RDI Calculation and Display/Save (Unchanged) ---
    rdi_scores = pd.DataFrame(index=CONCEPTS, columns=CONCEPTS, dtype=float)
    for p_concept in CONCEPTS:
        p_before = raw_scores.loc[p_concept, "original"]
        if pd.isna(p_before): continue
        for e_concept in CONCEPTS:
            p_after = raw_scores.loc[p_concept, e_concept]
            if pd.isna(p_after): continue
            chance = chance_scores.get(p_concept, 0)
            denominator = p_before - chance
            rdi = 1.0 - (p_after - chance) / denominator if denominator > 1e-9 else 0.0
            rdi_scores.loc[p_concept, e_concept] = rdi
    
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "-"*70 + f"\nRESULTS for Method: {method.upper()}\n" + "-"*70)
    print("\n--- Raw Probe Performance (Accuracy for pos/deplab; Cosine Sim for sd) ---")
    print("Columns: Concept Erased From Embeddings | Rows: Concept Being Probed\n")
    print(raw_scores)
    print("\n--- RDI (Normalized Relative Drop in Information) Scores ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(rdi_scores)
    print("\n" + "="*80)
    results_dir = f"Final/Results/{dataset_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    raw_scores.to_csv(f"{results_dir}/main_probing_raw_perf_{args.dataset}_{method}.csv")
    rdi_scores.to_csv(f"{results_dir}/main_probing_rdi_scores_{args.dataset}_{method}.csv")
    print(f"\nResults tables saved to CSV files in: {results_dir}")

if __name__ == "__main__":
    main()