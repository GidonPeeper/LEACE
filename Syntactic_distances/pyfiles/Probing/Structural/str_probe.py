"""
Runs an advanced probing analysis using the TwoWordPSDProbe from the
Hewitt & Manning `structural-probes` repository.

This script automates the process of evaluating how well syntactic distance is
encoded in various original and erased embeddings. It is built by wrapping the
exact logic from a proven, working script into a reusable function, ensuring
maximum reliability.

**Prerequisites:**
1.  The `structural-probes` submodule must be patched to fix the PyYAML `Loader` error.
    - In `.../run_experiment.py`, change `yaml.load(...)` to `yaml.load(..., Loader=yaml.FullLoader)`.
2.  The `seaborn` package must be installed in the conda environment.
    - `conda install seaborn` or `pip install seaborn`.
"""
import argparse
import pickle
import numpy as np
import torch
import os
import sys
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Ensure the structural_probes submodule is in the Python path
# This allows `from external...` to work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from external.structural_probes.structural_probe.probe import TwoWordPSDProbe

# =========================================================================
# 1. Core Probing Function (Adapted from the user-provided working script)
# =========================================================================

def run_single_probe_experiment(
    train_emb_path: str,
    test_emb_path: str,
    gold_dist_train_path: str,
    gold_dist_test_path: str,
    layer: int = 8,
    device: str = "cuda"
) -> float:
    """
    Runs a complete train/test cycle for a single structural probe experiment,
    using the exact logic from the user-provided working script to ensure reliability.
    """
    # --- Data Loading Logic ---
    def load_embeddings(file_path, layer):
        with open(file_path, "rb") as f: data = pickle.load(f)
        # Your script handles two cases for erased embeddings from different dev stages
        if isinstance(data, dict) and ("train_erased" in data or "test_erased" in data):
            return data["train_erased"] if "train" in file_path else data["test_erased"]
        elif isinstance(data, dict) and "all_erased" in data:
            # Assumes a single file for both train and test that needs splitting
            return data["all_erased"] 
        else: # Original embeddings format
            return [sent["embeddings_by_layer"][layer] for sent in data]

    def load_distances(file_path):
        with open(file_path, "rb") as f: data = pickle.load(f)
        return [sent["distance_matrix"] for sent in data]

    X_train_sents = load_embeddings(train_emb_path, layer)
    X_test_sents = load_embeddings(test_emb_path, layer)
    D_train = load_distances(gold_dist_train_path)
    D_test = load_distances(gold_dist_test_path)

    # --- Max Length and Padding ---
    max_len = 0
    for sents_list in [X_train_sents, X_test_sents]:
        for sent in sents_list: max_len = max(max_len, sent.shape[0])
    pad_value = max_len - 1

    def pad_and_clean(X_sents, D_sents, max_len, pad_val):
        D_padded, X_clean = [], []
        for i, d in enumerate(D_sents):
            if np.isfinite(d).all() and d.shape[0] == X_sents[i].shape[0]:
                n = d.shape[0]
                padded = np.full((max_len, max_len), pad_val, dtype=np.float32)
                padded[:n, :n] = d
                D_padded.append(padded)
                X_clean.append(X_sents[i])
        return X_clean, D_padded

    X_train_sents, D_train_padded = pad_and_clean(X_train_sents, D_train, max_len, pad_value)
    X_test_sents, D_test_padded = pad_and_clean(X_test_sents, D_test, max_len, pad_value)

    if not X_train_sents or not X_test_sents:
        print("Warning: No valid data remains after cleaning/matching. Returning NaN.")
        return np.nan

    # --- Standardization ---
    X_train_flat = np.vstack([x.numpy() if isinstance(x, torch.Tensor) else x for x in X_train_sents])
    scaler = StandardScaler().fit(X_train_flat)
    
    def apply_scaler(X_sents, scaler, device):
        X_flat = np.vstack([x.numpy() if isinstance(x, torch.Tensor) else x for x in X_sents])
        X_flat_scaled = scaler.transform(X_flat)
        X_scaled_sents, idx = [], 0
        for x in X_sents:
            n = x.shape[0]
            X_scaled_sents.append(torch.tensor(X_flat_scaled[idx:idx+n], dtype=torch.float32, device=device))
            idx += n
        return X_scaled_sents

    X_train_scaled_sents = apply_scaler(X_train_sents, scaler, device)
    X_test_scaled_sents = apply_scaler(X_test_sents, scaler, device)

    D_train_torch = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_train_padded]
    D_test_torch = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_test_padded]

    # --- Probe Training ---
    probe_rank = min(128, X_train_scaled_sents[0].shape[1])
    probe = TwoWordPSDProbe({'probe': {'maximum_rank': probe_rank}, 'model': {'hidden_dim': X_train_scaled_sents[0].shape[1]}}, device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-4)

    def probe_loss(pred, gold, pad_val):
        mask = (gold != pad_val)
        return ((pred - gold)[mask] ** 2).mean()

    for _ in range(10): # A fixed number of epochs is fine for this task
        probe.train()
        for x, d in zip(X_train_scaled_sents, D_train_torch):
            pred = probe(x.unsqueeze(0)).squeeze(0)
            loss = probe_loss(pred, d, pad_value)
            if not torch.isfinite(loss): continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # --- Evaluation ---
    probe.eval()
    all_preds, all_golds = [], []
    for x, d in zip(X_test_scaled_sents, D_test_torch):
        n = x.shape[0]
        with torch.no_grad():
            pred = probe(x.unsqueeze(0)).squeeze(0)[:n, :n].cpu().numpy()
        gold = d[:n, :n].cpu().numpy()
        all_preds.append(pred)
        all_golds.append(gold)
        
    all_preds_flat = np.concatenate([p.flatten() for p in all_preds])
    all_golds_flat = np.concatenate([g.flatten() for g in all_golds])

    return r2_score(all_golds_flat, all_preds_flat)


# =========================================================================
# 2. Main Orchestrator
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run an advanced structural probe analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to probe.")
    parser.add_argument("--method", choices=["leace", "oracle"], required=True, help="Erasure method to probe.")
    args = parser.parse_args()

    # --- Setup ---
    CONCEPTS = ["pos", "deplab", "sd", "sdm"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Advanced Probe Analysis ---")
    print(f"Dataset: {args.dataset}, Erasure Method: {args.method}, Device: {DEVICE}")

    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    raw_scores = pd.DataFrame(index=['structural_probe'], columns=CONCEPTS + ['original'], dtype=float)
    rdi_scores = pd.DataFrame(index=['structural_probe'], columns=CONCEPTS, dtype=float)

    # --- Define Gold Standard Data Paths ---
    # These must point to files that contain both embeddings and distance matrices
    # Let's assume your data generation pipeline creates combined train/test files now
    base_path = f"Final/Embeddings/Original/{dataset_dir_name}/"
    gold_dist_train_path = f"{base_path}Embed_{dataset_name_short}_distance_matrix_train.pkl" # Adjust if your names differ
    gold_dist_test_path = f"{base_path}Embed_{dataset_name_short}_distance_matrix_test.pkl"   # Adjust if your names differ

    # --- Baseline Calculation ---
    print("\n[Phase 1/3] Calculating baseline on original embeddings...")
    baseline_emb_train_path = gold_dist_train_path # Original embeddings are in the same file
    baseline_emb_test_path = gold_dist_test_path
    
    baseline_score = run_single_probe_experiment(
        train_emb_path=baseline_emb_train_path, test_emb_path=baseline_emb_test_path,
        gold_dist_train_path=gold_dist_train_path, gold_dist_test_path=gold_dist_test_path,
        device=DEVICE
    )
    raw_scores.loc['structural_probe', 'original'] = baseline_score
    print(f"Baseline R² on original embeddings: {baseline_score:.4f}")

    # --- Probing Erased Embeddings ---
    print("\n[Phase 2/3] Probing erased embeddings...")
    for erased_concept in tqdm(CONCEPTS, desc="Probing erased"):
        # This assumes your erasure script also saves separate train/test files.
        erased_base_path = f"Final/Embeddings/Erased/{dataset_dir_name}/"
        erased_train_path = f"{erased_base_path}{args.method}_{dataset_name_short}_{erased_concept}_train_vec.pkl" # Adjust name
        erased_test_path = f"{erased_base_path}{args.method}_{dataset_name_short}_{erased_concept}_test_vec.pkl"   # Adjust name

        if not (os.path.exists(erased_train_path) and os.path.exists(erased_test_path)):
            print(f"\nWarning: Erased files not found for '{erased_concept}', skipping.")
            raw_scores.loc['structural_probe', erased_concept] = np.nan
            continue
            
        score = run_single_probe_experiment(
            train_emb_path=erased_train_path, test_emb_path=erased_test_path,
            gold_dist_train_path=gold_dist_train_path, gold_dist_test_path=gold_dist_test_path,
            device=DEVICE
        )
        raw_scores.loc['structural_probe', erased_concept] = score

    # --- RDI Calculation and Reporting ---
    print("\n[Phase 3/3] Calculating RDI scores and reporting...")
    for erased_concept in CONCEPTS:
        r2_after = raw_scores.loc['structural_probe', erased_concept]
        r2_before = baseline_score
        
        if pd.isna(r2_after) or pd.isna(r2_before) or r2_before < 1e-6:
            rdi = np.nan
        else:
            rdi = max(0, 1.0 - (r2_after / r2_before))
        rdi_scores.loc['structural_probe', erased_concept] = rdi

    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"ADVANCED STRUCTURAL PROBE RESULTS for Dataset: {args.dataset.upper()}, Method: {args.method.upper()}")
    print("="*80)
    
    print("\n--- Raw Probe Performance (R-squared for Syntactic Distance) ---")
    print("Columns: Concept Erased From Embeddings (+ Original Baseline)\n")
    print(raw_scores)
    
    print("\n\n--- RDI Scores from Structural Probe ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(rdi_scores)
    print("\n" + "="*80)
    
    results_dir = f"Final/Results/{dataset_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    raw_scores.to_csv(f"{results_dir}/advanced_probe_raw_perf_{args.dataset}_{args.method}.csv")
    rdi_scores.to_csv(f"{results_dir}/advanced_probe_rdi_scores_{args.dataset}_{args.method}.csv")
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()