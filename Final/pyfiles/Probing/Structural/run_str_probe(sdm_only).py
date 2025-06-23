"""
Runs an advanced probing analysis using a direct, in-memory implementation
of the TwoWordPSDProbe.

This script automates the process of evaluating how well syntactic distance is
encoded in various original and erased embeddings. It is built by wrapping the
exact logic from a proven, working script into a reusable function to ensure
maximum reliability and consistency.

It loads single data files and performs its own consistent train/test split.
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure the structural_probes submodule is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from external.structural_probes.structural_probe.probe import TwoWordPSDProbe

# =========================================================================
# 1. Core Probing Function (Adapted to perform its own train/test split)
# =========================================================================

def run_probe_on_files(
    embeddings_path: str,
    gold_distances_path: str,
    layer: int = 8,
    device: str = "cuda",
    seed: int = 42
) -> float:
    """
    Runs a complete train/test cycle for a single structural probe experiment.
    This function loads full datasets and performs a consistent train/test split internally.
    """
    
    # --- Data Loading Logic ---
    def load_embeddings(file_path, layer):
        """Loads only the embeddings (X data) from a given file."""
        with open(file_path, 'rb') as f: data = pickle.load(f)
        if isinstance(data, dict) and "all_erased" in data:
            return data["all_erased"]
        else:
            return [s["embeddings_by_layer"][layer] for s in data]

    def load_distances(file_path):
        """Loads only the gold-standard distance matrices (D data)."""
        with open(file_path, 'rb') as f: data = pickle.load(f)
        return [s["distance_matrix"] for s in data]

    # Load the full datasets
    all_X_sents = load_embeddings(embeddings_path, layer)
    all_D_sents = load_distances(gold_distances_path)

    # --- Data Cleaning and Matching ---
    X_clean, D_clean = [], []
    num_samples = min(len(all_X_sents), len(all_D_sents))
    for i in range(num_samples):
        d = all_D_sents[i]
        x = all_X_sents[i]
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        if np.isfinite(d).all() and d.shape[0] == x_np.shape[0] and x_np.shape[0] > 0:
            X_clean.append(x)
            D_clean.append(d)

    if not X_clean:
        print("Warning: No valid data remains after cleaning. Returning NaN.")
        return np.nan

    # --- Perform Consistent Train/Test Split ---
    indices = range(len(X_clean))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=seed)
    
    X_train_sents = [X_clean[i] for i in train_indices]
    D_train = [D_clean[i] for i in train_indices]
    X_test_sents = [X_clean[i] for i in test_indices]
    D_test = [D_clean[i] for i in test_indices]
    
    if not X_train_sents or not X_test_sents:
        print("Warning: Train or test split is empty. Returning NaN.")
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

    D_train_torch = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_train]
    D_test_torch = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_test]

    # --- Probe Training ---
    probe_rank = min(128, X_train_scaled_sents[0].shape[1])
    probe_args = {
        'probe': {'maximum_rank': probe_rank},
        'model': {'hidden_dim': X_train_scaled_sents[0].shape[1]},
        'device': device 
    }
    probe = TwoWordPSDProbe(probe_args)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, amsgrad=True)

    def probe_loss(pred, gold): return ((pred - gold) ** 2).mean()

    for _ in range(15):
        probe.train()
        for i in range(len(X_train_scaled_sents)):
            x, d = X_train_scaled_sents[i], D_train_torch[i]
            pred = probe(x.unsqueeze(0)).squeeze(0)
            loss = probe_loss(pred[:d.shape[0], :d.shape[0]], d)
            if not torch.isfinite(loss): continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # --- Evaluation ---
    probe.eval()
    all_preds_flat, all_golds_flat = [], []
    for i in range(len(X_test_scaled_sents)):
        x, d = X_test_scaled_sents[i], D_test_torch[i]
        with torch.no_grad():
            pred = probe(x.unsqueeze(0)).squeeze(0)[:d.shape[0], :d.shape[0]]
        all_preds_flat.append(pred.cpu().numpy().flatten())
        all_golds_flat.append(d.cpu().numpy().flatten())
        
    return r2_score(np.concatenate(all_golds_flat), np.concatenate(all_preds_flat))


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
    SEED = 42 # For reproducible train/test splits
    print(f"--- Running Advanced Probe Analysis ---")
    print(f"Dataset: {args.dataset}, Erasure Method: {args.method}, Device: {DEVICE}")

    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    raw_scores = pd.DataFrame(index=['structural_probe'], columns=CONCEPTS + ['original'], dtype=float)
    rdi_scores = pd.DataFrame(index=['structural_probe'], columns=CONCEPTS, dtype=float)

    # --- Define Gold Standard and Baseline File Paths ---
    # These paths point to your single, unsplit data files.
    orig_base = f"Final/Embeddings/Original/{dataset_dir_name}"
    erased_base = f"Final/Embeddings/Erased/{dataset_dir_name}"

    # The file containing the gold-standard distances
    gold_distances_path = f"{orig_base}/Embed_{dataset_name_short}_distance_matrix.pkl"
    # The file containing the original embeddings for the baseline run
    baseline_embeddings_path = f"{orig_base}/Embed_{dataset_name_short}_distance_matrix.pkl"
    
    # --- Baseline Calculation ---
    print("\n[Phase 1/3] Calculating baseline on original embeddings...")
    baseline_score = run_probe_on_files(
        embeddings_path=baseline_embeddings_path,
        gold_distances_path=gold_distances_path,
        device=DEVICE,
        seed=SEED
    )
    raw_scores.loc['structural_probe', 'original'] = baseline_score
    print(f"Baseline R² on original embeddings: {baseline_score:.4f}")

    # --- Probing Erased Embeddings ---
    print("\n[Phase 2/3] Probing erased embeddings...")
    for erased_concept in tqdm(CONCEPTS, desc="Probing erased"):
        # Assumes your erased files are named like 'leace_nar_pos_vec.pkl'
        erased_embeddings_path = f"{erased_base}/{args.method}_{dataset_name_short}_{erased_concept}_vec.pkl"

        if not os.path.exists(erased_embeddings_path):
            print(f"\nWarning: Erased file not found, skipping: {erased_embeddings_path}")
            raw_scores.loc['structural_probe', erased_concept] = np.nan
            continue
            
        score = run_probe_on_files(
            embeddings_path=erased_embeddings_path,
            gold_distances_path=gold_distances_path,
            device=DEVICE,
            seed=SEED
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