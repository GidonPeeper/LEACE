"""
================================================================================
            DEDICATED ADVANCED SPECTRAL PROBING SCRIPT
================================================================================
This script runs a focused probing analysis using the TwoWordPSDProbe to
evaluate how well the geometry of SPECTRAL VECTORS is encoded in various
original and erased embeddings.
"""
import argparse
import pickle
import numpy as np
import torch
import os
import sys
import pandas as pd
import random
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from external.structural_probes.structural_probe.probe import TwoWordPSDProbe

# =========================================================================
# 1. Core Probing Function and Helpers
# =========================================================================

def load_embeddings(file_path, layer=8):
    try:
        with open(file_path, 'rb') as f: data = pickle.load(f)
        if isinstance(data, dict) and "all_erased" in data:
            return [x.numpy() if isinstance(x, torch.Tensor) else x for x in data["all_erased"]]
        else:
            return [s["embeddings_by_layer"][layer].numpy() for s in data]
    except FileNotFoundError: return None

def load_spectral_vectors(file_path):
    with open(file_path, 'rb') as f: data = pickle.load(f)
    return [s["distance_spectral_padded"].cpu().numpy() for s in data]

def vectors_to_distance_matrix(vectors):
    sum_sq = np.sum(vectors**2, axis=1)
    dot_prod = np.dot(vectors, vectors.T)
    dist_sq = sum_sq[:, np.newaxis] - 2 * dot_prod + sum_sq[np.newaxis, :]
    return np.maximum(dist_sq, 0)

def run_spectral_probe(X_sents, D_sents, device="cuda", seed=42):
    # This function is now correct and does not need to change.
    indices = range(len(X_sents))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=seed)
    X_train, D_train = [X_sents[i] for i in train_indices], [D_sents[i] for i in train_indices]
    X_test, D_test = [X_sents[i] for i in test_indices], [D_sents[i] for i in test_indices]

    if not X_train or not X_test: return np.nan
    
    scaler = StandardScaler().fit(np.vstack(X_train))
    def apply_scaler(sents, scaler, dev):
        if not sents: return []
        flat = np.vstack(sents); scaled = scaler.transform(flat)
        res, idx = [], 0
        for x in sents:
            n = x.shape[0]; res.append(torch.tensor(scaled[idx:idx+n], dtype=torch.float32, device=dev)); idx += n
        return res

    X_train_s, X_test_s = apply_scaler(X_train, scaler, device), apply_scaler(X_test, scaler, device)
    D_train_t = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_train]
    D_test_t = [torch.tensor(d, dtype=torch.float32, device=device) for d in D_test]

    probe_args = {'probe':{'maximum_rank':128}, 'model':{'hidden_dim':X_train_s[0].shape[1]}, 'device':device}
    probe = TwoWordPSDProbe(probe_args)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, amsgrad=True)
    
    for _ in range(15):
        probe.train()
        for i in range(len(X_train_s)):
            pred = probe(X_train_s[i].unsqueeze(0)).squeeze(0)
            loss = ((pred[:D_train_t[i].shape[0],:D_train_t[i].shape[0]] - D_train_t[i])**2).mean()
            if torch.isfinite(loss): optimizer.zero_grad(); loss.backward(); optimizer.step()
            
    probe.eval()
    preds_flat, golds_flat = [], []
    with torch.no_grad():
        for i in range(len(X_test_s)):
            pred = probe(X_test_s[i].unsqueeze(0)).squeeze(0)
            preds_flat.append(pred[:D_test_t[i].shape[0],:D_test_t[i].shape[0]].cpu().numpy().flatten())
            golds_flat.append(D_test_t[i].cpu().numpy().flatten())
            
    return r2_score(np.concatenate(golds_flat), np.concatenate(preds_flat))

# =========================================================================
# 2. Main Orchestrator
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run an advanced probe for spectral geometry.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--method", choices=["leace", "oracle"], required=True)
    parser.add_argument("--debug_subset_fraction", type=float, default=1.0)
    args = parser.parse_args()

    # --- Setup ---
    ERASED_CONCEPTS = ["pos", "deplab", "sd", "sdm"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    print(f"--- Running Advanced SPECTRAL Probe Analysis ---")
    print(f"Dataset: {args.dataset}, Method: {args.method}, Device: {DEVICE}")
    if args.debug_subset_fraction < 1.0:
        print(f"!!! DEBUG MODE ON: Using {args.debug_subset_fraction:.0%} of the data. !!!")

    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    raw_scores = pd.DataFrame(index=['spectral_dist'], columns=ERASED_CONCEPTS + ['original'], dtype=float)
    rdi_scores = pd.DataFrame(index=['spectral_dist'], columns=ERASED_CONCEPTS, dtype=float)

    # --- Step 1: Create a Unified, Aligned Dataset ---
    print("\n[Step 1/3] Loading and creating unified dataset...")
    orig_base = f"Final/Embeddings/Original/{dataset_dir_name}"
    erased_base = f"Final/Embeddings/Erased/{dataset_dir_name}"

    # Load all necessary data sources
    X_orig = load_embeddings(f"{orig_base}/Embed_{dataset_name_short}_distance_spectral_padded.pkl")
    Z_spectral_vecs = load_spectral_vectors(f"{orig_base}/Embed_{dataset_name_short}_distance_spectral_padded.pkl")
    erased_embeddings = {c: load_embeddings(f"{erased_base}/{args.method}_{dataset_name_short}_{c}_vec.pkl") for c in ERASED_CONCEPTS}

    # Create one unified list of dictionaries
    unified_data = []
    for i in range(len(X_orig)):
        # Check if this sentence exists in all loaded erased files
        is_valid = all(erased_embeddings.get(c) and len(erased_embeddings[c]) > i for c in ERASED_CONCEPTS)
        if not is_valid: continue

        # Check for length consistency across all loaded data for this sentence
        base_len = X_orig[i].shape[0]
        is_consistent = all(erased_embeddings[c][i].shape[0] == base_len for c in ERASED_CONCEPTS) and Z_spectral_vecs[i].shape[0] == base_len
        if not is_consistent: continue
        
        # If all checks pass, add it to our unified dataset
        data_point = {'X_original': X_orig[i], 'Z_spectral_vectors': Z_spectral_vecs[i]}
        for c in ERASED_CONCEPTS:
            data_point[f'X_erased_{c}'] = erased_embeddings[c][i]
        unified_data.append(data_point)

    print(f"Created a unified dataset with {len(unified_data)} fully consistent sentences.")

    # --- Step 2: Subsetting and Final Preparation ---
    if args.debug_subset_fraction < 1.0:
        num_to_keep = int(len(unified_data) * args.debug_subset_fraction)
        random.Random(SEED).shuffle(unified_data)
        unified_data = unified_data[:num_to_keep]
        print(f"--- Subset size: {len(unified_data)} sentences. ---")

    # Convert gold spectral vectors to gold distance matrices ONCE
    for dp in unified_data:
        dp['D_gold_spectral'] = vectors_to_distance_matrix(dp['Z_spectral_vectors'])

    # --- Step 3: Probing and Reporting ---
    print("\n[Step 2/3] Running probes...")
    
    # Extract data for probing from the unified list
    X_original_sents = [dp['X_original'] for dp in unified_data]
    D_gold_sents = [dp['D_gold_spectral'] for dp in unified_data]

    # Baseline Calculation
    print("  Calculating baseline on original embeddings...")
    baseline_score = run_spectral_probe(X_original_sents, D_gold_sents, device=DEVICE, seed=SEED)
    raw_scores.loc['spectral_dist', 'original'] = baseline_score
    print(f"  > Baseline R² on original embeddings: {baseline_score:.4f}")

    # Probing Erased Embeddings
    for erased_concept in tqdm(ERASED_CONCEPTS, desc="Probing erased embeddings"):
        X_erased_sents = [dp[f'X_erased_{erased_concept}'] for dp in unified_data]
        score = run_spectral_probe(X_erased_sents, D_gold_sents, device=DEVICE, seed=SEED)
        raw_scores.loc['spectral_dist', erased_concept] = score

    # --- Step 4: RDI Calculation and Reporting ---
    print("\n[Step 3/3] Calculating RDI and reporting results...")
    for erased_concept in ERASED_CONCEPTS:
        r2_after, r2_before = raw_scores.loc['spectral_dist', erased_concept], baseline_score
        if pd.isna(r2_after) or pd.isna(r2_before) or r2_before < 1e-9:
            rdi = np.nan
        else: rdi = max(0, 1.0 - (r2_after / r2_before))
        rdi_scores.loc['spectral_dist', erased_concept] = rdi

    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"SPECTRAL PROBE RESULTS for Dataset: {args.dataset.upper()}, Method: {args.method.upper()}")
    if args.debug_subset_fraction < 1.0: print(f"*** DEBUG RUN on {args.debug_subset_fraction:.0%} of data ***")
    print("="*80)
    print("\n--- Raw Probe Performance (R-squared for Spectral Distance) ---\n")
    print(raw_scores)
    print("\n\n--- RDI Scores from Spectral Probe ---\n")
    print(rdi_scores)
    print("\n" + "="*80)
    
    debug_suffix = f"_debug_{int(args.debug_subset_fraction*100)}pct" if args.debug_subset_fraction < 1.0 else ""
    results_dir = f"Final/Results/{dataset_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    raw_scores.to_csv(f"{results_dir}/spectral_probe_raw_perf_{args.dataset}_{args.method}{debug_suffix}.csv")
    rdi_scores.to_csv(f"{results_dir}/spectral_probe_rdi_scores_{args.dataset}_{args.method}{debug_suffix}.csv")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()