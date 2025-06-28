"""
================================================================================
            UNIFIED ADVANCED STRUCTURAL PROBING SCRIPT
================================================================================
This script runs a comprehensive analysis using the TwoWordPSDProbe to evaluate
how well various structural properties are encoded in embeddings.

It tests the ability to recover distance metrics derived from:
- Syntactic Tree Distance (sdm)
- Spectral Vector Geometry (sd)
- Part-of-Speech Tag Identity (pos)
- Dependency Label Identity (deplab)

It produces two 4x4 result tables: Raw R-squared and RDI scores.
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
        else: return [s["embeddings_by_layer"][layer].numpy() for s in data]
    except FileNotFoundError: return None

def prepare_concept_data(dataset, concept_cli):
    dataset_dir_name = "Narratives" if dataset == "narratives" else "UD"
    dataset_name_short = "nar" if dataset == "narratives" else "ud"
    concept_map = {
        "pos": {"internal": "pos_tags", "file": "pos"}, "deplab": {"internal": "deplabs", "file": "deplab"},
        "sd": {"internal": "distance_spectral_padded", "file": "distance_spectral_padded"},
        "sdm": {"internal": "distance_matrix", "file": "distance_matrix"}
    }
    internal_key, file_key = concept_map[concept_cli]["internal"], concept_map[concept_cli]["file"]
    file_path = f"Final/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{file_key}.pkl"
    try:
        with open(file_path, "rb") as f: data = pickle.load(f)
        return [sent[internal_key].cpu().numpy() if isinstance(sent[internal_key], torch.Tensor) else sent[internal_key] for sent in data]
    except FileNotFoundError: return None

def concept_to_distance_matrix(concept_data, concept_type):
    """Converts raw concept data for a sentence into a gold distance matrix."""
    if concept_type == 'sdm':
        return concept_data # Already a distance matrix
    elif concept_type == 'sd':
        vecs = concept_data
        sum_sq = np.sum(vecs**2, axis=1)
        dot_prod = np.dot(vecs, vecs.T)
        dist_sq = sum_sq[:, np.newaxis] - 2 * dot_prod + sum_sq[np.newaxis, :]
        return np.maximum(dist_sq, 0)
    elif concept_type in ['pos', 'deplab']:
        labels = np.array(concept_data)
        return (labels[:, None] != labels).astype(np.float32)

def run_structural_probe(X_sents, D_sents, device="cuda", seed=42):
    # This function is generic and does not need to change.
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
    with torch.no_grad():
        preds_flat = np.concatenate([probe(X_test_s[i].unsqueeze(0)).squeeze(0).cpu().numpy().flatten() for i in range(len(X_test_s))])
        golds_flat = np.concatenate([d.cpu().numpy().flatten() for d in D_test_t])
            
    return r2_score(golds_flat, preds_flat)

# =========================================================================
# 2. Main Orchestrator
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run a unified structural probe analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--method", choices=["leace", "oracle"], required=True)
    parser.add_argument("--debug_subset_fraction", type=float, default=1.0)
    args = parser.parse_args()

    PROBED_CONCEPTS = ["pos", "deplab", "sd", "sdm"]
    ERASED_CONCEPTS = ["pos", "deplab", "sd", "sdm"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    print(f"--- Running Unified Structural Probe Analysis ---")
    print(f"Dataset: {args.dataset}, Method: {args.method}, Device: {DEVICE}")
    if args.debug_subset_fraction < 1.0:
        print(f"!!! DEBUG MODE ON: Using {args.debug_subset_fraction:.0%} of the data. !!!")

    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    raw_scores = pd.DataFrame(index=PROBED_CONCEPTS, columns=ERASED_CONCEPTS + ['original'], dtype=float)
    rdi_scores = pd.DataFrame(index=PROBED_CONCEPTS, columns=ERASED_CONCEPTS, dtype=float)

    print("\n[Step 1/3] Loading and creating unified dataset...")
    orig_base = f"Final/Embeddings/Original/{dataset_dir_name}"
    erased_base = f"Final/Embeddings/Erased/{dataset_dir_name}"

    X_orig_sents = load_embeddings(f"{orig_base}/Embed_{dataset_name_short}_distance_matrix.pkl") # Use sdm as canonical original
    all_Z_data = {c: prepare_concept_data(args.dataset, c) for c in PROBED_CONCEPTS}
    erased_embeddings = {c: load_embeddings(f"{erased_base}/{args.method}_{dataset_name_short}_{c}_vec.pkl") for c in ERASED_CONCEPTS}
    
    unified_data = []
    for i in range(len(X_orig_sents)):
        is_valid = all(all_Z_data.get(c) and len(all_Z_data[c]) > i for c in PROBED_CONCEPTS) and \
                   all(erased_embeddings.get(c) and len(erased_embeddings[c]) > i for c in ERASED_CONCEPTS)
        if not is_valid: continue

        base_len = len(X_orig_sents[i])
        is_consistent = all(len(all_Z_data[c][i]) == base_len for c in PROBED_CONCEPTS) and \
                        all(len(erased_embeddings[c][i]) == base_len for c in ERASED_CONCEPTS)
        if not is_consistent: continue

        data_point = {'X_original': X_orig_sents[i]}
        for c in PROBED_CONCEPTS: data_point[f'Z_{c}'] = all_Z_data[c][i]
        for c in ERASED_CONCEPTS: data_point[f'X_erased_{c}'] = erased_embeddings[c][i]
        unified_data.append(data_point)

    print(f"Created a unified dataset with {len(unified_data)} fully consistent sentences.")

    if args.debug_subset_fraction < 1.0:
        num_to_keep = int(len(unified_data) * args.debug_subset_fraction)
        random.Random(SEED).shuffle(unified_data)
        unified_data = unified_data[:num_to_keep]
        print(f"--- Subset size: {len(unified_data)} sentences. ---")

    print("\n[Step 2/3] Running probes...")
    for probed_concept in tqdm(PROBED_CONCEPTS, desc="Probing Concept"):
        # Prepare the gold standard distance matrices for this concept
        D_gold_sents = [concept_to_distance_matrix(dp[f'Z_{probed_concept}'], probed_concept) for dp in unified_data]
        
        # Baseline
        X_original_sents = [dp['X_original'] for dp in unified_data]
        baseline_score = run_structural_probe(X_original_sents, D_gold_sents, DEVICE, SEED)
        raw_scores.loc[probed_concept, 'original'] = baseline_score

        # Erased
        for erased_concept in ERASED_CONCEPTS:
            X_erased_sents = [dp[f'X_erased_{erased_concept}'] for dp in unified_data]
            score = run_structural_probe(X_erased_sents, D_gold_sents, DEVICE, SEED)
            raw_scores.loc[probed_concept, erased_concept] = score

    print("\n[Step 3/3] Calculating RDI and reporting results...")
    for p_concept in PROBED_CONCEPTS:
        for e_concept in ERASED_CONCEPTS:
            r2_after, r2_before = raw_scores.loc[p_concept, e_concept], raw_scores.loc[p_concept, 'original']
            if pd.isna(r2_after) or pd.isna(r2_before) or r2_before < 1e-9:
                rdi = np.nan
            else: rdi = max(0, 1 - (r2_after / r2_before))
            rdi_scores.loc[p_concept, e_concept] = rdi

    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"UNIFIED STRUCTURAL PROBE RESULTS for Dataset: {args.dataset.upper()}, Method: {args.method.upper()}")
    if args.debug_subset_fraction < 1.0: print(f"*** DEBUG RUN on {args.debug_subset_fraction:.0%} of data ***")
    print("="*80)
    print("\n--- Raw Probe Performance (R-squared) ---\n")
    print(raw_scores)
    print("\n\n--- RDI Scores from Structural Probe ---\n")
    print(rdi_scores)
    print("\n" + "="*80)
    
    debug_suffix = f"_debug_{int(args.debug_subset_fraction*100)}pct" if args.debug_subset_fraction < 1.0 else ""
    results_dir = f"Final/Results/{dataset_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    raw_scores.to_csv(f"{results_dir}/unified_struct_raw_perf_{args.dataset}_{args.method}{debug_suffix}.csv")
    rdi_scores.to_csv(f"{results_dir}/unified_struct_rdi_scores_{args.dataset}_{args.method}{debug_suffix}.csv")
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()