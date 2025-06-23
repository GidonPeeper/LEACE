"""
================================================================================
                    THE UNIFIED, FINAL PROBING SCRIPT
================================================================================
This script runs a comprehensive probing analysis on original and erased embeddings.
It combines two types of probes:
1.  Linear Probes:
    - Logistic Regression for discrete concepts (pos, deplab).
    - Ridge Regression for continuous concepts (sd, sdm).
2.  Advanced Structural Probe:
    - TwoWordPSDProbe for the non-linear concept of syntactic distance.

It produces two sets of 4x4 tables for a given dataset and erasure method:
- One set for the Linear Probes (Raw Performance + RDI).
- One set for the Advanced Structural Probe (Raw Performance + RDI).
"""
import argparse
import pickle
import numpy as np
import torch
import os
import sys
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm

# Ensure the structural_probes submodule is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from external.structural_probes.structural_probe.probe import TwoWordPSDProbe

# =========================================================================
# 1. Data Loading and Preparation Helpers
# =========================================================================

def load_embeddings(file_path, layer=8):
    """Loads embeddings from a .pkl file into a list of numpy arrays."""
    try:
        with open(file_path, 'rb') as f: data = pickle.load(f)
        if isinstance(data, dict) and "all_erased" in data:
            return data["all_erased"]
        else:
            return [s["embeddings_by_layer"][layer].numpy() for s in data]
    except FileNotFoundError:
        print(f"Warning: Could not load embeddings from {file_path}")
        return None

def prepare_concept_data(dataset, concept_cli, layer=8):
    """Loads and prepares the ground truth data (Z) for a given concept."""
    dataset_dir_name = "Narratives" if dataset == "narratives" else "UD"
    dataset_name_short = "nar" if dataset == "narratives" else "ud"
    
    # This map defines the internal keys and filenames for each concept
    concept_map = {
        "pos": {"internal": "pos_tags", "file": "pos"},
        "deplab": {"internal": "deplabs", "file": "deplab"},
        "sd": {"internal": "distance_spectral_padded", "file": "distance_spectral_padded"},
        "sdm": {"internal": "distance_matrix", "file": "distance_matrix"}
    }
    internal_key = concept_map[concept_cli]["internal"]
    file_key = concept_map[concept_cli]["file"]
    file_path = f"Final/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{file_key}.pkl"

    try:
        with open(file_path, "rb") as f: data = pickle.load(f)
    except FileNotFoundError: return None, None

    if concept_cli in ["pos", "deplab"]:
        all_labels_flat = [lab for sent in data for lab in sent.get(internal_key, [])]
        if not all_labels_flat: return None, None
        counts = Counter(all_labels_flat)
        chance = counts.most_common(1)[0][1] / len(all_labels_flat)
        return all_labels_flat, chance

    elif concept_cli == "sd":
        Z = np.vstack([sent[internal_key].cpu().numpy() for sent in data])
        return Z, 0.0

    elif concept_cli == "sdm":
        all_sents = [s["distance_matrix"] for s in data]
        max_len = max(s.shape[0] for s in all_sents)
        all_z_vectors = []
        for dist_matrix in all_sents:
            sent_len = dist_matrix.shape[0]
            for i in range(sent_len):
                row_vector = dist_matrix[i, :]
                padded_vector = np.pad(row_vector, (0, max_len - sent_len), 'constant')
                all_z_vectors.append(padded_vector)
        Z = np.vstack(all_z_vectors).astype(np.float32)
        return Z, 0.0

# =========================================================================
# 2. Probe Implementations
# =========================================================================

def run_linear_probe(X, Z, concept_type, seed=42):
    """Trains and evaluates a linear probe."""
    if isinstance(Z, list): Z = np.array(Z)
    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.3, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    
    if concept_type in ["pos", "deplab"]:
        probe = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        probe.fit(X_train_scaled, Z_train)
        return accuracy_score(Z_test, probe.predict(X_test_scaled))
    else: # "sd", "sdm"
        probe = Ridge(random_state=seed)
        probe.fit(X_train_scaled, Z_train)
        return r2_score(Z_test, probe.predict(X_test_scaled))

def run_structural_probe(embeddings_path, gold_distances_path, device="cuda", seed=42):
    """Runs the advanced structural probe for syntactic distance."""
    # This is a simplified version of your working script's logic
    def load_sents(path):
        with open(path, 'rb') as f: data = pickle.load(f)
        return data["all_erased"] if "erased" in str(path) else [s['embeddings_by_layer'][8] for s in data]
    def load_dists(path):
        with open(path, 'rb') as f: data = pickle.load(f)
        return [s['distance_matrix'] for s in data]

    all_X, all_D = load_sents(embeddings_path), load_dists(gold_distances_path)
    
    X_clean, D_clean = [], []
    for i in range(min(len(all_X), len(all_D))):
        x_np = all_X[i].numpy() if isinstance(all_X[i], torch.Tensor) else all_X[i]
        if np.isfinite(all_D[i]).all() and all_D[i].shape[0] == x_np.shape[0] and x_np.shape[0] > 0:
            X_clean.append(all_X[i]); D_clean.append(all_D[i])
            
    if not X_clean: return np.nan

    indices = range(len(X_clean))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=seed)
    X_train, D_train = [X_clean[i] for i in train_indices], [D_clean[i] for i in train_indices]
    X_test, D_test = [X_clean[i] for i in test_indices], [D_clean[i] for i in test_indices]

    if not X_train or not X_test: return np.nan

    scaler = StandardScaler().fit(np.vstack([x.numpy() if isinstance(x, torch.Tensor) else x for x in X_train]))
    
    def apply_scaler(sents, scaler, dev):
        flat = np.vstack([x.numpy() if isinstance(x, torch.Tensor) else x for x in sents])
        scaled = scaler.transform(flat)
        res, idx = [], 0
        for x in sents:
            n = x.shape[0]; res.append(torch.tensor(scaled[idx:idx+n], dtype=torch.float32, device=dev)); idx += n
        return res

    X_train_s, X_test_s = apply_scaler(X_train, scaler, device), apply_scaler(X_test, scaler, device)
    D_train_t, D_test_t = [torch.tensor(d, device=device) for d in D_train], [torch.tensor(d, device=device) for d in D_test]

    probe_args = {'probe':{'maximum_rank':128}, 'model':{'hidden_dim':X_train_s[0].shape[1]}, 'device':device}
    probe = TwoWordPSDProbe(probe_args)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    
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
# 3. Main Orchestrator
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run a full linear and non-linear probing analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to probe.")
    parser.add_argument("--method", choices=["leace", "oracle"], required=True, help="Erasure method to probe.")
    args = parser.parse_args()

    # --- Setup ---
    CONCEPTS = ["pos", "deplab", "sd", "sdm"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    print(f"--- Running Full Probe Analysis ---")
    print(f"Dataset: {args.dataset}, Method: {args.method}, Device: {DEVICE}")

    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    # Dataframes for results
    linear_raw = pd.DataFrame(index=CONCEPTS, columns=CONCEPTS + ['original'], dtype=float)
    linear_rdi = pd.DataFrame(index=CONCEPTS, columns=CONCEPTS, dtype=float)
    struct_raw = pd.DataFrame(index=['distance'], columns=CONCEPTS + ['original'], dtype=float)
    struct_rdi = pd.DataFrame(index=['distance'], columns=CONCEPTS, dtype=float)

    orig_base = f"Final/Embeddings/Original/{dataset_dir_name}"
    erased_base = f"Final/Embeddings/Erased/{dataset_dir_name}"

    # --- Phase 1: Linear Probing ---
    print("\n" + "="*40 + "\n[Phase 1/2] Running Linear Probes\n" + "="*40)
    baseline_linear_scores = {}
    chance_scores = {}
    
    for concept in tqdm(CONCEPTS, desc="Baseline Linear Probes"):
        Z, chance = prepare_concept_data(args.dataset, concept)
        # We need the original embeddings generated for THAT concept for a fair baseline
        concept_map = {"pos":"pos", "deplab":"deplab", "sd":"distance_spectral_padded", "sdm":"distance_matrix"}
        emb_path = f"{orig_base}/Embed_{dataset_name_short}_{concept_map[concept]}.pkl"
        X = np.vstack(load_embeddings(emb_path))
        
        baseline_linear_scores[concept] = run_linear_probe(X, Z, concept, SEED)
        chance_scores[concept] = chance
        linear_raw.loc[concept, 'original'] = baseline_linear_scores[concept]
        
    for erased_concept in tqdm(CONCEPTS, desc="Erased Linear Probes "):
        emb_path = f"{erased_base}/{args.method}_{dataset_name_short}_{erased_concept}_vec.pkl"
        if not os.path.exists(emb_path): continue
        X_erased = np.vstack(load_embeddings(emb_path))
        
        for probed_concept in CONCEPTS:
            Z, _ = prepare_concept_data(args.dataset, probed_concept)
            score = run_linear_probe(X_erased, Z, probed_concept, SEED)
            linear_raw.loc[probed_concept, erased_concept] = score

    # Calculate Linear RDI
    for p_concept in CONCEPTS:
        for e_concept in CONCEPTS:
            p_after, p_before, chance = linear_raw.loc[p_concept, e_concept], baseline_linear_scores.get(p_concept), chance_scores.get(p_concept)
            if pd.isna(p_after) or p_before is None or pd.isna(p_before): rdi = np.nan
            else:
                denom = p_before - chance if p_concept in ['pos', 'deplab'] else p_before
                numer = p_after - chance if p_concept in ['pos', 'deplab'] else p_after
                rdi = max(0, 1 - (numer/denom)) if denom > 1e-6 else np.nan
            linear_rdi.loc[p_concept, e_concept] = rdi


    # --- Phase 2: Advanced Structural Probing ---
    print("\n" + "="*40 + "\n[Phase 2/2] Running Advanced Structural Probe\n" + "="*40)
    gold_dist_path = f"{orig_base}/Embed_{dataset_name_short}_distance_matrix.pkl"
    baseline_emb_path = gold_dist_path

    print("  Calculating structural baseline...")
    struct_baseline = run_structural_probe(baseline_emb_path, gold_dist_path, DEVICE, SEED)
    struct_raw.loc['distance', 'original'] = struct_baseline
    
    for erased_concept in tqdm(CONCEPTS, desc="Erased Structural Probes"):
        emb_path = f"{erased_base}/{args.method}_{dataset_name_short}_{erased_concept}_vec.pkl"
        if not os.path.exists(emb_path): continue
        score = run_structural_probe(emb_path, gold_dist_path, DEVICE, SEED)
        struct_raw.loc['distance', erased_concept] = score

    # Calculate Structural RDI
    for e_concept in CONCEPTS:
        r2_after, r2_before = struct_raw.loc['distance', e_concept], struct_baseline
        if pd.isna(r2_after) or pd.isna(r2_before) or r2_before < 1e-6: rdi = np.nan
        else: rdi = max(0, 1-(r2_after/r2_before))
        struct_rdi.loc['distance', e_concept] = rdi


    # --- Phase 3: Reporting ---
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"      FINAL PROBE RESULTS for Dataset: {args.dataset.upper()}, Method: {args.method.upper()}")
    print("="*80)
    
    print("\n\n--- [LINEAR PROBES] Raw Performance (Accuracy / R-squared) ---")
    print("Columns: Concept Erased From Embeddings, Rows: Concept Being Probed\n")
    print(linear_raw)
    
    print("\n\n--- [LINEAR PROBES] RDI (Remnant-to-Discarded Information) Scores ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(linear_rdi)
    
    print("\n\n--- [ADVANCED STRUCTURAL PROBE] Raw Performance (R-squared) ---")
    print("Columns: Concept Erased From Embeddings, Row: Probing for Syntactic Distance\n")
    print(struct_raw)

    print("\n\n--- [ADVANCED STRUCTURAL PROBE] RDI Scores ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(struct_rdi)

    print("\n" + "="*80)
    
    results_dir = f"Final/Results/{dataset_dir_name}"
    os.makedirs(results_dir, exist_ok=True)
    linear_raw.to_csv(f"{results_dir}/linear_probe_raw_perf_{args.dataset}_{args.method}.csv")
    linear_rdi.to_csv(f"{results_dir}/linear_probe_rdi_scores_{args.dataset}_{args.method}.csv")
    struct_raw.to_csv(f"{results_dir}/struct_probe_raw_perf_{args.dataset}_{args.method}.csv")
    struct_rdi.to_csv(f"{results_dir}/struct_probe_rdi_scores_{args.dataset}_{args.method}.csv")
    print(f"\nAll results saved to {results_dir}")

if __name__ == "__main__":
    main()