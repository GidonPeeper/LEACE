"""
Performs ORACLE "in-sample" erasure using the official `concept-erasure`
library. This version is updated to save the full erased embeddings as a single
NumPy array, consistent with the LEACE script's output format.
"""
import argparse
import pickle
import os
import json
import torch
import numpy as np
from concept_erasure.oracle import OracleFitter

def main():
    parser = argparse.ArgumentParser(description="Oracle erasure with official library.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--concept", choices=["pos", "deplab", "sd", "ld", "pe"], required=True)
    args = parser.parse_args()

    # --- Setup ---
    LAYER = 8
    SEED = 42
    torch.manual_seed(SEED)
    dataset_name = "nar" if args.dataset == "narratives" else "ud"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    concept_key_map = { 
        "pos": "pos_tags", "deplab": "deplabs", "sd": "sd", "ld": "ld", "pe": "pe" 
    }
    internal_concept_key = concept_key_map[args.concept]
    
    # --- File Paths ---
    dataset_dir_name = "UD" if args.dataset == "ud" else "Narratives"
    base_dir = "Final"
    emb_file = f"{base_dir}/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name}_{args.concept}.pkl"
    erased_dir = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}"
    results_dir = f"{base_dir}/Results/{dataset_dir_name}"
    for d in [erased_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Running ORACLE Erasure (In-sample) ---")
    print(f"Dataset: {args.dataset}, Concept to Erase: {args.concept}")

    with open(emb_file, "rb") as f:
        data = pickle.load(f)

    # --- Prepare Full Tensors ---
    print("Preparing full data tensors...")
    X_full = torch.cat([s["embeddings_by_layer"][LAYER] for s in data]).to(device).float()
    if args.concept in ["pos", "deplab"]:
        all_labels = sorted(list(set(lab for s in data for lab in s[internal_concept_key])))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        indices = torch.tensor([label_to_idx[lab] for s in data for lab in s[internal_concept_key]])
        Z_full = torch.nn.functional.one_hot(indices, num_classes=len(all_labels)).to(device).float()
    elif args.concept == 'ld':
        ld_flat = [val for s in data for val in s[internal_concept_key]]
        Z_full = torch.tensor(ld_flat, device=device).float().unsqueeze(1)
    else: # sd, pe
        Z_full = torch.cat([s[internal_concept_key] for s in data]).to(device).float()
    
    # --- Fit and Erase on Full Dataset ---
    print(f"Fitting Oracle eraser and erasing all {X_full.shape[0]} samples...")
    fitter = OracleFitter.fit(X_full, Z_full)
    eraser = fitter.eraser
    X_full_erased = eraser(X_full, Z_full) # Note: Oracle eraser needs Z during application

    # =====================================================================
    # === UPDATED SAVING LOGIC FOR CONSISTENCY ===
    # =====================================================================

    # 1. Save the FULL erased embeddings as a single NumPy array.
    #    This makes the format identical to the LEACE script's output.
    erased_emb_path = f"{erased_dir}/oracle_{dataset_name}_{args.concept}.pkl"
    print(f"\nSaving FULL erased embeddings array to: {erased_emb_path}")
    with open(erased_emb_path, "wb") as f:
        pickle.dump(X_full_erased.cpu().numpy(), f)

    # 2. Calculate metrics on the FULL SET and save the results JSON.
    #    This is the correct evaluation procedure for Oracle.
    l2_distance = torch.norm(X_full - X_full_erased, dim=1).mean().item()

    results_file = f"{results_dir}/oracle_{dataset_name}_{args.concept}_results.json"
    results = {
        "method": "oracle", "dataset": args.dataset, "concept": args.concept,
        "l2_distance_on_full_set": l2_distance,
        "fitter_details": {"class": "OracleFitter", "library": "concept-erasure"},
        "notes": "Eraser trained and L2 distance calculated on the full dataset."
    }
    print(f"Saving results summary to: {results_file}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nDone.")

if __name__ == "__main__":
    main()