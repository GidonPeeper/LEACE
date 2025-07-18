"""
Performs ORACLE "in-sample" erasure using the official `concept-erasure`
library. This version is updated to save the full erased embeddings as a single
NumPy array, consistent with the LEACE script's output format.
With optional --scaling flag.
"""
import argparse
import pickle
import os
import json
import torch
import numpy as np
from concept_erasure.oracle import OracleFitter
from sklearn.preprocessing import StandardScaler 

def main():
    parser = argparse.ArgumentParser(description="Oracle erasure with official library.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--concept", choices=["pos", "deplab", "sd", "ld", "pe"], required=True)
    parser.add_argument("--scaling", choices=["on", "off"], default="on", help="Enable StandardScaler before erasure ('on') or not ('off').")
    parser.add_argument("--data_fraction", type=int, choices=[1, 10, 100], default=100, help="Percentage of the dataset to use, corresponding to the pre-generated embedding file.")
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
    
    # --- File Paths (MODIFIED to be dynamic) ---
    # Create suffix for data fraction to append to filenames
    file_suffix = f"_{args.data_fraction}pct" if args.data_fraction < 100 else ""

    scaling_suffix = "_Scaled" if args.scaling == "on" else "_Unscaled"
    dataset_dir_name_base = "UD" if args.dataset == "ud" else "Narratives"
    dataset_dir_name_scaled = dataset_dir_name_base + scaling_suffix
    
    base_dir = "Final"
    # emb_file now points to the potentially subsetted, original embeddings
    emb_file = f"{base_dir}/Embeddings/Original/{dataset_dir_name_base}/Embed_{dataset_name}_{args.concept}{file_suffix}.pkl"
    erased_dir = f"{base_dir}/Embeddings/Erased/{dataset_dir_name_scaled}"
    results_dir = f"{base_dir}/Results/{dataset_dir_name_scaled}"
    for d in [erased_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Running ORACLE Erasure (Scaling: {args.scaling.upper()}) ---")
    print(f"Dataset: {args.dataset}, Concept: {args.concept}, Data Fraction: {args.data_fraction}%")
    print(f"Loading embeddings from: {emb_file}")

    try:
        with open(emb_file, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {emb_file}")
        print("Please ensure you have generated the embeddings for this data fraction first.")
        return

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
    
    # =====================================================================
    # === PREPARE DATA FOR FITTING WITH/WITHOUT SCALING ===
    # =====================================================================
    if args.scaling == "on":
        print("Fitting StandardScaler and transforming data...")
        scaler = StandardScaler()
        X_to_fit = torch.from_numpy(scaler.fit_transform(X_full.cpu().numpy())).to(device).float()
    else:
        print("Skipping scaling step.")
        X_to_fit = X_full

    # --- Fit and Erase on Full Dataset ---
    print(f"Fitting Oracle eraser and erasing all {X_to_fit.shape[0]} samples...")
    fitter = OracleFitter.fit(X_to_fit, Z_full)
    eraser = fitter.eraser
    X_erased_processed = eraser(X_to_fit, Z_full)

    # =====================================================================
    # === ERASURE LOGIC NOW HANDLES SCALING ===
    # =====================================================================

    # 1. Inverse transform if necessary to return to original embedding space.
    if args.scaling == "on":
        X_full_erased = torch.from_numpy(scaler.inverse_transform(X_erased_processed.cpu().numpy())).to(device)
    else:
        X_full_erased = X_erased_processed

    # 2. Save the final unscaled erased embeddings.
    erased_emb_path = f"{erased_dir}/oracle_{dataset_name}_{args.concept}{file_suffix}.pkl"
    print(f"\nSaving FULL erased embeddings array to: {erased_emb_path}")
    with open(erased_emb_path, "wb") as f:
        pickle.dump(X_full_erased.cpu().numpy(), f)

    # 3. Calculate metrics on the FULL SET and save results.
    l2_distance = torch.norm(X_full - X_full_erased, dim=1).mean().item()

    # Results saving 
    results_file = f"{results_dir}/oracle_{dataset_name}_{args.concept}{file_suffix}_results.json"
    results = {
        "method": "oracle", "dataset": args.dataset, "concept": args.concept, "scaling": args.scaling,
        "data_fraction_used": args.data_fraction,
        "l2_distance_on_full_set": l2_distance,
        "fitter_details": {"class": "OracleFitter", "library": "concept-erasure"},
        "notes": f"Eraser trained and L2 distance calculated on the full {args.data_fraction}% data subset."
    }
    print(f"Saving results summary to: {results_file}")
    with open(results_file, "w") as f: json.dump(results, f, indent=4)
    
    print("\nDone.")

if __name__ == "__main__":
    main()