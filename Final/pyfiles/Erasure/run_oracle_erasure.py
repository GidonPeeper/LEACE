"""
Performs ORACLE "in-sample" erasure using the official `concept-erasure`
library.

This experiment is designed to measure the maximum possible linear erasure on a
given dataset. It fits an eraser on the entire dataset and then applies it
to that same dataset.

The eraser produced by OracleFitter requires the concept labels `z` to be
provided again at erasure time, in exchange for a more surgical erasure.
"""
import argparse
import pickle
import os
import json
import torch
from concept_erasure.oracle import OracleFitter

# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(description="Oracle erasure with official library.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--concept", choices=["pos", "deplab", "sd"], required=True)
    args = parser.parse_args()

    # --- Setup ---
    LAYER = 8
    SEED = 42
    torch.manual_seed(SEED)
    dataset_name = "nar" if args.dataset == "narratives" else "ud"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    concept_key_map = {
        "pos": "pos_tags",
        "deplab": "deplabs",
        "sd": "distance_spectral_padded"
    }
    concept_key = concept_key_map[args.concept]
    
    # --- File Paths ---
    base_dir = "Final"
    dataset_dir_name = "UD" if args.dataset == "ud" else args.dataset.capitalize()
    emb_file = f"{base_dir}/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name}_{concept_key}.pkl"
    
    erased_dir = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}"
    eraser_dir = f"{base_dir}/Eraser_objects/{dataset_dir_name}"
    results_dir = f"{base_dir}/Results/{dataset_dir_name}"
    
    base_name = f"oracle_{dataset_name}_{args.concept}"
    erased_emb_file = f"{erased_dir}/{base_name}_vec.pkl"
    eraser_obj_file = f"{eraser_dir}/eraser_obj_{base_name}.pkl"
    results_file = f"{results_dir}/{base_name}_results.json"
    
    for d in [erased_dir, eraser_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Running ORACLE Erasure (In-sample) ---")
    print(f"Dataset: {args.dataset}, Concept: {args.concept}")

    # --- Data Loading ---
    with open(emb_file, "rb") as f: data = pickle.load(f)

    # --- Prepare Tensors ---
    X_full = torch.cat([s["embeddings_by_layer"][LAYER] for s in data]).to(device).float()
    if args.concept in ["pos", "deplab"]:
        all_labels = sorted(list(set(lab for s in data for lab in s[concept_key])))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        indices = torch.tensor([label_to_idx[lab] for s in data for lab in s[concept_key]])
        Z_full = torch.nn.functional.one_hot(indices, num_classes=len(all_labels)).to(device).float()
    else: # sd
        Z_full = torch.cat([s[concept_key] for s in data]).to(device).float()

    # --- Fit and Erase on Full Dataset ---
    print(f"Fitting OracleFitter on full dataset ({len(X_full)} samples)...")
    fitter = OracleFitter.fit(X_full, Z_full)
    eraser = fitter.eraser

    print("Applying the oracle eraser...")
    # Note: OracleEraser requires both X and Z for the erasure call
    X_full_erased = eraser(X_full, Z_full)

    # --- Save Outputs ---
    X_full_erased_np = X_full_erased.cpu().numpy()
    all_erased_sent_wise = []
    idx = 0
    for sent in data:
        n = sent["embeddings_by_layer"][LAYER].shape[0]
        all_erased_sent_wise.append(X_full_erased_np[idx:idx+n])
        idx += n

    with open(erased_emb_file, "wb") as f: pickle.dump({"all_erased": all_erased_sent_wise}, f)
    with open(eraser_obj_file, "wb") as f: pickle.dump(eraser, f)

    # --- Save Results Summary ---
    results = {
        "method": "oracle", "dataset": args.dataset, "concept": args.concept,
        "l2_distance": torch.norm(X_full - X_full_erased, dim=1).mean().item(),
        "fitter_details": {"class": "OracleFitter", "library": "concept-erasure"}
    }
    with open(results_file, "w") as f: json.dump(results, f, indent=4)
    
    print(f"\nDone. Erased embeddings saved to {erased_emb_file}")

if __name__ == "__main__":
    main()