"""
Performs LEACE erasure following the original paper's methodology.
This version is updated to save the full erased embeddings array for the entire
dataset, while still calculating metrics on the held-out test set.
This ensures compatibility with the final probing script.
MODIFIED: Added an optional --scaling flag.
"""
import argparse
import pickle
import os
import json
import torch
import numpy as np
from concept_erasure.leace import LeaceFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # ADDED

def main():
    parser = argparse.ArgumentParser(description="LEACE erasure for test-set evaluation.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--concept", choices=["pos", "deplab", "sd", "ld", "pe"], required=True)
    # ADDED a new argument for scaling
    parser.add_argument("--scaling", choices=["on", "off"], default="on", help="Enable StandardScaler before erasure ('on') or not ('off').")
    args = parser.parse_args()

    # --- Setup ---
    LAYER = 8
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset_name = "nar" if args.dataset == "narratives" else "ud"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concept_key_map = { 
        "pos": "pos_tags", "deplab": "deplabs", "sd": "sd", "ld": "ld", "pe": "pe" 
    }
    internal_concept_key = concept_key_map[args.concept]
    
    # --- File Paths (MODIFIED to be dynamic) ---
    scaling_suffix = "_Scaled" if args.scaling == "on" else "_Unscaled"
    dataset_dir_name_base = "UD" if args.dataset == "ud" else "Narratives"
    dataset_dir_name_scaled = dataset_dir_name_base + scaling_suffix
    
    base_dir = "Final"
    # emb_file still reads from the original, unscaled embeddings
    emb_file = f"{base_dir}/Embeddings/Original/{dataset_dir_name_base}/Embed_{dataset_name}_{args.concept}.pkl"
    # Output directories now depend on the scaling flag
    erased_dir = f"{base_dir}/Embeddings/Erased/{dataset_dir_name_scaled}"
    eraser_dir = f"{base_dir}/Eraser_objects/{dataset_dir_name_scaled}"
    results_dir = f"{base_dir}/Results/{dataset_dir_name_scaled}"
    for d in [erased_dir, eraser_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Running LEACE Erasure (Scaling: {args.scaling.upper()}) ---")
    print(f"Dataset: {args.dataset}, Concept to Erase: {args.concept}")

    with open(emb_file, "rb") as f:
        data = pickle.load(f)

    # --- Prepare Full Tensors (Unchanged) ---
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
    
    # --- Split Data (Unchanged) ---
    print("Splitting data into 80% train and 20% test sets...")
    indices = np.arange(X_full.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
    X_train, Z_train = X_full[train_indices], Z_full[train_indices]
    X_test, Z_test = X_full[test_indices], Z_full[test_indices]
    
    # =====================================================================
    # === ADDED: PREPARE DATA FOR FITTING WITH/WITHOUT SCALING ===
    # =====================================================================
    if args.scaling == "on":
        print("Fitting StandardScaler and transforming data for fitting...")
        scaler = StandardScaler()
        # Note: .fit_transform expects numpy array on CPU
        X_train_to_fit = torch.from_numpy(scaler.fit_transform(X_train.cpu().numpy())).to(device).float()
    else:
        print("Skipping scaling step.")
        scaler = None # Ensure scaler is None if not used
        X_train_to_fit = X_train

    # --- Fit Eraser on Train Set (Unchanged logic, but uses prepared data) ---
    print(f"Fitting LEACE eraser on {X_train_to_fit.shape[0]} training samples...")
    fitter = LeaceFitter.fit(X_train_to_fit, Z_train)
    eraser = fitter.eraser

    # =====================================================================
    # === MODIFIED: ERASURE LOGIC NOW HANDLES SCALING ===
    # =====================================================================

    # 1. Apply eraser to the FULL dataset, handling scaling if enabled.
    print(f"\nApplying eraser to the full dataset of {X_full.shape[0]} samples for saving...")
    if args.scaling == "on":
        # Scale -> Erase -> Inverse Scale
        X_full_scaled = torch.from_numpy(scaler.transform(X_full.cpu().numpy())).to(device).float()
        X_full_erased_scaled = eraser(X_full_scaled)
        X_full_erased = torch.from_numpy(scaler.inverse_transform(X_full_erased_scaled.cpu().numpy())).to(device)
    else:
        # Erase directly
        X_full_erased = eraser(X_full)

    # Saving logic is unchanged from your original
    erased_emb_path = f"{erased_dir}/leace_{dataset_name}_{args.concept}.pkl"
    print(f"Saving FULL erased embeddings array to: {erased_emb_path}")
    with open(erased_emb_path, "wb") as f:
        pickle.dump(X_full_erased.cpu().numpy(), f)

    # 2. Save the eraser OBJECT (Unchanged)
    eraser_obj_file = f"{eraser_dir}/eraser_obj_leace_{dataset_name}_{args.concept}.pkl"
    print(f"Saving eraser object to: {eraser_obj_file}")
    with open(eraser_obj_file, "wb") as f:
        pickle.dump(eraser, f)

    # 3. Calculate metrics on the TEST SET (Handles scaling)
    if args.scaling == "on":
        X_test_scaled = torch.from_numpy(scaler.transform(X_test.cpu().numpy())).to(device).float()
        X_test_erased_scaled = eraser(X_test_scaled)
        # We need the original X_test to compare against the un-scaled erased result
        X_test_erased = torch.from_numpy(scaler.inverse_transform(X_test_erased_scaled.cpu().numpy())).to(device)
    else:
        X_test_erased = eraser(X_test)
        
    l2_distance = torch.norm(X_test - X_test_erased, dim=1).mean().item()
    
    # Results saving is unchanged from your original
    results_file = f"{results_dir}/leace_{dataset_name}_{args.concept}_results.json"
    results = {
        "method": "leace", "dataset": args.dataset, "concept": args.concept, "scaling": args.scaling,
        "l2_distance_on_test_set": l2_distance,
        "fitter_details": {"class": "LeaceFitter", "library": "concept-erasure"},
        "notes": "Eraser trained on 80% of data. L2 distance calculated on the 20% test set."
    }
    print(f"Saving results summary to: {results_file}")
    with open(results_file, "w") as f: json.dump(results, f, indent=4)
    
    print("\nDone.")

if __name__ == "__main__":
    main()