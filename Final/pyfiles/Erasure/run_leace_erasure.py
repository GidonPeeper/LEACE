"""
Performs LEACE erasure following the original paper's methodology.

It learns an eraser on the training set (80%) and then saves a dedicated file
containing the original and erased versions of the HELD-OUT TEST SET (20%),
along with the corresponding labels. This file is specifically designed to be
used by the probing script for a pure generalizability evaluation.
"""
import argparse
import pickle
import os
import torch
import numpy as np
from concept_erasure.leace import LeaceFitter
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="LEACE erasure for test-set evaluation.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--concept", choices=["pos", "deplab", "sd"], required=True)
    args = parser.parse_args()

    # --- Setup ---
    LAYER = 8
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset_name = "nar" if args.dataset == "narratives" else "ud"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concept_key_map = {
        "pos": "pos_tags",
        "deplab": "deplabs",
        "sd": "distance_spectral_padded"
    }
    concept_key = concept_key_map[args.concept]
    
    # --- File Paths ---
    dataset_dir_name = "UD" if args.dataset == "ud" else "Narratives"
    base_dir = "Final"
    emb_file = f"{base_dir}/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name}_{concept_key}.pkl"
    erased_dir = f"{base_dir}/Embeddings/Erased/{dataset_dir_name}"
    os.makedirs(erased_dir, exist_ok=True)

    print(f"--- Running LEACE Erasure (for Test Set Evaluation) ---")
    print(f"Dataset: {args.dataset}, Concept to Erase: {args.concept}")

    with open(emb_file, "rb") as f: data = pickle.load(f)

    # --- Prepare Full Tensors ---
    X_full = torch.cat([s["embeddings_by_layer"][LAYER] for s in data]).to(device).float()
    if args.concept in ["pos", "deplab"]:
        all_labels = sorted(list(set(lab for s in data for lab in s[concept_key])))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        indices = torch.tensor([label_to_idx[lab] for s in data for lab in s[concept_key]])
        Z_full = torch.nn.functional.one_hot(indices, num_classes=len(all_labels)).to(device).float()
    else: # sd
        Z_full = torch.cat([s[concept_key] for s in data]).to(device).float()
    
    # --- Split Data into Master Train/Test Sets ---
    indices = np.arange(X_full.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
    
    X_train, Z_train = X_full[train_indices], Z_full[train_indices]
    X_test, Z_test = X_full[test_indices], Z_full[test_indices]

    # --- Fit Eraser on TRAIN ONLY ---
    print(f"Fitting LeaceFitter on training data ({len(X_train)} samples)...")
    fitter = LeaceFitter.fit(X_train, Z_train)
    eraser = fitter.eraser

    # --- Erase the HELD-OUT TEST SET ---
    print(f"Applying eraser to the held-out test set ({len(X_test)} samples)...")
    X_test_erased = eraser(X_test)

    # --- Save the Test Set Data for Probing ---
    # This single file contains everything the probing script needs for this concept.
    save_path = f"{erased_dir}/leace_{dataset_name}_{args.concept}_for_probing.pkl"
    print(f"\nSaving test set data for probing to: {save_path}")
    
    save_data = {
        "X_test_original": X_test.cpu().numpy(),
        "X_test_erased": X_test_erased.cpu().numpy(),
        "Z_test": Z_test.cpu().numpy(),
        "concept": args.concept # Store for reference
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    print("\nDone.")

if __name__ == "__main__":
    main()