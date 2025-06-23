"""
Unified script for concept erasure using LEACE (Linear Concept Erasure).

This script processes a specified dataset and concept, automatically performing
both standard LEACE (with a train/test split) and Oracle LEACE (fitting on
the full dataset). It generates all corresponding erased embedding files,
eraser objects, and results reports for both methods in a single run.

It supports four concepts:
- 'pos': Part-of-speech tags (categorical)
- 'deplab': Dependency relations (categorical)
- 'sd': Padded spectral graph embeddings (vector per word)
- 'sdm': Syntactic distance matrix (matrix per sentence)

Example usage:
    python run_erasure.py --dataset narratives --concept sd
    python run_erasure.py --dataset ud --concept pos
"""
import argparse
import pickle
import numpy as np
import os
import json
from collections import Counter
from sklearn.model_selection import train_test_split

# =========================================================================
# 1. Custom, Reliable LEACE Eraser Class
# =========================================================================
class LeaceEraser:
    """A reliable implementation of multiclass/multi-output LEACE using OLS regression."""
    def __init__(self):
        self.W, self.mean_x_train, self.mean_z_train = None, None, None
    def fit(self, X_train: np.ndarray, Z_train: np.ndarray):
        self.mean_x_train = np.mean(X_train, axis=0, keepdims=True)
        self.mean_z_train = np.mean(Z_train, axis=0, keepdims=True)
        X_train_c = X_train.astype(np.float64) - self.mean_x_train.astype(np.float64)
        Z_train_c = Z_train.astype(np.float64) - self.mean_z_train.astype(np.float64)
        Σ_XZ = X_train_c.T @ Z_train_c
        Σ_ZZ = Z_train_c.T @ Z_train_c
        self.W = Σ_XZ @ np.linalg.pinv(Σ_ZZ)
        self.W = self.W.astype(X_train.dtype)
    def erase(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.W is None: raise RuntimeError("Must call .fit() before .erase()")
        X_c = X - self.mean_x_train
        Z_c = Z - self.mean_z_train
        X_proj = Z_c @ self.W.T
        X_erased_c = X_c - X_proj
        return X_erased_c + self.mean_x_train

# =========================================================================
# 2. Helper Functions
# =========================================================================

def get_concept_vectors(data, concept_cli, concept_internal_key):
    """Prepares the concept vector matrix (Z) based on the concept type."""
    if concept_cli in ["pos", "deplab"]:
        all_labels_flat = [lab for sent in data for lab in sent.get(concept_internal_key, [])]
        if not all_labels_flat:
            raise ValueError(f"No data found for concept key '{concept_internal_key}'.")
        all_labels = sorted(list(set(all_labels_flat)))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        num_labels = len(all_labels)
        print(f"Found {num_labels} unique '{concept_cli}' labels.")
        y_indices = np.array([label_to_idx[val] for val in all_labels_flat], dtype=int)
        y_onehot = np.zeros((len(y_indices), num_labels), dtype=np.float32)
        y_onehot[np.arange(len(y_indices)), y_indices] = 1.0
        return y_onehot[:, :-1]
    
    elif concept_cli == "sd":
        Z_full = np.vstack([sent[concept_internal_key].cpu().numpy() for sent in data])
        print(f"Stacked '{concept_internal_key}' vectors.")
        return Z_full

    elif concept_cli == "sdm":
        print(f"Processing '{concept_internal_key}' matrices into word-level vectors...")
        max_len = max(sent[concept_internal_key].shape[0] for sent in data)
        print(f"Max sentence length is {max_len}. Padding all distance vectors to this size.")
        
        all_z_vectors = []
        for sent in data:
            dist_matrix = sent[concept_internal_key].numpy()
            sent_len = dist_matrix.shape[0]
            for i in range(sent_len):
                row_vector = dist_matrix[i, :]
                padded_vector = np.pad(row_vector, (0, max_len - sent_len), 'constant', constant_values=0)
                all_z_vectors.append(padded_vector)
        
        Z_full = np.vstack(all_z_vectors).astype(np.float32)
        print(f"Created Z matrix by padding and stacking matrix rows.")
        return Z_full
    else:
        raise ValueError(f"Unknown concept type: {concept_cli}")


def perform_erasure_and_save(X_full, Z_full, data, base_paths, method, seed):
    """Performs a single erasure run (either 'leace' or 'oracle') and saves all outputs."""
    print(f"\n--- Performing {method.upper()} LEACE ---")
    
    eraser = LeaceEraser()
    if method == "oracle":
        eraser.fit(X_full, Z_full)
        X_full_erased = eraser.erase(X_full, Z_full)
    elif method == "leace":
        indices = np.arange(X_full.shape[0])
        X_train, X_test, Z_train, Z_test, train_indices, test_indices = train_test_split(
            X_full, Z_full, indices, test_size=0.2, random_state=seed)
        eraser.fit(X_train, Z_train)
        X_erased_train = eraser.erase(X_train, Z_train)
        X_erased_test = eraser.erase(X_test, Z_test)
        X_full_erased = np.empty_like(X_full)
        X_full_erased[train_indices] = X_erased_train
        X_full_erased[test_indices] = X_erased_test
    else:
        raise ValueError(f"Unknown method: {method}")

    all_erased_sent_wise = []
    idx = 0
    for sent in data:
        n = sent["embeddings_by_layer"][base_paths['layer']].shape[0]
        all_erased_sent_wise.append(X_full_erased[idx:idx+n])
        idx += n

    erased_emb_file = base_paths['erased_emb_file'].format(method=method)
    eraser_obj_file = base_paths['eraser_obj_file'].format(method=method)
    print(f"Saving erased embeddings to {erased_emb_file}")
    with open(erased_emb_file, "wb") as f: pickle.dump({"all_erased": all_erased_sent_wise}, f)
    
    print(f"Saving eraser object to {eraser_obj_file}")
    with open(eraser_obj_file, "wb") as f: pickle.dump(eraser, f)

    l2_distance = np.linalg.norm(X_full - X_full_erased, axis=1).mean()
    results = {
        "dataset": base_paths['dataset_cli'], "concept": base_paths['concept_cli'], "method": method,
        "layer": base_paths['layer'], "l2_distance": float(l2_distance),
        "z_rank": int(np.linalg.matrix_rank(Z_full)),
        "fitting_data": "full_dataset" if method == "oracle" else f"train_split (80%, seed={seed})"}
    
    if base_paths['concept_cli'] in ["pos", "deplab"]:
        all_labels_flat = [lab for sent in data for lab in sent.get(base_paths['concept_internal_key'], [])]
        counts = Counter(all_labels_flat)
        most_frequent = counts.most_common(1)[0]
        results["num_labels"] = len(set(all_labels_flat))
        results["most_frequent_label"] = most_frequent[0]
        results["chance_accuracy"] = most_frequent[1] / len(all_labels_flat)

    results_file = base_paths['results_file'].format(method=method)
    print(f"Saving results to {results_file}")
    with open(results_file, "w") as f: json.dump(results, f, indent=4)
    print(f"--- {method.upper()} LEACE Complete ---")


# =========================================================================
# 3. Main Script Logic
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified script for running LEACE and Oracle LEACE.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to use.")
    parser.add_argument("--concept", choices=["pos", "deplab", "sd", "sdm"], required=True, help="Concept to erase.")
    args = parser.parse_args()

    # --- 1. Dynamic File and Parameter Setup ---
    LAYER = 8
    SEED = 42
    np.random.seed(SEED)
    
    # Correctly set up names for file and directory paths
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"

    # Map from CLI arg to the key used *inside* the .pkl file
    concept_internal_key_map = {
        "pos": "pos_tags",
        "deplab": "deplabs",
        "sd": "distance_spectral_padded",
        "sdm": "distance_matrix"
    }
    concept_internal_key = concept_internal_key_map[args.concept]

    # Map from CLI arg to the key used in the *input filename*
    input_file_key_map = {
        'pos': 'pos',
        'deplab': 'deplab',
        'sd': 'distance_spectral_padded',
        'sdm': 'distance_matrix'
    }
    input_file_key = input_file_key_map[args.concept]

    # Define path templates with a placeholder for the method ('leace' or 'oracle')
    base_name = f"{{method}}_{dataset_name_short}_{args.concept}"
    base_paths = {
        'dataset_cli': args.dataset,
        'concept_cli': args.concept,
        'concept_internal_key': concept_internal_key,
        'layer': LAYER,
        'embedding_file': f"Final/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{input_file_key}.pkl",
        'erased_emb_file': f"Final/Embeddings/Erased/{dataset_dir_name}/{base_name}_vec.pkl",
        'eraser_obj_file': f"Final/Eraser_objects/{dataset_dir_name}/eraser_obj_{base_name}.pkl",
        'results_file': f"Final/Results/{dataset_dir_name}/{base_name}_results.json"
    }
    
    # Create all necessary output directories
    for d in [f"Final/Embeddings/Erased/{dataset_dir_name}",
              f"Final/Eraser_objects/{dataset_dir_name}",
              f"Final/Results/{dataset_dir_name}"]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Running Full Erasure Pipeline ---")
    print(f"Dataset: {args.dataset}, Concept: {args.concept}")
    print(f"Loading embeddings from: {base_paths['embedding_file']}\n")

    # --- 2. Data Loading ---
    try:
        with open(base_paths['embedding_file'], "rb") as f: data = pickle.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at '{base_paths['embedding_file']}'")
        print("Please ensure your data generation scripts have run and the file exists with the correct name.")
        return

    # --- 3. Prepare X and Z matrices ---
    print("Preparing word-level features (X) and concept vectors (Z)...")
    X_full = np.vstack([sent["embeddings_by_layer"][LAYER].numpy() for sent in data])
    Z_full = get_concept_vectors(data, args.concept, concept_internal_key)
    print(f"X_full shape: {X_full.shape}, Z_full shape: {Z_full.shape}")

    # --- 4. Run BOTH Erasure Methods ---
    perform_erasure_and_save(X_full, Z_full, data, base_paths, method="leace", seed=SEED)
    perform_erasure_and_save(X_full, Z_full, data, base_paths, method="oracle", seed=SEED)

    print("\n\n--- All operations complete. ---")

if __name__ == "__main__":
    main()