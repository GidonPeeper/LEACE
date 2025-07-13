"""
A unified script to calculate L2 distances for all erasure experiments,
using the most methodologically sound approach for each erasure type.

- For 'oracle' (in-sample), L2 is calculated on the full dataset.
- For 'leace' and 'regressout' (generalization), L2 is calculated
  exclusively on the held-out test set (20% of the data).
"""
import argparse
import pickle
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Calculate L2 distances for all erasure experiments.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to analyze.")
    parser.add_argument("--data_fraction", type=int, choices=[1, 10, 100], default=100, help="Percentage of the dataset to use, corresponding to the pre-generated embedding files.")
    args = parser.parse_args()

    # --- Setup ---
    SEED = 42
    # --- CHANGE: 'pe' removed from list of concepts ---
    CONCEPTS = ["pos", "deplab", "ld", "sd"]
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    dataset_dir_name_base = "UD" if args.dataset == "ud" else "Narratives"
    base_dir = "Final"

    # Create suffix for data fraction to append to filenames
    file_suffix = f"_{args.data_fraction}pct" if args.data_fraction < 100 else ""

    conditions = [
        {"method": "oracle", "scaling": "off"},
        {"method": "leace", "scaling": "off"},
        #{"method": "leace", "scaling": "on"},
        {"method": "regressout", "scaling": "off"},
    ]
    
    results_df = pd.DataFrame(index=CONCEPTS, columns=[f"{c['method']}_{c['scaling']}" for c in conditions], dtype=float)

    print(f"\n--- Calculating L2 Distances for Dataset: {args.dataset.upper()} (Fraction: {args.data_fraction}%) ---")

    for concept in tqdm(CONCEPTS, desc="Processing Concepts"):
        # --- 1. Load the Original Full Dataset (X_full) ---
        # --- CHANGE: 'pe' removed from concept map ---
        concept_map = { 
            "pos": "pos", "deplab": "deplab", "sd": "sd", "ld": "ld"
        }
        fkey = concept_map[concept]
        orig_path = f"{base_dir}/Embeddings/Original/{dataset_dir_name_base}/Embed_{dataset_name_short}_{fkey}{file_suffix}.pkl"
        
        try:
            with open(orig_path, "rb") as f: data = pickle.load(f)
            X_full = np.vstack([s["embeddings_by_layer"][8].numpy() for s in data])
        except FileNotFoundError:
            print(f"  - WARNING: Original file for '{concept}' not found at {orig_path}. Skipping."); continue

        # --- 2. Isolate the test set indices for leace/regressout ---
        indices = np.arange(X_full.shape[0])
        _, test_indices = train_test_split(indices, test_size=0.2, random_state=SEED)
        X_test_original = X_full[test_indices]

        # --- 3. Loop through each experimental condition ---
        for cond in conditions:
            method, scaling_str = cond["method"], cond["scaling"]
            col_name = f"{method}_{scaling_str}"
            
            scaling_suffix = "_Scaled" if scaling_str == "on" else "_Unscaled"
            erased_dir_name = dataset_dir_name_base + scaling_suffix
            erased_path = f"{base_dir}/Embeddings/Erased/{erased_dir_name}/{method}_{dataset_name_short}_{concept}{file_suffix}.pkl"
            
            try:
                with open(erased_path, "rb") as f: X_full_erased = pickle.load(f)
                if not isinstance(X_full_erased, np.ndarray): raise TypeError("Not a numpy array")

                # --- 4. Select the correct subset for L2 calculation ---
                if method == "oracle":
                    # For Oracle, compare the full original to the full erased
                    X_orig_subset = X_full
                    X_erased_subset = X_full_erased
                    note = "on full dataset"
                else: # leace and regressout
                    # For generalization methods, compare the original test set
                    # to the corresponding slice of the erased data.
                    X_orig_subset = X_test_original
                    X_erased_subset = X_full_erased[test_indices]
                    note = "on held-out test set"

                if X_orig_subset.shape != X_erased_subset.shape:
                    raise ValueError("Shape mismatch between original and erased subsets.")

                l2_distance = np.linalg.norm(X_orig_subset - X_erased_subset, axis=1).mean()
                results_df.loc[concept, col_name] = l2_distance
                
            except (FileNotFoundError, TypeError, ValueError) as e:
                results_df.loc[concept, col_name] = np.nan
                continue
                
    # --- 5. Display and Save the Final Table ---
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"FINAL L2 DISTANCE SUMMARY for Dataset: {args.dataset.upper()} (Fraction: {args.data_fraction}%)")
    print("Scores represent avg geometric distance. Oracle is on full data; others on test set.")
    print("="*80 + "\n")
    print(results_df)
    
    results_dir = f"{base_dir}/Results/{dataset_dir_name_base}"
    os.makedirs(results_dir, exist_ok=True)
    save_path = f"{results_dir}/l2_distance_summary_{args.dataset}{file_suffix}.csv"
    results_df.to_csv(save_path)
    print(f"\nSummary table saved to: {save_path}")

if __name__ == "__main__":
    main()