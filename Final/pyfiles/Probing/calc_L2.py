"""
A unified script to calculate L2 distances for all erasure experiments.

This script standardizes the L2 score calculation by always comparing the
original full dataset against the corresponding erased full dataset. This ensures
a fair "apples-to-apples" comparison of the "cost" or "damage" inflicted by
each erasure method and scaling condition.

It will generate a single summary table showing the L2 distance for:
- oracle (unscaled)
- leace (unscaled)
- leace (scaled)
- regressout (unscaled)
"""
import argparse
import pickle
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Calculate L2 distances for all erasure experiments.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to analyze.")
    args = parser.parse_args()

    # --- Setup ---
    CONCEPTS = ["pos", "deplab", "ld", "sd", "pe"]
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    dataset_dir_name_base = "UD" if args.dataset == "ud" else "Narratives"
    base_dir = "Final"

    # Define all the experimental conditions we want to test
    conditions = [
        {"method": "oracle", "scaling": "off"},
        {"method": "leace", "scaling": "off"},
        {"method": "leace", "scaling": "on"},
        {"method": "regressout", "scaling": "off"},
    ]
    
    # Create a DataFrame to store all results
    results_df = pd.DataFrame(index=CONCEPTS, columns=[f"{c['method']}_{c['scaling']}" for c in conditions], dtype=float)

    print(f"\n--- Calculating L2 Distances for Dataset: {args.dataset.upper()} ---")
    print("[Note: All scores are calculated between the original and erased full datasets.]\n")

    for concept in tqdm(CONCEPTS, desc="Processing Concepts"):
        # --- 1. Load the Original Full Dataset (X_full) ---
        # This only needs to be done once per concept
        concept_map = { 
            "pos": "pos", "deplab": "deplab", "sd": "sd", "ld": "ld", "pe": "pe" 
        }
        fkey = concept_map[concept]
        orig_path = f"{base_dir}/Embeddings/Original/{dataset_dir_name_base}/Embed_{dataset_name_short}_{fkey}.pkl"
        
        try:
            with open(orig_path, "rb") as f:
                data = pickle.load(f)
            # We assume Layer 8 for consistency
            X_full = np.vstack([s["embeddings_by_layer"][8].numpy() for s in data])
        except FileNotFoundError:
            print(f"  - WARNING: Original file for '{concept}' not found at {orig_path}. Skipping concept.")
            continue

        # --- 2. Loop through each experimental condition ---
        for cond in conditions:
            method = cond["method"]
            scaling_str = cond["scaling"]
            
            # Construct the path to the erased file
            scaling_suffix = "_Scaled" if scaling_str == "on" else "_Unscaled"
            erased_dir_name = dataset_dir_name_base + scaling_suffix
            erased_path = f"{base_dir}/Embeddings/Erased/{erased_dir_name}/{method}_{dataset_name_short}_{concept}.pkl"
            
            try:
                with open(erased_path, "rb") as f:
                    X_full_erased = pickle.load(f)
                
                # Ensure it's a flat numpy array
                if not isinstance(X_full_erased, np.ndarray):
                    raise TypeError("Loaded data is not a numpy array.")
                if X_full_erased.shape != X_full.shape:
                    raise ValueError("Shape mismatch between original and erased embeddings.")
                    
                # --- 3. Calculate L2 Distance ---
                l2_distance = np.linalg.norm(X_full - X_full_erased, axis=1).mean()
                
                # Store it in the DataFrame
                col_name = f"{method}_{scaling_str}"
                results_df.loc[concept, col_name] = l2_distance
                
            except (FileNotFoundError, TypeError, ValueError) as e:
                # print(f"  - INFO: Could not process {erased_path}. Reason: {e}. Skipping.")
                results_df.loc[concept, f"{method}_{scaling_str}"] = np.nan
                continue
                
    # --- 4. Display and Save the Final Table ---
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"FINAL L2 DISTANCE SUMMARY for Dataset: {args.dataset.upper()}")
    print("Scores represent the average geometric distance between original and erased vectors.")
    print("="*80 + "\n")
    print(results_df)
    
    results_dir = f"Final/Results/{dataset_dir_name_base}" # Save to base results dir
    os.makedirs(results_dir, exist_ok=True)
    save_path = f"{results_dir}/l2_distance_summary_{args.dataset}.csv"
    results_df.to_csv(save_path)
    print(f"\nSummary table saved to: {save_path}")

if __name__ == "__main__":
    main()