"""
Unified script for linear probing, corrected to use Cosine Similarity for the
high-dimensional Syntactic Distance (sd) regression task.

This version saves the final summary tables to a JSON file for easy access
and reproducibility, in addition to printing them to the console.
"""
import argparse, pickle, numpy as np, os, json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ... (Probing models and train_and_evaluate are unchanged) ...
class LinearProbe(nn.Module):
    def __init__(self, i, o): super().__init__(); self.linear = nn.Linear(i,o)
    def forward(self, x): return self.linear(x)
class RegressionProbe(nn.Module):
    def __init__(self, i, o): super().__init__(); self.linear = nn.Linear(i,o)
    def forward(self, x): return self.linear(x)
def cosine_similarity_score(y_true, y_pred):
    y_true = y_true.cpu().detach(); y_pred = y_pred.cpu().detach()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(y_pred, y_true).mean().item()
def train_and_evaluate(probe, X_train, y_train, X_test, y_test, task_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long() if task_type == 'classification' else torch.from_numpy(y_train).float()
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    probe.train()
    for epoch in range(15):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(); outputs = probe(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward(); optimizer.step()
    probe.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).float()
        predictions = probe(X_test_t)
        if task_type == 'classification':
            return accuracy_score(y_test_t.numpy(), torch.argmax(predictions.cpu(), dim=1).numpy())
        else:
            return cosine_similarity_score(y_test_t, predictions)

# =========================================================================
# 2. Main Probing Orchestration
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified script for probing erased embeddings.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to use.")
    args = parser.parse_args()

    # --- Setup ---
    LAYER = 8; SEED = 42
    dataset_name = "nar" if args.dataset == "narratives" else "ud"
    concepts_to_probe = ["pos", "deplab", "sd"]
    embedding_types = ["original", "pos", "deplab", "sd"]
    
    # This map must be in sync with your data generation and erasure scripts
    concept_key_map = {
        "pos": "pos_tags",
        "deplab": "deplabs",
        "sd": "distance_spectral_padded"
    }
    
    # =====================================================================
    # === NEW: A dictionary to hold results for all methods ===
    # =====================================================================
    all_methods_results = {}

    for method in ["leace", "oracle"]:
        print(f"\n\n{'='*60}\nEVALUATING EMBEDDINGS ERASED WITH: {method.upper()}\n{'='*60}\n")
        results_table = {concept: {} for concept in concepts_to_probe}
        chance_scores = {}

        # ... (The main probing loop is unchanged) ...
        for concept_to_probe in concepts_to_probe:
            print(f"--- Probing for concept: {concept_to_probe.upper()} ---")
            concept_key = concept_key_map[concept_to_probe]
            ground_truth_path = f"Final/Embeddings/Original/{args.dataset.capitalize()}/Embed_{dataset_name}_{concept_key}.pkl"
            try:
                with open(ground_truth_path, "rb") as f: label_data = pickle.load(f)
            except FileNotFoundError:
                print(f"ERROR: Could not find ground-truth file '{ground_truth_path}'. Skipping.")
                continue
            if concept_to_probe in ["pos", "deplab"]:
                task_type, all_labels_flat = 'classification', [lab for sent in label_data for lab in sent.get(concept_key, [])]
                label_to_idx = {label: i for i, label in enumerate(sorted(list(set(all_labels_flat))))}
                y_full = np.array([label_to_idx[val] for val in all_labels_flat], dtype=int)
                counts = Counter(all_labels_flat); chance_scores[concept_to_probe] = counts.most_common(1)[0][1] / len(all_labels_flat)
            else:
                task_type = 'regression'; y_full = np.vstack([sent[concept_key].cpu().numpy() for sent in label_data])
            for emb_type in embedding_types:
                print(f"  > Probing on '{emb_type}' embeddings...")
                if emb_type == "original": emb_data = label_data
                else:
                    emb_path = f"Final/Embeddings/Erased/{args.dataset.capitalize()}/{method}_{dataset_name}_{emb_type}_vec.pkl"
                    try:
                        with open(emb_path, "rb") as f: emb_data = pickle.load(f)
                    except FileNotFoundError:
                        print(f"    - WARNING: Erased file not found at '{emb_path}'. Skipping.")
                        results_table[concept_to_probe][emb_type] = "N/A"; continue
                X_full = np.vstack(emb_data['all_erased']) if 'all_erased' in emb_data else np.vstack([s["embeddings_by_layer"][LAYER].numpy() for s in emb_data])
                if X_full.shape[0] != y_full.shape[0]:
                    print(f"    - FATAL ERROR: Mismatch between X ({X_full.shape[0]}) and Y ({y_full.shape[0]}). Skipping."); continue
                X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
                if task_type == 'classification': probe = LinearProbe(X_train.shape[1], len(label_to_idx))
                else: probe = RegressionProbe(X_train.shape[1], y_train.shape[1])
                score = train_and_evaluate(probe, X_train, y_train, X_test, y_test, task_type)
                results_table[concept_to_probe][emb_type] = score
        
        # --- Store results for this method ---
        method_results = print_and_save_results(method, results_table, chance_scores)
        all_methods_results[method] = method_results

    # =====================================================================
    # === NEW: Save all results to a single JSON file at the end ===
    # =====================================================================
    results_dir = f"Final/Results/{args.dataset.capitalize()}"
    os.makedirs(results_dir, exist_ok=True)
    save_path = f"{results_dir}/probing_summary_results.json"
    
    print(f"\n\nSaving complete summary of all probing results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(all_methods_results, f, indent=4)
    print("Done.")

# =========================================================================
# 3. Updated Results Function
# =========================================================================
def print_and_save_results(method, results_table, chance_scores):
    """Formats, prints, and returns the results tables with normalized RDI."""
    print(f"\n\n--- FINAL RESULTS for Method: {method.upper()} ---")
    
    output_data = {
        "performance": {"metric_note": "Accuracy for pos/deplab, Cosine Similarity for sd"},
        "rdi": {"metric_note": "Using normalized RDI formula: 1 - (Score_after' / Score_before')"}
    }

    # --- Performance Table ---
    print("\nTable 1: Probe Performance (Acc / Cosine Sim)")
    header = f"{'Concept':<10} | {'Original':>12} | {'POS-Erased':>14} | {'Deplab-Erased':>17} | {'SD-Erased':>14}"
    print(header); print("-" * len(header))
    for concept, scores in results_table.items():
        f = lambda s: f"{s:.4f}" if isinstance(s, (int, float)) else s
        line = (f"{concept:<10} | {f(scores.get('original')):>12} | {f(scores.get('pos')):>14} | "
                f"{f(scores.get('deplab')):>17} | {f(scores.get('sd')):>14}")
        print(line)
        output_data["performance"][concept] = {k: f(v) for k, v in scores.items()}
    
    pos_chance = chance_scores.get('pos', 'N/A')
    dep_chance = chance_scores.get('deplab', 'N/A')
    f = lambda s: f"{s:.4f}" if isinstance(s, (int, float)) else s
    print(f"{'Chance Acc':<10} | {f(pos_chance):>12} | {'-':>14} | {f(dep_chance):>17} | {'-':>14}")
    output_data["performance"]["chance_accuracy"] = {"pos": f(pos_chance), "deplab": f(dep_chance)}

    # --- RDI Table with NORMALIZED formulas ---
    print("\nTable 2: Relative Drop in Information (Normalized RDI)")
    header = f"{'Concept':<10} | {'vs POS-Erased':>15} | {'vs Deplab-Erased':>18} | {'vs SD-Erased':>15}"
    print(header); print("-" * len(header))
    for concept, scores in results_table.items():
        score_before = scores.get('original')
        if not isinstance(score_before, float): continue
        rdi_scores = {}
        for emb_type in ['pos', 'deplab', 'sd']:
            score_after = scores.get(emb_type)
            if not isinstance(score_after, float):
                rdi_scores[emb_type] = "N/A"; continue
            
            if concept in ['pos', 'deplab']:
                chance = chance_scores.get(concept, 0.0)
                score_before_norm = score_before - chance
                score_after_norm = score_after - chance
                # Your formula: 1 - (after' / before')
                rdi = 1.0 - (score_after_norm / score_before_norm) if abs(score_before_norm) > 1e-9 else 0.0
            else: # sd
                # Your formula: 1 - (after / before)
                rdi = 1.0 - (score_after / score_before) if abs(score_before) > 1e-9 else 0.0
            
            rdi_scores[emb_type] = rdi
        
        f = lambda s: f"{s:.4f}" if isinstance(s, (int, float)) else s
        print(f"{concept:<10} | {f(rdi_scores.get('pos')):>15} | {f(rdi_scores.get('deplab')):>18} | {f(rdi_scores.get('sd')):>15}")
        output_data["rdi"][concept] = {k: f(v) for k, v in rdi_scores.items()}
    print("\n" * 2)

    return output_data

if __name__ == "__main__":
    main()