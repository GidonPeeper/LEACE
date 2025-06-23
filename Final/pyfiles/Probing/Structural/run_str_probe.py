"""
Runs a non-linear structural probe analysis on original and erased embeddings.
... (docstring) ...
"""
import argparse
import pickle
import numpy as np
import os
import pandas as pd
import subprocess
import tempfile
import yaml
import h5py
from conllu import parse_incr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configuration
PROBE_SCRIPT_PATH = 'external/structural_probes/structural_probe/run_experiment.py'
PROBE_TIMEOUT_SECONDS = 900

# Universal Data Filtering
def get_golden_sentences(dataset):
    # ... (this function is correct and unchanged) ...
    print("[Phase 1/4] Constructing the golden set of sentences...")
    if dataset == "narratives":
        conllu_files = ["data/narratives/train_clean.conllu", "data/narratives/test_clean.conllu"]
    else: # "ud"
        conllu_files = ["data/ud_ewt/en_ewt-ud-train.conllu", "data/ud_ewt/en_ewt-ud-dev.conllu", "data/ud_ewt/en_ewt-ud-test.conllu"]
    
    all_token_lists = []
    for file_path in conllu_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_token_lists.extend(list(parse_incr(f)))
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")

    golden_set = []
    for tl in all_token_lists:
        words = [t for t in tl if isinstance(t['id'], int)]
        heads = [t['head'] for t in words]
        if len(words) > 1:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(len(words)))
            for i, h in enumerate(heads):
                if h > 0: G.add_edge(i, h - 1)
            if nx.is_connected(G):
                golden_set.append(tl)
    
    print(f"Golden set contains {len(golden_set)} sentences.")
    return golden_set

# Probe Interfacing and Execution
def prepare_and_run_probe(embeddings, golden_sentences):
    # ... (this function is correct and unchanged) ...
    if not embeddings or not golden_sentences: return np.nan
    embedding_dim = embeddings[0].shape[1]
    scaler = StandardScaler()
    sent_lengths = [e.shape[0] for e in embeddings]
    flat_embeddings = np.vstack(embeddings)
    scaled_flat_embeddings = scaler.fit_transform(flat_embeddings)
    scaled_embeddings = []
    current_pos = 0
    for length in sent_lengths:
        scaled_embeddings.append(scaled_flat_embeddings[current_pos:current_pos + length])
        current_pos += length

    with tempfile.TemporaryDirectory() as temp_dir:
        matched_indices = [i for i, emb in enumerate(scaled_embeddings) if emb.shape[0] == len([t for t in golden_sentences[i] if isinstance(t['id'], int)])]

        hdf5_path = os.path.join(temp_dir, 'embeddings.hdf5')
        with h5py.File(hdf5_path, 'w') as f:
            for new_idx, old_idx in enumerate(matched_indices):
                f.create_dataset(str(new_idx), data=scaled_embeddings[old_idx])

        conllu_path = os.path.join(temp_dir, 'corpus.conllu')
        with open(conllu_path, 'w', encoding='utf-8') as f:
            for old_idx in matched_indices:
                token_list_copy = golden_sentences[old_idx].copy()
                for token in token_list_copy:
                    if 'xpos' not in token or not token['xpos']:
                        token['xpos'] = token['upos']
                f.write(token_list_copy.serialize())
        
        if not matched_indices:
            print("\nWarning: No sentences with matching embedding lengths found.")
            return np.nan

        config = {
            'dataset': {'observation_fieldnames': ['index', 'sentence', 'lemma_sentence', 'upos_sentence', 'xpos_sentence', 'morph', 'head_indices', 'governance_relations', 'secondary_relations', 'extra_info', 'embeddings'],
                        'corpus': {'root': temp_dir, 'train_path': 'corpus.conllu', 'dev_path': 'corpus.conllu'},
                        'embeddings': {'root': temp_dir, 'train_path': 'embeddings.hdf5', 'dev_path': 'embeddings.hdf5'}, 'batch_size': 20},
            'model': {'hidden_dim': embedding_dim, 'model_type': 'ELMo-disk', 'model_layer': 0},
            'probe': {'task_name': 'parse-distance', 'maximum_rank': 128, 'psd_parameters': True},
            'probe_training': {'epochs': 20, 'loss': 'L1'},
            'reporting': {'root': temp_dir, 'reporting_methods': ['spearmanr']}
        }
        yaml_path = os.path.join(temp_dir, 'probe_config.yaml')
        with open(yaml_path, 'w') as f: yaml.dump(config, f)
        
        try:
            subprocess.run(['python', PROBE_SCRIPT_PATH, yaml_path],
                           check=True, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS)
        except FileNotFoundError:
            print(f"\nFATAL ERROR: Could not find the probe script at '{PROBE_SCRIPT_PATH}'. Please check the path.")
            exit(1)
        except subprocess.TimeoutExpired:
            print(f"\nERROR: Structural probe subprocess timed out ({PROBE_TIMEOUT_SECONDS}s).")
            return np.nan
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: Structural probe subprocess failed with exit code {e.returncode}.")
            print("STDERR (last 500 chars):\n", e.stderr[-500:])
            return np.nan

        result_file = os.path.join(temp_dir, 'dev.spearmanr-5-50-mean')
        try:
            with open(result_file, 'r') as f: score = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            print(f"\nWarning: Could not find or parse result file {result_file}")
            return np.nan

    return score

# Main Script Logic
def load_embeddings_from_pkl(file_path, erased=False):
    """Helper to load embeddings from a pickle file into a list of numpy arrays."""
    try:
        with open(file_path, "rb") as f: data = pickle.load(f)
        if erased:
            # Erased files contain a list of numpy arrays
            return data["all_erased"]
        else:
            # Original files contain a list of dicts with torch tensors
            return [sent['embeddings_by_layer'][8].numpy() for sent in data]
    except FileNotFoundError:
        print(f"Warning: Could not load embeddings from {file_path}")
        return None

def main():
    # ... (main function is correct and unchanged) ...
    parser = argparse.ArgumentParser(description="Run a non-linear structural probe analysis.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="Dataset to probe.")
    parser.add_argument("--method", choices=["leace", "oracle"], required=True, help="Erasure method to probe.")
    args = parser.parse_args()

    CONCEPTS = ["pos", "deplab", "sd", "sdm"]
    dataset_dir_name = "Narratives" if args.dataset == "narratives" else "UD"
    dataset_name_short = "nar" if args.dataset == "narratives" else "ud"
    
    raw_scores = pd.DataFrame(index=['parse-distance'], columns=CONCEPTS + ['original'], dtype=float)
    rdi_scores = pd.DataFrame(index=['parse-distance'], columns=CONCEPTS, dtype=float)
    
    golden_sentences = get_golden_sentences(args.dataset)

    print("\n[Phase 2/4] Calculating baseline performance on original embeddings...")
    file_key = "distance_matrix" 
    file_path = f"Final/Embeddings/Original/{dataset_dir_name}/Embed_{dataset_name_short}_{file_key}.pkl"
    original_embeddings = load_embeddings_from_pkl(file_path)
    baseline_score = prepare_and_run_probe(original_embeddings, golden_sentences)
    raw_scores.loc['parse-distance', 'original'] = baseline_score
    print(f"Baseline SpearmanR on original (sdm) embeddings: {baseline_score:.4f}")

    print("\n[Phase 3/4] Calculating performance on ERASED embeddings...")
    for erased_concept in tqdm(CONCEPTS, desc="Probing erased"):
        file_path = f"Final/Embeddings/Erased/{dataset_dir_name}/{args.method}_{dataset_name_short}_{erased_concept}_vec.pkl"
        erased_embeddings = load_embeddings_from_pkl(file_path, erased=True)
        
        score = prepare_and_run_probe(erased_embeddings, golden_sentences)
        raw_scores.loc['parse-distance', erased_concept] = score

    print("\n[Phase 4/4] Calculating RDI scores...")
    for erased_concept in CONCEPTS:
        p_after = raw_scores.loc['parse-distance', erased_concept]
        p_before = baseline_score
        
        if pd.isna(p_after) or p_before is None or pd.isna(p_before) or p_before < 1e-6:
            rdi = np.nan
        else:
            rdi = max(0, 1.0 - (p_after / p_before))
        rdi_scores.loc['parse-distance', erased_concept] = rdi

    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n\n" + "="*80)
    print(f"STRUCTURAL PROBE RESULTS for Dataset: {args.dataset.upper()}, Method: {args.method.upper()}")
    print("="*80)
    
    print("\n--- Raw Probe Performance (Spearman Correlation for Parse Distance) ---")
    print("Columns: Concept Erased From Embeddings (+ Original Baseline)\n")
    print(raw_scores)
    
    print("\n\n--- RDI Scores from Structural Probe ---")
    print("1.0 = Full Erasure, 0.0 = No Erasure\n")
    print(rdi_scores)
    print("\n" + "="*80)
    
    results_dir = f"Final/Results/{dataset_dir_name}"
    raw_scores.to_csv(f"{results_dir}/structural_probe_raw_perf_{args.dataset}_{args.method}.csv")
    rdi_scores.to_csv(f"{results_dir}/structural_probe_rdi_scores_{args.dataset}_{args.method}.csv")
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()