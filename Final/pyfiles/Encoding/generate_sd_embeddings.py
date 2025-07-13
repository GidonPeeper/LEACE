"""
Generates GPT-2 embeddings with aligned SYNTACTIC DISTANCE (sd) vectors.

This method represents each word's structural position by taking its row from
the all-pairs shortest path distance matrix and padding it to a fixed length
(the max sentence length in the dataset).

It uses the same universal filtering and file naming conventions as the
other generation scripts to ensure perfect pipeline compatibility.
"""
import argparse
import pickle
import os
import torch
import numpy as np
import networkx as nx
from conllu import parse_incr
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm

# ======================================================================
# Universal Data Loading and Filtering Pipeline (Identical to your script)
# ======================================================================

def parse_conllu_universal(file_path):
    # ... (Identical to your reference script) ...
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                words, pos_tags, deplabs, heads = [], [], [], []
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        words.append(token['form']); pos_tags.append(token['upos'])
                        deplabs.append(token['deprel']); heads.append(token['head'])
                if words: sentences.append({"words": words, "pos_tags": pos_tags, "deplabs": deplabs, "heads": heads})
    except FileNotFoundError: print(f"Warning: File not found at {file_path}. Skipping."); return []
    return sentences

def filter_and_validate_sentences(sentences):
    # ... (Identical to your reference script) ...
    print("\n--- Step 2: Filtering Data to Create Golden Set ---")
    initial_count = len(sentences)
    valid_sentences = []
    for s in sentences:
        num_words = len(s['words'])
        if num_words > 1 and (num_words == len(s['pos_tags']) == len(s['deplabs'])) and all(isinstance(h, int) and h >= 0 for h in s['heads']):
            valid_sentences.append(s)
    print(f"Step 2.1: Kept {len(valid_sentences)}/{initial_count} after basic validation.")
    tree_sentences = []
    for s in valid_sentences:
        n = len(s['heads']); G = nx.Graph(); G.add_nodes_from(range(n))
        for i, head in enumerate(s['heads']):
            if head > 0: G.add_edge(i, head - 1)
        if nx.is_connected(G): tree_sentences.append(s)
    print(f"Step 2.2: Kept {len(tree_sentences)}/{len(valid_sentences)} that form a valid dependency tree.")
    print(f"--- Filtering Complete. Final count: {len(tree_sentences)} ---")
    return tree_sentences

# ======================================================================
# Concept Calculation: Padded Syntactic Distance Vectors
# ======================================================================

def add_padded_distance_vectors(sentences):
    """Computes distance matrices, then converts them to padded per-word vectors."""
    print("\n--- Step 3: Computing Padded Syntactic Distance Vectors ---")
    
    # First pass: compute matrices to find the max_len for padding
    max_len = 0
    temp_sentences = []
    for s in tqdm(sentences, desc="Computing matrices"):
        n = len(s['words'])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i, head in enumerate(s['heads']):
            if head > 0: G.add_edge(i, head - 1)
        
        dist_matrix = nx.floyd_warshall_numpy(G, nodelist=range(n))
        s['distance_matrix'] = dist_matrix
        temp_sentences.append(s)
        if n > max_len: max_len = n

    print(f"Max sentence length found: {max_len}. Vectors will be padded to this dimension.")
    
    # Second pass: create the final padded vectors
    final_sentences = []
    for s in tqdm(temp_sentences, desc="Creating padded vectors"):
        dist_matrix = s.pop('distance_matrix') # Remove the temporary matrix
        sent_len = dist_matrix.shape[0]
        padded_vectors = []
        for i in range(sent_len):
            row_vector = dist_matrix[i, :]
            # Pad with 0, which is a neutral value for regression and distance.
            padded_vector = np.pad(row_vector, (0, max_len - sent_len), 'constant', constant_values=0)
            padded_vectors.append(padded_vector)
        
        # The key for this concept will be 'sd'
        s['sd'] = torch.from_numpy(np.array(padded_vectors, dtype=np.float32))
        final_sentences.append(s)
        
    return final_sentences

# ======================================================================
# Generic Tokenization and Encoding Functions
# ======================================================================

def tokenize_and_align(sentences, tokenizer, concept_key):
    # This function is now robust enough to handle the 'sd' tensor
    print("\n--- Step 4: Tokenizing and Aligning Data ---")
    tokenized_sentences = []
    for sentence in tqdm(sentences, desc=f"Tokenizing for '{concept_key}'"):
        words, labels = sentence["words"], sentence[concept_key]
        input_ids, word_to_token_positions, original_word_indices = [], [], []
        current_token_position = 0
        for i, word in enumerate(words):
            word_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if not word_ids: continue
            token_positions = list(range(current_token_position, current_token_position + len(word_ids)))
            word_to_token_positions.append(token_positions); input_ids.extend(word_ids)
            current_token_position += len(word_ids); original_word_indices.append(i)
        if not input_ids: continue
        
        aligned_labels = labels[original_word_indices] # Select rows from the tensor
            
        tokenized_sentences.append({
            "input_ids": input_ids, "word_to_token_positions": word_to_token_positions,
            concept_key: aligned_labels
        })
    return tokenized_sentences

def encode_with_gpt2(tokenized_sentences, model, device, concept_key):
    # This function is identical to the one in your reference script
    print("\n--- Step 5: Encoding with GPT-2 ---")
    all_outputs = []
    for sentence in tqdm(tokenized_sentences, desc="Encoding with GPT-2"):
        input_ids = torch.tensor(sentence["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        word_embeddings_by_layer = []
        for layer_tensor in outputs.hidden_states:
            word_embs = average_subtokens_per_word(layer_tensor.squeeze(0), sentence["word_to_token_positions"])
            if word_embs is not None and len(word_embs) == len(sentence[concept_key]):
                word_embeddings_by_layer.append(word_embs.cpu())
        if word_embeddings_by_layer:
            all_outputs.append({"embeddings_by_layer": word_embeddings_by_layer, concept_key: sentence[concept_key]})
    return all_outputs

def average_subtokens_per_word(hidden_states, word_to_token_positions):
    # This function is identical to the one in your reference script
    word_embeddings = []
    for token_idxs in word_to_token_positions:
        if not token_idxs: continue
        word_embeddings.append(hidden_states[token_idxs].mean(dim=0))
    return torch.stack(word_embeddings) if word_embeddings else None

# ======================================================================
# Main Execution
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with padded syntactic distance vectors.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--data_fraction", type=int, choices=[1, 10, 100], default=100, help="Percentage of the filtered dataset to use for encoding.")
    args = parser.parse_args()

    # Set seed for reproducibility of data subsetting
    np.random.seed(42)

    # The concept key is fixed for this script
    CONCEPT_KEY = "sd"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}, Concept: {CONCEPT_KEY.upper()}, Device: {device}, Data Fraction: {args.data_fraction}%")

    # Identical file loading logic from your reference script
    if args.dataset == "narratives":
        dataset_name_short = "nar"
        data_files = ["data/narratives/train_clean.conllu", "data/narratives/test_clean.conllu"]
        save_dir = "Final/Embeddings/Original/Narratives"
    else: # args.dataset == "ud"
        dataset_name_short = "ud"
        data_files = ["data/ud_ewt/en_ewt-ud-train.conllu", "data/ud_ewt/en_ewt-ud-dev.conllu", "data/ud_ewt/en_ewt-ud-test.conllu"]
        save_dir = "Final/Embeddings/Original/UD"

    os.makedirs(save_dir, exist_ok=True)
    
    print("\n--- Step 1: Parsing and Loading Data ---")
    all_sentences = [s for f in data_files for s in parse_conllu_universal(f)]
    print(f"Loaded a total of {len(all_sentences)} sentences.")

    golden_sentences = filter_and_validate_sentences(all_sentences)
    
    if not golden_sentences:
        print("\nPipeline resulted in zero sentences. Exiting."); return

    # --- SUBSETTING LOGIC ---
    if args.data_fraction < 100:
        print(f"\n--- Subsetting Data to {args.data_fraction}% ---")
        num_samples = int(len(golden_sentences) * (args.data_fraction / 100.0))
        # np.random.seed(42) was set at the start of main for reproducibility
        subset_indices = np.random.choice(len(golden_sentences), num_samples, replace=False)
        golden_sentences = [golden_sentences[i] for i in subset_indices]
        print(f"Using a random subset of {len(golden_sentences)} sentences for encoding.")

    final_sentences = add_padded_distance_vectors(golden_sentences)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenize_and_align(final_sentences, tokenizer, concept_key=CONCEPT_KEY)

    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    all_embeddings = encode_with_gpt2(tokenized_data, model, device, concept_key=CONCEPT_KEY)

    # --- DYNAMIC FILENAME LOGIC ---
    if args.data_fraction == 100:
        file_suffix = ""
    else:
        file_suffix = f"_{args.data_fraction}pct"

    save_path = os.path.join(save_dir, f"Embed_{dataset_name_short}_{CONCEPT_KEY}{file_suffix}.pkl")
    
    with open(save_path, "wb") as f: pickle.dump(all_embeddings, f)
    print(f"\n--- DONE ---\nSaved {len(all_embeddings)} sentences to {save_path}")

if __name__ == "__main__":
    main()