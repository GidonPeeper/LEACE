"""
Generates GPT-2 embeddings with aligned POSITIONAL information.

This script can generate two types of positional concepts, controlled by the
--positional_type argument:
1.  'ld': Linear Distance (simple token index).
2.  'pe': Positional Encoding (sinusoidal vectors).

It uses the same universal filtering and file naming conventions as the
categorical generation script to ensure perfect pipeline compatibility.
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
# Universal Data Loading and Filtering Pipeline 
# ======================================================================

def parse_conllu_universal(file_path):
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                words, pos_tags, deplabs, heads = [], [], [], []
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        words.append(token['form']); pos_tags.append(token['upos'])
                        deplabs.append(token['deprel']); heads.append(token['head'])
                if words:
                    sentences.append({"words": words, "pos_tags": pos_tags, "deplabs": deplabs, "heads": heads})
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping."); return []
    return sentences

def filter_and_validate_sentences(sentences):
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
# Positional Concept Calculation
# ======================================================================

def get_positional_encoding(max_len, d_model):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def add_positional_concepts(sentences, positional_type, pe_dim=128):
    if positional_type == 'pe':
        max_len = max(len(s['words']) for s in sentences)
        pe_matrix = get_positional_encoding(max_len, pe_dim)
        print(f"\nGenerated sinusoidal positional encodings up to length {max_len} with dim={pe_dim}.")

    for sentence in sentences:
        n_words = len(sentence['words'])
        if positional_type == 'ld':
            sentence['ld'] = [float(i) for i in range(n_words)]
        elif positional_type == 'pe':
            sentence['pe'] = pe_matrix[:n_words, :]
    return sentences

# ======================================================================
# Generic Tokenization and Encoding Functions (Adapted for positional data)
# ======================================================================

def tokenize_and_align(sentences, tokenizer, concept_key):
    print("\n--- Step 3: Tokenizing and Aligning Data ---")
    tokenized_sentences = []
    for sentence in tqdm(sentences, desc=f"Tokenizing for '{concept_key}'"):
        words = sentence["words"]; labels = sentence[concept_key]
        input_ids, word_to_token_positions, original_word_indices = [], [], []
        current_token_position = 0
        for i, word in enumerate(words):
            word_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if not word_ids: continue
            token_positions = list(range(current_token_position, current_token_position + len(word_ids)))
            word_to_token_positions.append(token_positions); input_ids.extend(word_ids)
            current_token_position += len(word_ids); original_word_indices.append(i)
        if not input_ids: continue
        
        # This logic handles both lists of scalars (ld) and tensors of vectors (pe)
        if isinstance(labels, list):
            aligned_labels = [labels[i] for i in original_word_indices]
        else: # It's a tensor
            aligned_labels = labels[original_word_indices]
            
        tokenized_sentences.append({
            "input_ids": input_ids, "word_to_token_positions": word_to_token_positions,
            concept_key: aligned_labels
        })
    return tokenized_sentences

def encode_with_gpt2(tokenized_sentences, model, device, concept_key):
    print("\n--- Step 4: Encoding with GPT-2 ---")
    # ... (This function is identical to the one in your script) ...
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
    # ... (This function is identical to the one in your script) ...
    word_embeddings = []
    for token_idxs in word_to_token_positions:
        if not token_idxs: continue
        word_embeddings.append(hidden_states[token_idxs].mean(dim=0))
    return torch.stack(word_embeddings) if word_embeddings else None

# ======================================================================
# Main Execution (Mirrors your provided script's structure)
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with positional concepts.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--positional_type", choices=["ld", "pe"], required=True, help="Type of positional concept to generate.")
    parser.add_argument("--data_fraction", type=int, choices=[1, 10, 100], default=100, help="Percentage of the filtered dataset to use for encoding.")
    args = parser.parse_args()

    # Set seed for reproducibility of data subsetting
    np.random.seed(42)

    # The concept_map now translates the new --positional_type argument
    concept_map = {
        "ld": {"internal_key": "ld", "file_key": "ld"},
        "pe": {"internal_key": "pe", "file_key": "pe"}
    }
    internal_concept_key = concept_map[args.positional_type]["internal_key"]
    file_concept_key = concept_map[args.positional_type]["file_key"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}, Concept: {args.positional_type}, Device: {device}, Data Fraction: {args.data_fraction}%")

    # Identical file loading logic
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


    final_sentences = add_positional_concepts(golden_sentences, positional_type=internal_concept_key)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenize_and_align(final_sentences, tokenizer, concept_key=internal_concept_key)

    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    all_embeddings = encode_with_gpt2(tokenized_data, model, device, concept_key=internal_concept_key)

    # --- DYNAMIC FILENAME LOGIC ---
    if args.data_fraction == 100:
        file_suffix = ""
    else:
        file_suffix = f"_{args.data_fraction}pct"
    
    save_path = os.path.join(save_dir, f"Embed_{dataset_name_short}_{file_concept_key}{file_suffix}.pkl")
    
    with open(save_path, "wb") as f: pickle.dump(all_embeddings, f)
    print(f"\n--- DONE ---\nSaved {len(all_embeddings)} sentences to {save_path}")

if __name__ == "__main__":
    main()