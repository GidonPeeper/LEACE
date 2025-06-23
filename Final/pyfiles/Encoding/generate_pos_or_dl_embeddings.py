"""
Generates GPT-2 embeddings with aligned categorical labels (POS tags or
Dependency Relations).

This script uses the universal parsing and filtering pipeline to ensure perfect
data alignment with other concept embeddings. It is fully command-line driven
for selecting the dataset and the specific concept to embed.
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
# Universal Data Loading and Filtering Pipeline (Now Fully Consistent)
# ======================================================================

def parse_conllu_universal(file_path):
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                words, pos_tags, deplabs, heads = [], [], [], []
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        words.append(token['form'])
                        pos_tags.append(token['upos'])
                        deplabs.append(token['deprel'])
                        heads.append(token['head'])
                if words:
                    sentences.append({"words": words, "pos_tags": pos_tags, "deplabs": deplabs, "heads": heads})
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
        return []
    return sentences

def filter_and_validate_sentences(sentences):
    print("\n--- Step 2: Filtering Data to Create Golden Set ---")
    initial_count = len(sentences)
    
    # *** CRITICAL FIX FOR CONSISTENCY ***
    # Changed num_words > 0 to num_words > 1 to match other scripts.
    # This guarantees the "golden set" of sentences is identical for all concepts.
    valid_sentences = []
    for s in sentences:
        num_words = len(s['words'])
        if num_words > 1 and (num_words == len(s['pos_tags']) == len(s['deplabs'])) and all(isinstance(h, int) and h >= 0 for h in s['heads']):
            valid_sentences.append(s)
    print(f"Step 2.1: Kept {len(valid_sentences)}/{initial_count} after basic validation (non-empty, >1 word, aligned, valid heads).")
    
    tree_sentences = []
    for s in valid_sentences:
        n = len(s['heads'])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i, head in enumerate(s['heads']):
            if head > 0: G.add_edge(i, head - 1)
        if nx.is_connected(G):
            tree_sentences.append(s)
    print(f"Step 2.2: Kept {len(tree_sentences)}/{len(valid_sentences)} that form a valid dependency tree.")
    print(f"--- Filtering Complete. Final count: {len(tree_sentences)} ---")
    return tree_sentences

# ======================================================================
# Generic Tokenization and Encoding Functions
# ======================================================================

def tokenize_and_align(sentences, tokenizer, concept_key):
    print("\n--- Step 3: Tokenizing and Aligning Data ---")
    tokenized_sentences = []
    for sentence in tqdm(sentences, desc=f"Tokenizing for '{concept_key}'"):
        words, labels = sentence["words"], sentence[concept_key]
        input_ids, word_to_token_positions, original_word_indices = [], [], []
        current_token_position = 0
        for i, word in enumerate(words):
            word_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if not word_ids: continue
            token_positions = list(range(current_token_position, current_token_position + len(word_ids)))
            word_to_token_positions.append(token_positions)
            input_ids.extend(word_ids)
            current_token_position += len(word_ids)
            original_word_indices.append(i)
        if not input_ids: continue

        # This list comprehension is correct for categorical (string) labels
        aligned_labels = [labels[i] for i in original_word_indices]
        
        tokenized_sentences.append({
            "input_ids": input_ids, "word_to_token_positions": word_to_token_positions,
            concept_key: aligned_labels
        })
    return tokenized_sentences

def encode_with_gpt2(tokenized_sentences, model, device, concept_key):
    print("\n--- Step 4: Encoding with GPT-2 ---")
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
    word_embeddings = []
    for token_idxs in word_to_token_positions:
        if not token_idxs: continue
        word_embeddings.append(hidden_states[token_idxs].mean(dim=0))
    return torch.stack(word_embeddings) if word_embeddings else None

# ======================================================================
# Main Execution
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate GPT-2 embeddings with aligned POS or Deplab labels.")
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True, help="The dataset to process.")
    parser.add_argument("--concept", choices=["pos", "deplab"], required=True, help="The concept to align (pos or deplab).")
    args = parser.parse_args()

    # This map translates the user-friendly CLI argument to the actual key in the parsed data.
    # It also provides the key to use in the output filename.
    concept_map = {
        "pos": {"internal_key": "pos_tags", "file_key": "pos"},
        "deplab": {"internal_key": "deplabs", "file_key": "deplab"}
    }
    internal_concept_key = concept_map[args.concept]["internal_key"]
    file_concept_key = concept_map[args.concept]["file_key"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}, Concept: {args.concept} (using data key '{internal_concept_key}'), Device: {device}")

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
        print("\nPipeline resulted in zero sentences. Cannot proceed. Exiting.")
        return

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenize_and_align(golden_sentences, tokenizer, concept_key=internal_concept_key)

    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    all_embeddings = encode_with_gpt2(tokenized_data, model, device, concept_key=internal_concept_key)

    # Use the short, user-friendly key for the filename for compatibility with the erasure script.
    save_path = os.path.join(save_dir, f"Embed_{dataset_name_short}_{file_concept_key}.pkl")
    with open(save_path, "wb") as f: pickle.dump(all_embeddings, f)
    print(f"\n--- DONE ---\nSaved {len(all_embeddings)} sentences to {save_path}")

if __name__ == "__main__":
    main()