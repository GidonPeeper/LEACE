"""
encode_linear_distances.py

This script processes Universal Dependencies .conllu files (train, validation, test)
and computes GPT-2 embeddings for each word in each sentence. For each sentence,
it also computes the pairwise **linear distance matrix** between words, where
the (i, j) entry is the absolute difference in word positions: |i - j|.

The output for each split is a list of dictionaries, each containing:
    - "embeddings_by_layer": list of [num_words, hidden_dim] tensors (one per GPT-2 layer)
    - "linear_distance_matrix": [num_words, num_words] tensor of linear distances

Embeddings and distance matrices are saved as pickled files for later use in
structural probe experiments or other analyses.
"""

import pickle
import os
from conllu import parse_incr
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
import torch

# ----------------------------------------------------------------------
# Step 1: Parse .conllu into words
# ----------------------------------------------------------------------

def parse_conllu_words(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            for token in tokenlist:
                if isinstance(token['id'], int):  # Skip multi-word tokens
                    words.append(token['form'])
            sentences.append({
                "words": words
            })
    return sentences

# ----------------------------------------------------------------------
# Step 2: Compute pairwise linear distances for each sentence
# ----------------------------------------------------------------------

def compute_linear_distances(num_words):
    """
    Returns: distance matrix [num_words x num_words] where entry (i, j) = |i - j|
    """
    idxs = torch.arange(num_words)
    dist = torch.abs(idxs.unsqueeze(0) - idxs.unsqueeze(1))
    return dist.float()

# ----------------------------------------------------------------------
# Step 3: Tokenize with GPT-2 and align word-level distances
# ----------------------------------------------------------------------

def tokenize_and_align(sentences, tokenizer):
    tokenized_sentences = []
    for sentence in tqdm(sentences, desc="Tokenizing and aligning"):
        words = sentence["words"]
        num_words = len(words)
        linear_distance_matrix = compute_linear_distances(num_words)

        input_ids = []
        attention_mask = []
        word_to_token_positions = []
        current_token_position = 0

        for word in words:
            word_tokens = tokenizer.tokenize(word)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            if not word_ids:
                continue
            token_positions = list(range(current_token_position, current_token_position + len(word_ids)))
            word_to_token_positions.append(token_positions)
            input_ids.extend(word_ids)
            attention_mask.extend([1] * len(word_ids))
            current_token_position += len(word_ids)

        # SKIP sentences with no tokens
        if len(input_ids) == 0:
            continue

        tokenized_sentences.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_to_token_positions": word_to_token_positions,
            "linear_distance_matrix": linear_distance_matrix  # [num_words x num_words]
        })

    return tokenized_sentences

# ----------------------------------------------------------------------
# Step 4: Run GPT-2 and average subtoken embeddings per word
# ----------------------------------------------------------------------

def average_subtokens_per_word(hidden_states, word_to_token_positions):
    word_embeddings = []
    for token_idxs in word_to_token_positions:
        vectors = hidden_states[token_idxs]
        avg_vector = vectors.mean(dim=0)
        word_embeddings.append(avg_vector)
    return torch.stack(word_embeddings)

def encode_with_gpt2(tokenized_sentences, model, device):
    all_outputs = []
    for sentence in tqdm(tokenized_sentences, desc="Encoding with GPT-2"):
        input_ids = torch.tensor(sentence["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(sentence["attention_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_layers = outputs.hidden_states  # List[Tensor] of shape [1, seq_len, hidden_dim]

        word_embeddings_by_layer = []
        for layer_tensor in all_layers:
            layer_tensor = layer_tensor.squeeze(0)  # [seq_len, hidden_dim]
            word_embeddings = average_subtokens_per_word(
                layer_tensor,
                sentence["word_to_token_positions"]
            )
            word_embeddings_by_layer.append(word_embeddings.cpu())

        all_outputs.append({
            "embeddings_by_layer": word_embeddings_by_layer,  # List[Tensor] per layer
            "linear_distance_matrix": sentence["linear_distance_matrix"]  # Tensor [num_words x num_words]
        })

    return all_outputs

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # Paths
    train_file = "data/narratives/train_clean.conllu"
    test_file = "data/narratives/test_clean.conllu"
    save_dir = "Distances/Embeddings/Narratives"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Step 1: Parse train/valid/test
    print(f"Parsing {train_file} for training set...")
    train_sentences = parse_conllu_words(train_file)
    print(f"Parsed {len(train_sentences)} training sentences.")

    print(f"Parsing {test_file} for test set...")
    test_sentences = parse_conllu_words(test_file)
    print(f"Parsed {len(test_sentences)} test sentences.")

    # Step 2–3: Tokenize and align
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = tokenize_and_align(train_sentences, tokenizer)
    tokenized_test = tokenize_and_align(test_sentences, tokenizer)

    # Step 4: Encode
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    train_embeddings = encode_with_gpt2(tokenized_train, model, device)
    test_embeddings = encode_with_gpt2(tokenized_test, model, device)

    # Save
    with open(os.path.join(save_dir, "gpt2_embeddings_train_linear_dist.pt"), "wb") as f:
        pickle.dump(train_embeddings, f)
    with open(os.path.join(save_dir, "gpt2_embeddings_test_linear_dist.pt"), "wb") as f:
        pickle.dump(test_embeddings, f)
    print(f"Saved embeddings with linear distances to {save_dir}")

if __name__ == "__main__":
    main()