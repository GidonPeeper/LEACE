import pickle
import os
from conllu import parse_incr
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
import torch
import networkx as nx

# ----------------------------------------------------------------------
# Step 1: Parse .conllu into words and dependency heads
# ----------------------------------------------------------------------

def parse_conllu_with_heads(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            heads = []
            for token in tokenlist:
                if isinstance(token['id'], int):  # Skip multi-word tokens
                    words.append(token['form'])
                    heads.append(token['head'])
            sentences.append({
                "words": words,
                "heads": heads
            })
    return sentences

# ----------------------------------------------------------------------
# Step 2: Compute pairwise tree distances for each sentence
# ----------------------------------------------------------------------

def compute_tree_distances(heads):
    n = len(heads)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, head in enumerate(heads):
        if head is None:
            return None
        if head == 0:
            continue
        G.add_edge(i, head - 1)
    try:
        dist = nx.floyd_warshall_numpy(G, nodelist=range(n))
    except nx.NetworkXError:
        return None
    return torch.tensor(dist, dtype=torch.float32)

# ----------------------------------------------------------------------
# Step 3: Tokenize with GPT-2 and align word-level distances
# ----------------------------------------------------------------------

def tokenize_and_align(sentences, tokenizer):
    tokenized_sentences = []
    for sentence in tqdm(sentences, desc="Tokenizing and aligning"):
        words = sentence["words"]
        heads = sentence["heads"]
        distance_matrix = compute_tree_distances(heads)
        if distance_matrix is None:
            continue

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

        if len(input_ids) == 0:
            continue

        tokenized_sentences.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_to_token_positions": word_to_token_positions,
            "distance_matrix": distance_matrix
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
            all_layers = outputs.hidden_states

        word_embeddings_by_layer = []
        for layer_tensor in all_layers:
            layer_tensor = layer_tensor.squeeze(0)
            word_embeddings = average_subtokens_per_word(
                layer_tensor,
                sentence["word_to_token_positions"]
            )
            word_embeddings_by_layer.append(word_embeddings.cpu())

        all_outputs.append({
            "embeddings_by_layer": word_embeddings_by_layer,
            "distance_matrix": sentence["distance_matrix"]
        })
    return all_outputs

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # Paths
    train_file = "data/ud_ewt/en_ewt-ud-train.conllu"
    valid_file = "data/ud_ewt/en_ewt-ud-dev.conllu"
    test_file = "data/ud_ewt/en_ewt-ud-test.conllu"
    save_dir = "Final/Embeddings/UD"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Step 1: Parse train, val, and test, then concatenate
    print(f"Parsing {train_file} for training set...")
    train_sentences = parse_conllu_with_heads(train_file)
    print(f"Parsed {len(train_sentences)} training sentences.")

    print(f"Parsing {valid_file} for validation set...")
    val_sentences = parse_conllu_with_heads(valid_file)
    print(f"Parsed {len(val_sentences)} validation sentences.")

    print(f"Parsing {test_file} for test set...")
    test_sentences = parse_conllu_with_heads(test_file)
    print(f"Parsed {len(test_sentences)} test sentences.")

    all_sentences = train_sentences + val_sentences + test_sentences
    print(f"Total sentences (train + val + test): {len(all_sentences)}")

    # Step 2–3: Tokenize and align
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_all = tokenize_and_align(all_sentences, tokenizer)

    # Step 4: Encode
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    all_embeddings = encode_with_gpt2(tokenized_all, model, device)

    # Save as one file
    save_path = os.path.join(save_dir, "Embed_ud_sd.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_embeddings, f)
    print(f"Saved concatenated embeddings with distances to {save_path}")

if __name__ == "__main__":
    main()