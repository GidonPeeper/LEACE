import argparse
import pickle
import os
from conllu import parse_incr
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
import torch

# ----------------------------------------------------------------------
# Step 1: Parse .conllu into words and POS tags
# ----------------------------------------------------------------------

def parse_conllu(file_path):
    sentences = []
    all_pos_tags = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            pos_tags = []
            for token in tokenlist:
                if isinstance(token['id'], int):  # Skip multi-word tokens
                    words.append(token['form'])
                    pos_tags.append(token['upostag'])
                    all_pos_tags.add(token['upostag'])
            sentences.append({
                "words": words,
                "pos_tags": pos_tags
            })
    return sentences, all_pos_tags

# ----------------------------------------------------------------------
# Step 2: Label generator for all POS tags as concepts
# ----------------------------------------------------------------------

def get_feature_matrix(pos_tags, all_pos_tags):
    labels = {}
    for pos in all_pos_tags:
        labels[pos] = [1 if tag == pos else 0 for tag in pos_tags]
    return labels

# ----------------------------------------------------------------------
# Step 3: Tokenize with GPT-2 and align word-level labels
# ----------------------------------------------------------------------

def tokenize_and_label(sentences, all_pos_tags, tokenizer):
    tokenized_sentences = []
    all_pos_tags = sorted(list(all_pos_tags))
    for sentence in tqdm(sentences, desc="Tokenizing and aligning"):
        words = sentence["words"]
        pos_tags = sentence["pos_tags"]
        feature_labels = get_feature_matrix(pos_tags, all_pos_tags)
        feature_names = list(feature_labels.keys())

        input_ids = []
        attention_mask = []
        word_to_token_positions = []
        word_labels = {feature: [] for feature in feature_names}
        current_token_position = 0

        for i, word in enumerate(words):
            word_tokens = tokenizer.tokenize(word)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            if not word_ids:
                continue
            token_positions = list(range(current_token_position, current_token_position + len(word_ids)))
            word_to_token_positions.append(token_positions)
            input_ids.extend(word_ids)
            attention_mask.extend([1] * len(word_ids))
            current_token_position += len(word_ids)
            for feature in feature_names:
                word_labels[feature].append(feature_labels[feature][i])

        tokenized_sentences.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_to_token_positions": word_to_token_positions,
            "word_labels": word_labels
        })

    return tokenized_sentences, all_pos_tags

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
            "word_labels": sentence["word_labels"]             # Dict[str, List[int]]
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
    save_dir = "Stage3/Embeddings"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Step 1: Parse and concatenate train + valid for training set
    print(f"Parsing {train_file} and {valid_file} for training set...")
    train_sentences, train_pos_tags = parse_conllu(train_file)
    valid_sentences, valid_pos_tags = parse_conllu(valid_file)
    all_train_sentences = train_sentences + valid_sentences
    all_pos_tags = train_pos_tags.union(valid_pos_tags)
    print(f"Parsed {len(all_train_sentences)} training+valid sentences.")
    print(f"POS tags found: {sorted(list(all_pos_tags))}")

    # Step 2–3: Tokenize and label training set
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_train, all_pos_tags = tokenize_and_label(all_train_sentences, all_pos_tags, tokenizer)

    # Step 4: Encode training set
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    train_embeddings = encode_with_gpt2(tokenized_train, model, device)

    # Save training embeddings
    train_save_path = os.path.join(save_dir, "gpt2_embeddings.pt")
    with open(train_save_path, "wb") as f:
        pickle.dump(train_embeddings, f)
    print(f"Saved training embeddings to {train_save_path}")

    # Step 1: Parse test set
    print(f"Parsing {test_file} for test set...")
    test_sentences, test_pos_tags = parse_conllu(test_file)
    print(f"Parsed {len(test_sentences)} test sentences.")

    # Step 2–3: Tokenize and label test set (use all_pos_tags from train+valid)
    tokenized_test, _ = tokenize_and_label(test_sentences, all_pos_tags, tokenizer)

    # Step 4: Encode test set
    test_embeddings = encode_with_gpt2(tokenized_test, model, device)

    # Save test embeddings
    test_save_path = os.path.join(save_dir, "gpt2_embeddings_test.pt")
    with open(test_save_path, "wb") as f:
        pickle.dump(test_embeddings, f)
    print(f"Saved test embeddings to {test_save_path}")

if __name__ == "__main__":
    main()