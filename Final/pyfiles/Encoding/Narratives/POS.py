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

def parse_conllu_with_pos(file_path):
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
# Step 2: Tokenize with GPT-2 and align word-level POS tags
# ----------------------------------------------------------------------

def tokenize_and_align_with_pos(sentences, all_pos_tags, tokenizer):
    tokenized_sentences = []
    all_pos_tags = sorted(list(all_pos_tags))
    for sentence in tqdm(sentences, desc="Tokenizing and aligning"):
        words = sentence["words"]
        pos_tags = sentence["pos_tags"]

        input_ids = []
        attention_mask = []
        word_to_token_positions = []
        aligned_pos_tags = []
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
            aligned_pos_tags.append(pos_tags[i])

        # SKIP sentences with no tokens
        if len(input_ids) == 0:
            continue

        tokenized_sentences.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_to_token_positions": word_to_token_positions,
            "pos_tags": aligned_pos_tags  # List[str], one per word
        })

    return tokenized_sentences, all_pos_tags

# ----------------------------------------------------------------------
# Step 3: Run GPT-2 and average subtoken embeddings per word
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
            "pos_tags": sentence["pos_tags"]                  # List[str], one per word
        })

    return all_outputs

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # Paths
    train_file = "data/narratives/train_clean.conllu"
    test_file = "data/narratives/test_clean.conllu"
    save_dir = "Final/Embeddings/Original/Narratives"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Step 1: Parse train and test, then concatenate
    print(f"Parsing {train_file} for training set...")
    train_sentences, train_pos_tags = parse_conllu_with_pos(train_file)
    print(f"Parsed {len(train_sentences)} training sentences.")

    print(f"Parsing {test_file} for test set...")
    test_sentences, test_pos_tags = parse_conllu_with_pos(test_file)
    print(f"Parsed {len(test_sentences)} test sentences.")

    all_sentences = train_sentences + test_sentences
    all_pos_tags = train_pos_tags.union(test_pos_tags)
    print(f"Total sentences (train + test): {len(all_sentences)}")
    print(f"POS tags found: {sorted(list(all_pos_tags))}")

    # Step 2–3: Tokenize and align
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_all, all_pos_tags = tokenize_and_align_with_pos(all_sentences, all_pos_tags, tokenizer)

    # Step 4: Encode
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device).eval()
    all_embeddings = encode_with_gpt2(tokenized_all, model, device)

    # Save as one file
    save_path = os.path.join(save_dir, "Embed_nar_pos.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_embeddings, f)
    print(f"Saved concatenated embeddings with POS tags to {save_path}")

if __name__ == "__main__":
    main()