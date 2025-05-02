import torch
import pickle
from transformers import GPT2Tokenizer

# Load the embeddings
with open('gpt2_embeddings.pt', 'rb') as f:
    embeddings = pickle.load(f)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Print basic information
print(f"Number of sentences: {len(embeddings)}")
print("\nFirst sentence structure:")
first_sentence = embeddings[0]
print(f"Keys in first sentence: {first_sentence.keys()}")
print(f"Number of layers: {len(first_sentence['embeddings_by_layer'])}")
print(f"Embedding dimension: {first_sentence['embeddings_by_layer'][0].shape[1]}")
print(f"Number of words: {len(first_sentence['embeddings_by_layer'][0])}")

# Print feature labels
print("\nFeature labels:")
print(first_sentence['word_labels'].keys())

# Print first sentence details
print("\nFirst sentence details:")
print("-" * 50)
print("Word\t\tPOS Tag\t\tFunction Word\tGPT-2 Tokens")
print("-" * 50)

# Load the original CONLLU file to get the words and POS tags
from conllu import parse_incr
with open('data/ud_ewt/en_ewt-ud-train.conllu', 'r', encoding='utf-8') as f:
    first_sentence_conllu = next(parse_incr(f))
    for token in first_sentence_conllu:
        if isinstance(token['id'], int):  # Skip multi-word tokens
            word = token['form']
            pos = token['upostag']
            is_function = pos in {'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ'}
            gpt2_tokens = tokenizer.tokenize(word)
            print(f"{word:15}\t{pos:10}\t{is_function}\t\t{' '.join(gpt2_tokens)}")
print("-" * 50)

# Print embedding statistics
print("\nEmbedding Statistics:")
print(f"Number of layers: {len(first_sentence['embeddings_by_layer'])}")
print(f"Embedding dimension: {first_sentence['embeddings_by_layer'][0].shape[1]}")
print("\nFirst word embedding (first 5 dimensions) from each layer:")
for i, layer_emb in enumerate(first_sentence['embeddings_by_layer']):
    print(f"Layer {i}: {layer_emb[0][:5].tolist()}")

# Print some statistics
print("\nGeneral Statistics:")
print(f"Total number of sentences: {len(embeddings)}")
print(f"Average words per sentence: {sum(len(s['embeddings_by_layer'][0]) for s in embeddings) / len(embeddings):.2f}")
print(f"Total number of words: {sum(len(s['embeddings_by_layer'][0]) for s in embeddings)}")

# Print tokenization statistics
print("\nTokenization Statistics:")
total_subwords = 0
total_words = 0
for sentence in embeddings:
    total_words += len(sentence['embeddings_by_layer'][0])
    # The number of subwords is equal to the length of the first layer's embeddings
    total_subwords += len(sentence['embeddings_by_layer'][0])
print(f"Average subwords per word: {total_subwords/total_words:.2f}") 