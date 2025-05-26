import sys
sys.path.append('/gpfs/home2/gpeeper/LEACE')
import pickle
import torch
import os
import json
import numpy as np
from torch.optim import Adam
import torch.nn as nn
from depprobe.utils.setup import setup_model, setup_criterion, run
from torch.nn.utils.rnn import pad_sequence

# ---- Settings ----
LAYER = 8
EMBEDDING_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings.pt"
TEST_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings_test.pt"
RESULTS_FILE = "Stage4/Results/UD/Synt_deps/depprobe_original.json"
DEP_DIM = 256
EMB_LAYERS = [8, 8]

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# ---- Load data ----
with open(EMBEDDING_FILE, "rb") as f:
    train_data = pickle.load(f)
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

# ---- Helper functions ----
def get_embeddings_by_sentence(data, layer):
    return [sentence["embeddings_by_layer"][layer] for sentence in data]

class PrecomputedEmbeddingModel(nn.Module):
    def __init__(self, precomputed_embs):
        super().__init__()
        self.precomputed_embs = precomputed_embs.copy()
        self.emb_dim = self.precomputed_embs[0].shape[1]

    def forward(self, sentences, *args, **kwargs):
        batch_embs = []
        lengths = []
        for _ in sentences:
            emb = self.precomputed_embs.pop(0)  # shape: [L, D]
            if torch.cuda.is_available():
                emb = emb.cuda()
            batch_embs.append(emb)
            lengths.append(emb.shape[0])
        batch_tensor = pad_sequence(batch_embs, batch_first=True)  # [B, max_len, D]
        max_len = batch_tensor.shape[1]
        att_mask = torch.zeros((len(batch_embs), max_len), dtype=torch.bool, device=batch_tensor.device)
        for i, l in enumerate(lengths):
            att_mask[i, :l] = True
        return [batch_tensor, batch_tensor], att_mask

def extract_dep_labels(word_labels, label2idx=None):
    """
    Converts a word_labels dict to a list of label indices.
    label2idx: dict mapping label name to index. If None, will create one.
    Returns: dep_labels (list of int), label2idx (dict)
    """
    if label2idx is None:
        label2idx = {label: idx for idx, label in enumerate(word_labels.keys())}
    num_tokens = len(next(iter(word_labels.values())))
    dep_labels = []
    for i in range(num_tokens):
        found = False
        for label, idx in label2idx.items():
            if word_labels[label][i]:
                dep_labels.append(idx)
                found = True
                break
        if not found:
            dep_labels.append(label2idx.get('root', 0))  # fallback to 'root' or 0
    return dep_labels, label2idx

def pad_distance_matrix(distances, max_len):
    padded_distances = torch.zeros((max_len, max_len), dtype=distances.dtype, device=distances.device)
    L = distances.shape[0]
    padded_distances[:L, :L] = distances
    return padded_distances

def pad_distance_matrix_to(pred_distances, trgt_distances):
    max_len = pred_distances.shape[1]
    L = trgt_distances.shape[1]
    padded = torch.full((1, max_len, max_len), -1, dtype=trgt_distances.dtype, device=trgt_distances.device)
    padded[:, :L, :L] = trgt_distances
    return padded

def get_depprobe_batches(data, batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_lens = [s["embeddings_by_layer"][LAYER].shape[0] for s in batch]
        max_len = max(batch_lens)
        dep_heads_batch = []
        dep_labels_batch = []
        distances_batch = []
        for sentence, sentence_len in zip(batch, batch_lens):
            dep_heads = torch.tensor(sentence.get("dep_heads", [0]*sentence_len), dtype=torch.long, device=device)
            dep_labels = torch.tensor(sentence.get("dep_labels", [0]*sentence_len), dtype=torch.long, device=device)
            if sentence_len < max_len:
                pad = torch.full((max_len - sentence_len,), -1, dtype=torch.long, device=device)
                dep_labels = torch.cat([dep_labels, pad])
                dep_heads = torch.cat([dep_heads, pad])
            distances = torch.zeros((sentence_len, sentence_len), dtype=torch.float, device=device)
            padded_distances = pad_distance_matrix(distances, max_len).to(device)
            dep_heads_batch.append(dep_heads)
            dep_labels_batch.append(dep_labels)
            distances_batch.append(padded_distances)
        dep_heads_batch = torch.stack(dep_heads_batch)
        dep_labels_batch = torch.stack(dep_labels_batch)
        distances_batch = torch.stack(distances_batch)
        targets = {"heads": dep_heads_batch, "rels": dep_labels_batch, "distances": distances_batch}
        sentences = [[""] * max_len for _ in range(len(batch))]
        # Debug prints for batch content
        print("dep_labels unique:", torch.unique(dep_labels_batch))
        print("dep_heads unique:", torch.unique(dep_heads_batch))
        print("dep_labels_batch shape:", dep_labels_batch.shape)
        print("dep_heads_batch shape:", dep_heads_batch.shape)
        print("distances_batch shape:", distances_batch.shape)
        batches.append((sentences, targets, len(data) - (i + batch_size)))
    return batches

# ---- Preprocess: Extract dep_labels from word_labels ----
label2idx = None
for sentence in train_data:
    if sentence.get("word_labels") is not None:
        dep_labels, label2idx = extract_dep_labels(sentence["word_labels"], label2idx)
        sentence["dep_labels"] = dep_labels
    if sentence.get("heads") is not None:
        sentence["dep_heads"] = sentence["heads"]
for sentence in test_data:
    if sentence.get("word_labels") is not None:
        dep_labels, _ = extract_dep_labels(sentence["word_labels"], label2idx)
        sentence["dep_labels"] = dep_labels
    if sentence.get("heads") is not None:
        sentence["dep_heads"] = sentence["heads"]

# ---- Depprobe on original embeddings ----
results = []
train_embs_orig = get_embeddings_by_sentence(train_data, LAYER)
test_embs_orig = get_embeddings_by_sentence(test_data, LAYER)
train_batches = get_depprobe_batches(train_data)
test_batches = get_depprobe_batches(test_data)

probe_orig = setup_model(
    lm_name="gpt2",
    dep_dim=DEP_DIM,
    parser_type="depprobe",
    emb_layers=EMB_LAYERS
)
criterion_orig = setup_criterion(parser_type="depprobe")
optimizer_orig = Adam(probe_orig.parameters(), lr=1e-3)

probe_orig._emb = PrecomputedEmbeddingModel(train_embs_orig.copy())
run(probe_orig, criterion_orig, optimizer_orig, train_batches, mode='train')
probe_orig._emb = PrecomputedEmbeddingModel(test_embs_orig.copy())
eval_stats_orig = run(probe_orig, criterion_orig, optimizer_orig, test_batches, mode='eval')

results.append({
    "type": "original",
    "acc_graph": float(np.mean(eval_stats_orig.get('acc_graph', [0])))
})

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to", RESULTS_FILE)
print("Sample train_data[0]:", train_data[0])
print("Sample dep_labels:", train_data[0].get("dep_labels"))
print("Sample dep_heads:", train_data[0].get("dep_heads"))

