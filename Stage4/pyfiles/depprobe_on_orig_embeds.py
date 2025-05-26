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

with open("Stage4/Embeddings/UD/Synt_deps/Original_embeddings/label2idx.json") as f:
    label2idx = json.load(f)
print("label2idx mapping:", label2idx)

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

def extract_dep_labels(word_labels, label2idx):
    """
    Converts a word_labels dict to a list of label indices.
    label2idx: dict mapping label name to index.
    Returns: dep_labels (list of int)
    """
    num_tokens = len(next(iter(word_labels.values())))
    dep_labels = []
    for i in range(num_tokens):
        found = False
        for label, idx in label2idx.items():
            if label in word_labels and word_labels[label][i]:
                dep_labels.append(idx)
                found = True
                break
        if not found:
            print(f"[extract_dep_labels] WARNING: No label found for token {i}, assigning 'root' or 0")
            dep_labels.append(label2idx.get('root', 0))  # fallback to 'root' or 0
    return dep_labels

def pad_distance_matrix(distances, max_len):
    padded_distances = torch.zeros((max_len, max_len), dtype=distances.dtype, device=distances.device)
    L = distances.shape[0]
    padded_distances[:L, :L] = distances
    return padded_distances

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
        batches.append((sentences, targets, len(data) - (i + batch_size)))
    return batches

# ---- Preprocess labels and heads ----
for idx, sentence in enumerate(train_data):
    if sentence.get("word_labels") is not None:
        dep_labels = extract_dep_labels(sentence["word_labels"], label2idx)
        sentence["dep_labels"] = dep_labels
    if sentence.get("heads") is not None:
        sentence["dep_heads"] = sentence["heads"]

for idx, sentence in enumerate(test_data):
    if sentence.get("word_labels") is not None:
        dep_labels = extract_dep_labels(sentence["word_labels"], label2idx)
        sentence["dep_labels"] = dep_labels
    if sentence.get("heads") is not None:
        sentence["dep_heads"] = sentence["heads"]

# ---- Check for unseen labels in test ----
unseen_labels = set()
for s in test_data:
    for label in s["word_labels"]:
        if label not in label2idx:
            unseen_labels.add(label)
print("Unseen labels in test:", unseen_labels)

# ---- Minimal data integrity checks ----
print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")
assert len(train_data) > 0, "Train data is empty!"
assert len(test_data) > 0, "Test data is empty!"

# Check sample dep_labels and their string equivalents
inv_label2idx = {v: k for k, v in label2idx.items()}
print("Sample train dep_labels (indices):", train_data[0].get("dep_labels"))
print("Sample train dep_labels (labels):", [inv_label2idx[i] for i in train_data[0].get("dep_labels", []) if i in inv_label2idx])
print("Sample test dep_labels (indices):", test_data[0].get("dep_labels"))
print("Sample test dep_labels (labels):", [inv_label2idx[i] for i in test_data[0].get("dep_labels", []) if i in inv_label2idx])

# ---- Check batch label variety ----
train_batches = get_depprobe_batches(train_data)
for batch in train_batches[:1]:
    targets = batch[1]
    print("Batch dep_labels unique indices:", torch.unique(targets["rels"]))
    print("Batch dep_labels unique (labels):", [inv_label2idx[i.item()] for i in torch.unique(targets["rels"]) if i.item() in inv_label2idx])
    print("Batch dep_heads unique:", torch.unique(targets["heads"]))

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
sample_batch = train_batches[0]
probe_orig.train()
outputs = probe_orig(sample_batch[0])
print("Label logits shape:", outputs['label_logits'].shape)
print("Sample label logits:", outputs['label_logits'][0,0,:].detach().cpu().numpy())

probe_orig._emb = PrecomputedEmbeddingModel(test_embs_orig.copy())
eval_stats_orig = run(probe_orig, criterion_orig, optimizer_orig, test_batches, mode='eval')

results.append({
    "type": "original",
    "acc_graph": float(np.mean(eval_stats_orig.get('acc_graph', [0])))
})

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to", RESULTS_FILE)

