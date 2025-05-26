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

def get_depprobe_batches(data):
    batches = []
    for sentence in data:
        sentence_len = sent["embeddings_by_layer"][LAYER].shape[0]
        heads = torch.tensor(sentence.get("heads", [0]*sentence), dtype=torch.long)
        rels = torch.tensor(sentence.get("rels", [0]*sentence), dtype=torch.long)
        # Add dummy distances (zeros)
        distances = torch.zeros((sentence_len, sentence_len), dtype=torch.float)
        distances = distances.unsqueeze(0)  # [1, L, L]
        targets = {"heads": heads, "rels": rels, "distances": distances}
        sentences = [""] * sentence_len  # tokens not needed
        batches.append((sentences, targets, 1))
    return batches

results = []

# ---- Depprobe on original embeddings ----
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