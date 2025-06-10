"""
structural_probe.py

This script probes to what extent the concept of syntactic distances (as a vector per word)
is present in word embeddings, using the **TwoWordPSDProbe** (structural probe) from structural-probes.

Inputs:
    - Pickled embeddings (train and test), each as a list of [num_words, hidden_dim] arrays
    - Syntactic distance matrices for each sentence (as in your LEACE scripts)
    - (Optional) PCA object, if probing on PCA-reduced concept

Outputs:
    - MSE and R^2 for the probe on the test set
"""

import pickle
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from external.structural_probes.structural_probe.probe import TwoWordPSDProbe

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_test_synt_dist.pt"
# 1. Original
EMB_PROBE_FILE = EMBEDDING_FILE
EMB_PROBE_TEST_FILE = TEST_FILE
PCA_OBJ_FILE = None

# 2. LEACE-erased (vector concept)
# EMB_PROBE_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec.pkl"
# EMB_PROBE_TEST_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec.pkl"
# PCA_OBJ_FILE = None

# 3. LEACE-erased (vector concept, PCA-reduced)
# EMB_PROBE_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec_pca.pkl"
# EMB_PROBE_TEST_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec_pca.pkl"
# PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/UD/leace_pca_synt_dist_vec.pkl"

results_dir = "Syntactic_distances/Results/UD/LEACE/SD_on_SD_structural/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load data
# --------------------------
def load_embeddings_and_distances(embedding_file, layer, max_sentence_length=None, pad_value=None):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    X = []
    D = []
    for sent in data:
        emb = sent["embeddings_by_layer"][layer] if isinstance(sent, dict) else sent
        if isinstance(sent, dict) and "distance_matrix" in sent:
            dist = sent["distance_matrix"]
            X.append(emb)
            # Pad distance matrix to (max_sentence_length, max_sentence_length)
            n = dist.shape[0]
            if max_sentence_length is not None and pad_value is not None:
                padded = np.full((max_sentence_length, max_sentence_length), pad_value, dtype=np.float32)
                padded[:n, :n] = dist
                D.append(padded)
            else:
                D.append(dist)
    return X, D

# --------------------------
# Get max sentence length and pad value for concept vector
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    train_data = pickle.load(f)
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

def get_max_sentence_length(*datasets, layer=LAYER):
    max_len = 0
    for dataset in datasets:
        for sent in dataset:
            n = sent["embeddings_by_layer"][layer].shape[0]
            if n > max_len:
                max_len = n
    return max_len

max_sentence_length = get_max_sentence_length(train_data, test_data, layer=LAYER)
pad_value = max_sentence_length - 1

# --------------------------
# Load embeddings and distances
# --------------------------
print("Loading train embeddings and distances...")
X_train_sents, D_train = load_embeddings_and_distances(EMBEDDING_FILE, LAYER, max_sentence_length, pad_value)
print("Loading test embeddings and distances...")
X_test_sents, D_test = load_embeddings_and_distances(TEST_FILE, LAYER, max_sentence_length, pad_value)

# If probing on erased embeddings, load those instead
if "leace_embeddings" in EMB_PROBE_FILE:
    with open(EMB_PROBE_FILE, "rb") as f:
        erased = pickle.load(f)
    X_train_sents = erased["train_erased"]
    with open(EMB_PROBE_TEST_FILE, "rb") as f:
        erased_test = pickle.load(f)
    X_test_sents = erased_test["test_erased"]

# Clean training data: remove any samples where the distance matrix is not finite
X_train_sents_clean = []
D_train_clean = []
removed_indices = []
for i, (x, d) in enumerate(zip(X_train_sents, D_train)):
    if np.isfinite(d).all():
        X_train_sents_clean.append(x)
        D_train_clean.append(d)
    else:
        removed_indices.append(i)
X_train_sents = X_train_sents_clean
D_train = D_train_clean

print(f"Removed {len(removed_indices)} non-finite samples from training data. Indices: {removed_indices}")

# Clean test data: remove any samples where the distance matrix is not finite
X_test_sents_clean = []
D_test_clean = []
removed_test_indices = []
for i, (x, d) in enumerate(zip(X_test_sents, D_test)):
    if np.isfinite(d).all():
        X_test_sents_clean.append(x)
        D_test_clean.append(d)
    else:
        removed_test_indices.append(i)
X_test_sents = X_test_sents_clean
D_test = D_test_clean

print(f"Removed {len(removed_test_indices)} non-finite samples from test data. Indices: {removed_test_indices}")

# --------------------------
# Standardize X (per token, then reconstruct sentences)
# --------------------------
def flatten_and_standardize(X_sents):
    X_flat = np.vstack([x if isinstance(x, np.ndarray) else x.numpy() for x in X_sents])
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    # Reconstruct sentences
    X_scaled_sents = []
    idx = 0
    for x in X_sents:
        n = x.shape[0]
        X_scaled_sents.append(torch.tensor(X_flat_scaled[idx:idx+n], dtype=torch.float32, device=DEVICE))
        idx += n
    return X_scaled_sents, scaler

X_train_scaled_sents, scaler = flatten_and_standardize(X_train_sents)
# Use train scaler for test set
def standardize_with_scaler(X_sents, scaler):
    X_flat = np.vstack([x if isinstance(x, np.ndarray) else x.numpy() for x in X_sents])
    X_flat_scaled = scaler.transform(X_flat)
    X_scaled_sents = []
    idx = 0
    for x in X_sents:
        n = x.shape[0]
        X_scaled_sents.append(torch.tensor(X_flat_scaled[idx:idx+n], dtype=torch.float32, device=DEVICE))
        idx += n
    return X_scaled_sents

X_test_scaled_sents = standardize_with_scaler(X_test_sents, scaler)

# --------------------------
# Prepare distance matrices as torch tensors
# --------------------------
D_train_torch = [torch.tensor(d, dtype=torch.float32, device=DEVICE) for d in D_train]
D_test_torch = [torch.tensor(d, dtype=torch.float32, device=DEVICE) for d in D_test]

# --------------------------
# Structural probe: TwoWordPSDProbe
# --------------------------
probe_args = {
    'probe': {'maximum_rank': min(128, X_train_scaled_sents[0].shape[1])},
    'model': {'hidden_dim': X_train_scaled_sents[0].shape[1]},
    'device': DEVICE,
}
probe = TwoWordPSDProbe(probe_args)

# Loss: MSE between predicted and gold distances (mask out padding)
def probe_loss(pred, gold, pad_value):
    mask = (gold != pad_value)
    return ((pred - gold)[mask] ** 2).mean()  # Return tensor

# Training loop (simple, not optimized)
lr = 1e-4
epochs = 10
optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

print("Training structural probe...")
# Training loop
for epoch in range(epochs):
    probe.train()
    total_loss = 0
    for x, d in zip(X_train_scaled_sents, D_train_torch):
        n = x.shape[0]
        pred = probe(x.unsqueeze(0)).squeeze(0)[:n, :n]
        gold = d[:n, :n]
        loss = probe_loss(pred, gold, pad_value)
        if not torch.isfinite(loss).item():
            print("Non-finite loss encountered!")
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Train MSE: {total_loss/len(X_train_scaled_sents):.4f}")

# --------------------------
# Evaluation
# --------------------------
probe.eval()
all_preds = []
all_golds = []
for x, d in zip(X_test_scaled_sents, D_test_torch):
    n = x.shape[0]
    with torch.no_grad():
        pred = probe(x.unsqueeze(0)).squeeze(0)[:n, :n].cpu().numpy()
    gold = d[:n, :n].cpu().numpy()
    mask = (gold != pad_value)
    all_preds.append(pred[mask])
    all_golds.append(gold[mask])
all_preds = np.concatenate(all_preds)
all_golds = np.concatenate(all_golds)

mse = mean_squared_error(all_golds, all_preds)
r2 = r2_score(all_golds, all_preds)
print(f"MSE on test set: {mse:.4f}")
print(f"R^2 on test set: {r2:.4f}")

# --------------------------
# Save results with informative filename
# --------------------------
os.makedirs(results_dir, exist_ok=True)
emb_name = os.path.splitext(os.path.basename(EMB_PROBE_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_structural_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"MSE: {mse}\nR2: {r2}\n")
print(f"Results saved to {results_file}")