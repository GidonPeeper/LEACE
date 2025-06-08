"""
linear_probe_oracle.py

This script probes to what extent the concept of syntactic distances (as a vector per word)
is present in word embeddings after **Oracle LEACE** erasure.
Oracle LEACE removes the concept from the *entire embedding space* (not just a train/test split),
so you should probe on the same data that was erased.

Inputs:
    - Pickled Oracle LEACE-erased embeddings (with "train_erased" and "test_erased" arrays)
    - Syntactic distance vectors (concepts) for each word, as in your LEACE scripts
    - (Optional) PCA object, if probing on PCA-reduced concept

Outputs:
    - MSE and R^2 for the probe on the erased embeddings
"""

import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Settings
# --------------------------
# Narratives or UD
LAYER = 8
ORIGINAL_EMB_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"
results_dir = "Syntactic_distances/Results/UD/Oracle/SD_on_SD/"

# Select one of the following:
# 1. Oracle LEACE (no PCA)
ORACLE_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec.pkl"
PCA_OBJ_FILE = None

# 2. Oracle LEACE + PCA
# ORACLE_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec_pca.pkl"
# PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/UD/oracle_pca_synt_dist_vec.pkl"

# 3. Original (no erasure)
# ORACLE_EMB_FILE = ORIGINAL_EMB_FILE
# PCA_OBJ_FILE = None


# --------------------------
# Helper functions for alignment
# --------------------------
def get_sentence_lengths_from_erased(erased_list):
    return [arr.shape[0] for arr in erased_list]

def get_sentence_lengths_from_original(orig_data, layer):
    return [sent["embeddings_by_layer"][layer].shape[0] for sent in orig_data]

def load_concepts(embedding_file, layer, max_sentence_length=None, pad_value=None):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    Z = []
    for sent in data:
        emb = sent["embeddings_by_layer"][layer] if isinstance(sent, dict) else sent
        if isinstance(sent, dict) and "distance_matrix" in sent:
            dist = sent["distance_matrix"]
            for i in range(emb.shape[0]):
                row = dist[i].tolist()
                if max_sentence_length is not None and pad_value is not None:
                    row += [pad_value] * (max_sentence_length - len(row))
                Z.append(row)
    Z = np.stack(Z)
    return Z

# --------------------------
# Get max sentence length and pad value for concept vector
# --------------------------
with open(ORIGINAL_EMB_FILE, "rb") as f:
    orig_data = pickle.load(f)

def get_max_sentence_length(dataset, layer=LAYER):
    max_len = 0
    for sent in dataset:
        n = sent["embeddings_by_layer"][layer].shape[0]
        if n > max_len:
            max_len = n
    return max_len

max_sentence_length = get_max_sentence_length(orig_data, layer=LAYER)
pad_value = max_sentence_length - 1

# --------------------------
# Load embeddings and concepts
# --------------------------
if ORACLE_EMB_FILE.endswith(".pt"):
    # Original (not erased): load as list of dicts
    with open(ORACLE_EMB_FILE, "rb") as f:
        orig_data_probe = pickle.load(f)
    X_erased_sents = [sent["embeddings_by_layer"][LAYER] for sent in orig_data_probe]
else:
    # Erased: load as dict with "train_erased"
    with open(ORACLE_EMB_FILE, "rb") as f:
        erased = pickle.load(f)
    X_erased_sents = erased["train_erased"]

# --------------------------
# Load concepts (syntactic distance vectors)
# --------------------------
print("Loading concepts...")
Z = load_concepts(ORIGINAL_EMB_FILE, LAYER, max_sentence_length, pad_value)

# --------------------------
# Align erased embeddings and concepts by sentence length
# --------------------------
if ORACLE_EMB_FILE.endswith(".pt"):
    # Original: align using original data
    orig_lengths = get_sentence_lengths_from_original(orig_data, LAYER)
    X_erased_flat = []
    Z_aligned = []
    z_idx = 0
    for o_len, emb_arr in zip(orig_lengths, X_erased_sents):
        X_erased_flat.append(emb_arr)
        Z_aligned.extend(Z[z_idx:z_idx+o_len])
        z_idx += o_len
    X_erased_flat = np.vstack(X_erased_flat)
    Z_aligned = np.stack(Z_aligned)
else:
    # Erased: align using both erased and original lengths
    erased_lengths = get_sentence_lengths_from_erased(X_erased_sents)
    orig_lengths = get_sentence_lengths_from_original(orig_data, LAYER)
    X_erased_flat = []
    Z_aligned = []
    z_idx = 0
    for e_len, o_len, emb_arr in zip(erased_lengths, orig_lengths, X_erased_sents):
        if e_len != o_len:
            z_idx += o_len  # skip these tokens in Z
            continue
        X_erased_flat.append(emb_arr)
        Z_aligned.extend(Z[z_idx:z_idx+o_len])
        z_idx += o_len
    X_erased_flat = np.vstack(X_erased_flat)
    Z_aligned = np.stack(Z_aligned)

print(f"Aligned {X_erased_flat.shape[0]} tokens for probing.")

# --------------------------
# Standardize X
# --------------------------
scaler = StandardScaler()
X_erased_scaled = scaler.fit_transform(X_erased_flat)

# Remove any rows with non-finite values in X or Z
mask = np.isfinite(X_erased_scaled).all(axis=1) & np.isfinite(Z_aligned).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values.")
    X_erased_scaled = X_erased_scaled[mask]
    Z_aligned = Z_aligned[mask]

# If probing on PCA-reduced concept, project Z
if PCA_OBJ_FILE is not None:
    with open(PCA_OBJ_FILE, "rb") as f:
        pca = pickle.load(f)
    Z_aligned = pca.transform(Z_aligned)

# --------------------------
# Probe: Ridge regression (multi-output)
# --------------------------
print("Fitting linear probe on selected embeddings...")
probe = Ridge(alpha=1.0)
probe.fit(X_erased_scaled, Z_aligned)
Z_pred = probe.predict(X_erased_scaled)

mse = mean_squared_error(Z_aligned, Z_pred)
r2 = r2_score(Z_aligned, Z_pred)
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# --------------------------
# Save results
# --------------------------

os.makedirs(results_dir, exist_ok=True)
emb_name = os.path.splitext(os.path.basename(ORACLE_EMB_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"MSE: {mse}\nR2: {r2}\n")
print(f"Results saved to {results_file}")