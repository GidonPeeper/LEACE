"""
probe_synt_dist_vector.py

This script probes to what extent the concept of syntactic distances (as a vector per word)
is present in word embeddings. It works for original and LEACE-erased embeddings.

Inputs:
    - Pickled embeddings (train and test), each as a list of [num_words, hidden_dim] arrays
    - Syntactic distance vectors (concepts) for each word, as in your LEACE scripts
    - (Optional) PCA object, if probing on PCA-reduced concept

Outputs:
    - MSE and R^2 for the probe on the test set
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
EMBEDDING_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_test_synt_dist.pt"

# Choose one of the following for probing:
# 1. Original
EMB_PROBE_FILE = EMBEDDING_FILE
EMB_PROBE_TEST_FILE = TEST_FILE
PCA_OBJ_FILE = None

# 2. LEACE-erased (vector concept)
# EMB_PROBE_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec.pkl"
# EMB_PROBE_TEST_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec.pkl"
# PCA_OBJ_FILE = None

# 3. LEACE-erased (vector concept, PCA-reduced)
EMB_PROBE_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec_pca.pkl"
EMB_PROBE_TEST_FILE = "Syntactic_distances/Embeddings/Erased/UD/leace_embeddings_synt_dist_vec_pca.pkl"
PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/UD/leace_pca_synt_dist_vec.pkl"

results_dir = "Syntactic_distances/Results/UD/LEACE/SD_on_SD/"

# --------------------------
# Load data
# --------------------------
def load_embeddings_and_concepts(embedding_file, layer, max_sentence_length=None, pad_value=None):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    X = []
    Z = []
    for sent in data:
        emb = sent["embeddings_by_layer"][layer] if isinstance(sent, dict) else sent  # [num_words, hidden_dim] or just [num_words, hidden_dim]
        if isinstance(sent, dict) and "distance_matrix" in sent:
            dist = sent["distance_matrix"]
            for i in range(emb.shape[0]):
                X.append(emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i])
                row = dist[i].tolist()
                if max_sentence_length is not None and pad_value is not None:
                    row += [pad_value] * (max_sentence_length - len(row))
                Z.append(row)
        else:
            # If erased embeddings, just append
            for i in range(emb.shape[0]):
                X.append(emb[i])
    X = np.stack(X)
    if Z:
        Z = np.stack(Z)
    else:
        Z = None
    return X, Z

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
# Load embeddings and concepts
# --------------------------
print("Loading train embeddings and concepts...")
X_train, Z_train = load_embeddings_and_concepts(EMBEDDING_FILE, LAYER, max_sentence_length, pad_value)
print("Loading test embeddings and concepts...")
X_test, Z_test = load_embeddings_and_concepts(TEST_FILE, LAYER, max_sentence_length, pad_value)

# If probing on erased embeddings, load those instead
if "leace_embeddings" in EMB_PROBE_FILE:
    with open(EMB_PROBE_FILE, "rb") as f:
        erased = pickle.load(f)
    X_train = np.vstack(erased["train_erased"])
    X_test = np.vstack(erased["test_erased"])

# If probing on PCA-reduced concept, project Z
if PCA_OBJ_FILE is not None:
    import pickle
    with open(PCA_OBJ_FILE, "rb") as f:
        pca = pickle.load(f)
    # Remove any rows with non-finite values in Z_train or Z_test before PCA
    mask_train = np.isfinite(Z_train).all(axis=1)
    if not mask_train.all():
        print(f"Filtered out {np.sum(~mask_train)} rows with non-finite values in Z_train before PCA.")
        Z_train = Z_train[mask_train]
        X_train = X_train[mask_train]  # Also filter X_train so alignment is preserved

    mask_test = np.isfinite(Z_test).all(axis=1)
    if not mask_test.all():
        print(f"Filtered out {np.sum(~mask_test)} rows with non-finite values in Z_test before PCA.")
        Z_test = Z_test[mask_test]
        X_test = X_test[mask_test]  # Also filter X_test so alignment is preserved

    # Now you can safely apply PCA
    Z_train = pca.transform(Z_train)
    Z_test = pca.transform(Z_test)

    # Then scale X as usual
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    # --------------------------
    # Standardize X
    # --------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Remove any rows with non-finite values in X or Z
mask = np.isfinite(X_train_scaled).all(axis=1) & np.isfinite(Z_train).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values in training set.")
    X_train_scaled = X_train_scaled[mask]
    Z_train = Z_train[mask]

mask_test = np.isfinite(X_test_scaled).all(axis=1) & np.isfinite(Z_test).all(axis=1)
if not mask_test.all():
    print(f"Filtered out {np.sum(~mask_test)} rows with non-finite values in test set.")
    X_test_scaled = X_test_scaled[mask_test]
    Z_test = Z_test[mask_test]

# --------------------------
# Probe: Ridge regression (multi-output)
# --------------------------
print("Fitting linear probe...")
probe = Ridge(alpha=1.0)
probe.fit(X_train_scaled, Z_train)
Z_pred = probe.predict(X_test_scaled)

mse = mean_squared_error(Z_test, Z_pred)
r2 = r2_score(Z_test, Z_pred)
print(f"MSE on test set: {mse:.4f}")
print(f"R^2 on test set: {r2:.4f}")

# --------------------------
# Save results with informative filename
# --------------------------
os.makedirs(results_dir, exist_ok=True)

# Use the embedding file name to distinguish results
emb_name = os.path.splitext(os.path.basename(EMB_PROBE_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"MSE: {mse}\nR2: {r2}\n")
print(f"Results saved to {results_file}")