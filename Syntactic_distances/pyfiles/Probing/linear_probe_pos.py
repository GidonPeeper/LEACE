"""
linear_probe_pos.py

This script probes to what extent the concept of POS (part-of-speech) tags is present in word embeddings.
It works for original, LEACE-erased, and PCA-erased embeddings.

Inputs:
    - Pickled embeddings (train and test), each as a list of [num_words, hidden_dim] arrays
    - POS tags for each word (from the original CoNLL-U files, filtered with the same logic as in preprocessing)
    - (Optional) PCA object, if probing on PCA-reduced concept

Outputs:
    - Accuracy and macro F1 for the probe on the test set
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
TRAIN_CONLLU = "data/narratives/train_clean.conllu"
TEST_CONLLU = "data/narratives/test_clean.conllu"

# 1. Original
# EMB_PROBE_FILE = EMBEDDING_FILE
# EMB_PROBE_TEST_FILE = TEST_FILE
# PCA_OBJ_FILE = None

# 2. LEACE-erased (vector concept)
# EMB_PROBE_FILE = "Distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec.pkl"
# EMB_PROBE_TEST_FILE = "Distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec.pkl"
# PCA_OBJ_FILE = None

# 3. LEACE-erased (vector concept, PCA-reduced)
EMB_PROBE_FILE = "Distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec_pca.pkl"
EMB_PROBE_TEST_FILE = "Distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec_pca.pkl"
PCA_OBJ_FILE = "Distances/Eraser_objects/Narratives/leace_pca_synt_dist_vec.pkl"

# --------------------------
# Helper: Extract POS tags from CoNLL-U using the same filtering as preprocessing
# --------------------------
def extract_filtered_pos_from_conllu(conllu_path):
    sentences = []
    texts = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        sent = []
        text = ""
        for line in f:
            if line.startswith("# text = "):
                text = line[len("# text = "):].strip()
            if line.startswith("#") or not line.strip():
                if sent:
                    # Filtering: skip if empty or if only one token and that token is the sentence text
                    valid_tokens = [pos for pos in sent if pos.strip()]
                    if len(valid_tokens) == 0:
                        sent = []
                        text = ""
                        continue
                    if len(valid_tokens) == 1 and valid_tokens[0].strip() == text.strip():
                        sent = []
                        text = ""
                        continue
                    sentences.append(sent)
                    texts.append(text)
                    sent = []
                    text = ""
                continue
            fields = line.strip().split("\t")
            if len(fields) > 3:
                sent.append(fields[3])  # UPOS
        if sent:
            valid_tokens = [pos for pos in sent if pos.strip()]
            if len(valid_tokens) == 0:
                pass
            elif len(valid_tokens) == 1 and valid_tokens[0].strip() == text.strip():
                pass
            else:
                sentences.append(sent)
                texts.append(text)
    return sentences, texts

# --------------------------
# Load embeddings and align POS tags by filtering both with the same logic
# --------------------------
def load_embeddings_and_pos(embedding_file, layer, pos_sentences, erased_key=None):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and erased_key is not None:
        emb_sents = data[erased_key]
    else:
        emb_sents = []
        for sent in data:
            emb = sent["embeddings_by_layer"][layer] if isinstance(sent, dict) else sent
            emb_sents.append(emb)
    X = []
    Y = []
    for emb, pos_tags in zip(emb_sents, pos_sentences):
        assert emb.shape[0] == len(pos_tags), "Mismatch after alignment!"
        for i in range(emb.shape[0]):
            X.append(emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i])
            Y.append(pos_tags[i])
    X = np.stack(X)
    Y = np.array(Y)
    return X, Y

def get_sentence_lengths_from_embeddings(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    lengths = []
    for sent in data:
        emb = sent["embeddings_by_layer"][layer] if isinstance(sent, dict) else sent
        lengths.append(emb.shape[0])
    return lengths

def align_pos_to_embeddings(pos_sentences, emb_lengths):
    aligned_pos = []
    pos_idx = 0
    for emb_len in emb_lengths:
        # Skip POS sentences until we find one with the right length
        while pos_idx < len(pos_sentences) and len(pos_sentences[pos_idx]) != emb_len:
            pos_idx += 1
        if pos_idx == len(pos_sentences):
            break
        aligned_pos.append(pos_sentences[pos_idx])
        pos_idx += 1
    return aligned_pos

# --------------------------
# Map POS tags to integer labels
# --------------------------
def pos_to_int_labels(pos_list):
    unique_pos = sorted(set(pos_list))
    pos2int = {pos: i for i, pos in enumerate(unique_pos)}
    int_labels = np.array([pos2int[pos] for pos in pos_list])
    return int_labels, pos2int

# --------------------------
# Main
# --------------------------
print("Extracting POS tags from CoNLL-U files...")
train_pos_sentences, _ = extract_filtered_pos_from_conllu(TRAIN_CONLLU)
test_pos_sentences, _ = extract_filtered_pos_from_conllu(TEST_CONLLU)

print("Getting sentence lengths from original embeddings...")
train_emb_lengths = get_sentence_lengths_from_embeddings(EMBEDDING_FILE, LAYER)
test_emb_lengths = get_sentence_lengths_from_embeddings(TEST_FILE, LAYER)

print("Aligning POS tags to embeddings...")
aligned_train_pos = align_pos_to_embeddings(train_pos_sentences, train_emb_lengths)
aligned_test_pos = align_pos_to_embeddings(test_pos_sentences, test_emb_lengths)

print("Loading train embeddings and POS...")
# For original embeddings:
# X_train, Y_train_str = load_embeddings_and_pos(EMB_PROBE_FILE, LAYER, train_pos_sentences)
# X_test, Y_test_str = load_embeddings_and_pos(EMB_PROBE_TEST_FILE, LAYER, test_pos_sentences)

# For erased embeddings:
X_train, Y_train_str = load_embeddings_and_pos(EMB_PROBE_FILE, LAYER, aligned_train_pos, erased_key="train_erased")
X_test, Y_test_str = load_embeddings_and_pos(EMB_PROBE_TEST_FILE, LAYER, aligned_test_pos, erased_key="test_erased")

print("Loading test embeddings and POS...")

# If probing on erased embeddings, load those instead
if "leace_embeddings" in EMB_PROBE_FILE:
    with open(EMB_PROBE_FILE, "rb") as f:
        erased = pickle.load(f)
    X_train = np.vstack(erased["train_erased"])
    X_test = np.vstack(erased["test_erased"])

# If probing on PCA-reduced concept, just use the embeddings as is (no change to Y)
if PCA_OBJ_FILE is not None:
    import pickle
    with open(PCA_OBJ_FILE, "rb") as f:
        pca = pickle.load(f)
    # No change to Y_train/Y_test for POS

# Map POS tags to integer labels (fit on train, apply to test)
Y_train, pos2int = pos_to_int_labels(Y_train_str)
Y_test = np.array([pos2int.get(pos, -1) for pos in Y_test_str])
mask_valid = Y_test != -1
if not mask_valid.all():
    print(f"Warning: {np.sum(~mask_valid)} POS tags in test set not seen in train set. These will be ignored.")
    X_test = X_test[mask_valid]
    Y_test = Y_test[mask_valid]

# --------------------------
# Standardize X
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Remove any rows with non-finite values in X or Y
mask = np.isfinite(X_train_scaled).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values in training set.")
    X_train_scaled = X_train_scaled[mask]
    Y_train = Y_train[mask]

mask_test = np.isfinite(X_test_scaled).all(axis=1)
if not mask_test.all():
    print(f"Filtered out {np.sum(~mask_test)} rows with non-finite values in test set.")
    X_test_scaled = X_test_scaled[mask_test]
    Y_test = Y_test[mask_test]

# --------------------------
# Probe: Logistic regression (multi-class)
# --------------------------
print("Fitting linear probe for POS...")
probe = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
probe.fit(X_train_scaled, Y_train)
Y_pred = probe.predict(X_test_scaled)

acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='macro')
print(f"Accuracy on test set: {acc:.4f}")
print(f"Macro F1 on test set: {f1:.4f}")

# --------------------------
# Save results with informative filename
# --------------------------
results_dir = "Distances/Results/Narratives"
os.makedirs(results_dir, exist_ok=True)

emb_name = os.path.splitext(os.path.basename(EMB_PROBE_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_pos_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"Accuracy: {acc}\nMacro F1: {f1}\n")
print(f"Results saved to {results_file}")