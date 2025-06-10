"""
structural_probe_pos.py

This script probes to what extent POS tags are present in word embeddings,
using the TwoWordPSDProbe (structural probe) from structural-probes.

Inputs:
    - Pickled embeddings (train and test), each as a list of [num_words, hidden_dim] arrays
    - POS tags for each word (from the original CoNLL-U files, filtered with the same logic as preprocessing)
    - (Optional) PCA object, if probing on PCA-reduced concept

Outputs:
    - Accuracy, macro F1, and per-class metrics for the probe on the test set
"""

import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from external.structural_probes.structural_probe.probe import TwoWordPSDProbe

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Syntactic_distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Syntactic_distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
# TRAIN_CONLLU = "data/ud_ewt/en_ewt-ud-train.conllu"
# TEST_CONLLU = "data/ud_ewt/en_ewt-ud-test.conllu"
TRAIN_CONLLU = "data/narratives/train_clean.conllu"
TEST_CONLLU = "data/narratives/test_clean.conllu"

results_dir = "Syntactic_distances/Results/Narratives/LEACE/SD_on_POS_structural/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Original
# EMB_PROBE_FILE = EMBEDDING_FILE
# EMB_PROBE_TEST_FILE = TEST_FILE
# PCA_OBJ_FILE = None

# 2. LEACE-erased (vector concept)
# EMB_PROBE_FILE = "Syntactic_distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec.pkl"
# EMB_PROBE_TEST_FILE = "Syntactic_distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec.pkl"
# PCA_OBJ_FILE = None

# 3. LEACE-erased (vector concept, PCA-reduced)
EMB_PROBE_FILE = "Syntactic_distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec_pca.pkl"
EMB_PROBE_TEST_FILE = "Syntactic_distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec_pca.pkl"
PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/Narratives/leace_pca_synt_dist_vec.pkl"

# --------------------------
# Helper: Extract POS tags from CoNLL-U using the same filtering as preprocessing
# --------------------------
def extract_filtered_pos_from_conllu(conllu_path):
    sentences = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        sent = []
        text = ""
        for line in f:
            if line.startswith("# text = "):
                text = line[len("# text = "):].strip()
            if line.startswith("#") or not line.strip():
                if sent:
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
    return sentences

def get_sentence_lengths_from_embeddings(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    lengths = []
    for sent in data:
        emb = sent["embeddings_by_layer"][layer] if isinstance(sent, dict) else sent
        lengths.append(emb.shape[0])
    return lengths

def align_labels_to_embeddings(label_sentences, emb_lengths):
    aligned_labels = []
    label_idx = 0
    for emb_len in emb_lengths:
        while label_idx < len(label_sentences) and len(label_sentences[label_idx]) != emb_len:
            label_idx += 1
        if label_idx == len(label_sentences):
            break
        aligned_labels.append(label_sentences[label_idx])
        label_idx += 1
    return aligned_labels

def load_embeddings_and_labels(embedding_file, layer, label_sentences, erased_key=None):
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
    for emb, labels in zip(emb_sents, label_sentences):
        assert emb.shape[0] == len(labels), "Mismatch after alignment!"
        for i in range(emb.shape[0]):
            X.append(emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i])
            Y.append(labels[i])
    X = np.stack(X)
    Y = np.array(Y)
    return X, Y

def pos_to_int_labels(pos_list):
    unique_pos = sorted(set(pos_list))
    pos2int = {pos: i for i, pos in enumerate(unique_pos)}
    int_labels = np.array([pos2int[pos] for pos in pos_list])
    return int_labels, pos2int

# --------------------------
# Main
# --------------------------
print("Extracting POS tags from CoNLL-U files...")
train_pos_sentences = extract_filtered_pos_from_conllu(TRAIN_CONLLU)
test_pos_sentences = extract_filtered_pos_from_conllu(TEST_CONLLU)

print("Getting sentence lengths from original embeddings...")
train_emb_lengths = get_sentence_lengths_from_embeddings(EMBEDDING_FILE, LAYER)
test_emb_lengths = get_sentence_lengths_from_embeddings(TEST_FILE, LAYER)

print("Aligning POS tags to embeddings...")
aligned_train_pos = align_labels_to_embeddings(train_pos_sentences, train_emb_lengths)
aligned_test_pos = align_labels_to_embeddings(test_pos_sentences, test_emb_lengths)

print("Loading train embeddings and POS tags...")
X_train, Y_train_str = load_embeddings_and_labels(EMB_PROBE_FILE, LAYER, aligned_train_pos, erased_key="train_erased")
print("Loading test embeddings and POS tags...")
X_test, Y_test_str = load_embeddings_and_labels(EMB_PROBE_TEST_FILE, LAYER, aligned_test_pos, erased_key="test_erased")

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
# Structural probe: TwoWordPSDProbe
# --------------------------
probe_args = {
    'probe': {'maximum_rank': min(128, X_train_scaled.shape[1])},
    'model': {'hidden_dim': X_train_scaled.shape[1]},
    'device': DEVICE,
}
probe = TwoWordPSDProbe(probe_args)

# For POS, we use the mean of the predicted pairwise distances for each word as a feature for classification
def get_structural_features(X, probe):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        D_pred = probe(X_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
    # For each word, use the mean predicted distance to all other words as a feature
    features = D_pred.mean(axis=1, keepdims=True)
    return features

print("Extracting structural features for train set...")
struct_train_features = []
idx = 0
while idx < X_train_scaled.shape[0]:
    # Find sentence length (all sentences are contiguous in X_train_scaled)
    sent_len = 1
    while idx + sent_len < X_train_scaled.shape[0] and Y_train[idx + sent_len] != Y_train[idx]:
        sent_len += 1
    X_sent = X_train_scaled[idx:idx+sent_len]
    struct_train_features.append(get_structural_features(X_sent, probe))
    idx += sent_len
struct_train_features = np.vstack(struct_train_features)

print("Extracting structural features for test set...")
struct_test_features = []
idx = 0
while idx < X_test_scaled.shape[0]:
    sent_len = 1
    while idx + sent_len < X_test_scaled.shape[0] and Y_test[idx + sent_len] != Y_test[idx]:
        sent_len += 1
    X_sent = X_test_scaled[idx:idx+sent_len]
    struct_test_features.append(get_structural_features(X_sent, probe))
    idx += sent_len
struct_test_features = np.vstack(struct_test_features)

# --------------------------
# Probe: Logistic regression (multi-class) on structural features
# --------------------------
from sklearn.linear_model import LogisticRegression

print("Fitting linear classifier on structural probe features for POS tags...")
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf.fit(struct_train_features, Y_train)
Y_pred = clf.predict(struct_test_features)

acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='macro')
print(f"Accuracy on test set: {acc:.4f}")
print(f"Macro F1 on test set: {f1:.4f}")

# Per-class F1 and support
target_names = [pos for pos, idx in sorted(pos2int.items(), key=lambda x: x[1])]
labels = list(range(len(target_names)))
report = classification_report(Y_test, Y_pred, target_names=target_names, labels=labels, digits=3, zero_division=0)
print("\nPer-class F1 and support:\n")
print(report)

# --------------------------
# Save results with informative filename
# --------------------------
os.makedirs(results_dir, exist_ok=True)
emb_name = os.path.splitext(os.path.basename(EMB_PROBE_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_structural_pos_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"Accuracy: {acc}\nMacro F1: {f1}\n\n")
    f.write("Per-class F1 and support:\n")
    f.write(report)
print(f"Results saved to {results_file}")