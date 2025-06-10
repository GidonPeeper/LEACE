"""
linear_probe_oracle_pos.py

This script probes to what extent the concept of POS (part-of-speech) tags is present in word embeddings
after Oracle LEACE erasure of syntactic distances, optionally with PCA.

Inputs:
    - Pickled Oracle LEACE-erased embeddings (with "train_erased" array)
    - POS tags for each word (from the original CoNLL-U files, filtered with the same logic as in preprocessing)
    - (Optional) PCA object, if probing on PCA-reduced embeddings

Outputs:
    - Accuracy, macro F1, and per-class metrics for the probe on the erased embeddings (on the training data)
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Settings
# --------------------------
# Narratives or UD
results_dir = "Syntactic_distances/Results/UD/Oracle/SD_on_POS/"
LAYER = 8

# Select one of the following:
ORIGINAL_EMB_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"

# 1. Oracle LEACE (no PCA)
# ORACLE_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec.pkl"
# PCA_OBJ_FILE = None

# 2. Oracle LEACE + PCA
ORACLE_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec_pca.pkl"
PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/UD/oracle_pca_synt_dist_vec.pkl"

# 3. Original (no erasure)
# ORACLE_EMB_FILE = ORIGINAL_EMB_FILE
# PCA_OBJ_FILE = None

# TRAIN_CONLLU = "data/narratives/train_clean.conllu"
TRAIN_CONLLU = "data/ud_ewt/en_ewt-ud-train.conllu"

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

def get_sentence_lengths_from_erased(erased_list):
    return [arr.shape[0] for arr in erased_list]

def get_sentence_lengths_from_original(orig_data, layer):
    return [sent["embeddings_by_layer"][layer].shape[0] for sent in orig_data]

def pos_to_int_labels(pos_list):
    unique_pos = sorted(set(pos_list))
    pos2int = {pos: i for i, pos in enumerate(unique_pos)}
    int_labels = np.array([pos2int[pos] for pos in pos_list])
    return int_labels, pos2int

# --------------------------
# Main
# --------------------------
print("Extracting POS tags from CoNLL-U file...")
pos_sentences, _ = extract_filtered_pos_from_conllu(TRAIN_CONLLU)

print("Loading original embeddings for alignment...")
with open(ORIGINAL_EMB_FILE, "rb") as f:
    orig_data = pickle.load(f)

print("Loading selected embeddings...")
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

# Align by sentence length
if ORACLE_EMB_FILE.endswith(".pt"):
    orig_lengths = get_sentence_lengths_from_original(orig_data, LAYER)
    X_erased_flat = []
    Y_pos = []
    pos_idx = 0
    for o_len, emb_arr in zip(orig_lengths, X_erased_sents):
        while pos_idx < len(pos_sentences) and len(pos_sentences[pos_idx]) != o_len:
            pos_idx += 1
        if pos_idx == len(pos_sentences):
            break
        if o_len == len(pos_sentences[pos_idx]):
            X_erased_flat.append(emb_arr)
            Y_pos.extend(pos_sentences[pos_idx])
        pos_idx += 1
    X_erased_flat = np.vstack(X_erased_flat)
    Y_pos = np.array(Y_pos)
else:
    erased_lengths = get_sentence_lengths_from_erased(X_erased_sents)
    orig_lengths = get_sentence_lengths_from_original(orig_data, LAYER)
    X_erased_flat = []
    Y_pos = []
    pos_idx = 0
    for e_len, o_len, emb_arr in zip(erased_lengths, orig_lengths, X_erased_sents):
        while pos_idx < len(pos_sentences) and len(pos_sentences[pos_idx]) != e_len:
            pos_idx += 1
        if pos_idx == len(pos_sentences):
            break
        if e_len == o_len and e_len == len(pos_sentences[pos_idx]):
            X_erased_flat.append(emb_arr)
            Y_pos.extend(pos_sentences[pos_idx])
        pos_idx += 1
    X_erased_flat = np.vstack(X_erased_flat)
    Y_pos = np.array(Y_pos)

print(f"Aligned {X_erased_flat.shape[0]} tokens for POS probing.")

# Map POS tags to integer labels
Y_labels, pos2int = pos_to_int_labels(Y_pos)

# --------------------------
# Standardize X
# --------------------------
scaler = StandardScaler()
X_erased_scaled = scaler.fit_transform(X_erased_flat)

# Remove any rows with non-finite values in X or Y
mask = np.isfinite(X_erased_scaled).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values.")
    X_erased_scaled = X_erased_scaled[mask]
    Y_labels = Y_labels[mask]

# --------------------------
# Probe: Logistic regression (multi-class)
# --------------------------
print("Fitting linear probe for POS on selected embeddings...")
probe = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
probe.fit(X_erased_scaled, Y_labels)
Y_pred = probe.predict(X_erased_scaled)

acc = accuracy_score(Y_labels, Y_pred)
f1 = f1_score(Y_labels, Y_pred, average='macro')
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")

# Per-class F1 and support
target_names = [pos for pos, idx in sorted(pos2int.items(), key=lambda x: x[1])]
report = classification_report(Y_labels, Y_pred, target_names=target_names, digits=3)
print("\nPer-class F1 and support:\n")
print(report)

# --------------------------
# Save results
# --------------------------
os.makedirs(results_dir, exist_ok=True)
emb_name = os.path.splitext(os.path.basename(ORACLE_EMB_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_oracle_pos_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"Accuracy: {acc}\nMacro F1: {f1}\n\n")
    f.write("Per-class F1 and support:\n")
    f.write(report)
print(f"Results saved to {results_file}")