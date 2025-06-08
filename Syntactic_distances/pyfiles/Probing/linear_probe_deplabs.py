"""
linear_probe_dep.py

This script probes to what extent the concept of dependency labels is present in word embeddings.
It works for original, LEACE-erased, and PCA-erased embeddings.

Inputs:
    - Pickled embeddings (train and test), each as a list of [num_words, hidden_dim] arrays
    - Dependency labels for each word (from the original CoNLL-U files, filtered with the same logic as in preprocessing)
    - (Optional) PCA object, if probing on PCA-reduced concept

Outputs:
    - Accuracy and macro F1 for the probe on the test set
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Syntactic_distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Syntactic_distances/Embeddings/Original/Narratives/gpt2_embeddings_test_synt_dist.pt"
TRAIN_CONLLU = "data/narratives/train_clean.conllu"
TEST_CONLLU = "data/narratives/test_clean.conllu"

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
# Helper: Extract dependency labels from CoNLL-U using the same filtering as preprocessing
# --------------------------
def extract_filtered_deprel_from_conllu(conllu_path):
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
                    valid_tokens = [lbl for lbl in sent if lbl.strip()]
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
            if len(fields) > 7:
                sent.append(fields[7])  # DEPREL
        if sent:
            valid_tokens = [lbl for lbl in sent if lbl.strip()]
            if len(valid_tokens) == 0:
                pass
            elif len(valid_tokens) == 1 and valid_tokens[0].strip() == text.strip():
                pass
            else:
                sentences.append(sent)
                texts.append(text)
    return sentences, texts

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
        # Skip label sentences until we find one with the right length
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

def label_to_int_labels(label_list):
    unique_labels = sorted(set(label_list))
    label2int = {lbl: i for i, lbl in enumerate(unique_labels)}
    int_labels = np.array([label2int[lbl] for lbl in label_list])
    return int_labels, label2int

# --------------------------
# Main
# --------------------------
print("Extracting dependency labels from CoNLL-U files...")
train_deprel_sentences, _ = extract_filtered_deprel_from_conllu(TRAIN_CONLLU)
test_deprel_sentences, _ = extract_filtered_deprel_from_conllu(TEST_CONLLU)

print("Getting sentence lengths from original embeddings...")
train_emb_lengths = get_sentence_lengths_from_embeddings(EMBEDDING_FILE, LAYER)
test_emb_lengths = get_sentence_lengths_from_embeddings(TEST_FILE, LAYER)

print("Aligning dependency labels to embeddings...")
aligned_train_deprel = align_labels_to_embeddings(train_deprel_sentences, train_emb_lengths)
aligned_test_deprel = align_labels_to_embeddings(test_deprel_sentences, test_emb_lengths)

print("Loading train embeddings and dependency labels...")
X_train, Y_train_str = load_embeddings_and_labels(EMB_PROBE_FILE, LAYER, aligned_train_deprel, erased_key="train_erased")
print("Loading test embeddings and dependency labels...")
X_test, Y_test_str = load_embeddings_and_labels(EMB_PROBE_TEST_FILE, LAYER, aligned_test_deprel, erased_key="test_erased")

# If probing on PCA-reduced concept, just use the embeddings as is (no change to Y)
if PCA_OBJ_FILE is not None:
    import pickle
    with open(PCA_OBJ_FILE, "rb") as f:
        pca = pickle.load(f)
    # No change to Y_train/Y_test for dependency labels

# Map dependency labels to integer labels (fit on train, apply to test)
Y_train, label2int = label_to_int_labels(Y_train_str)
Y_test = np.array([label2int.get(lbl, -1) for lbl in Y_test_str])
mask_valid = Y_test != -1
if not mask_valid.all():
    print(f"Warning: {np.sum(~mask_valid)} dependency labels in test set not seen in train set. These will be ignored.")
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
print("Fitting linear probe for dependency labels...")
probe = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
probe.fit(X_train_scaled, Y_train)
Y_pred = probe.predict(X_test_scaled)

acc = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='macro')
print(f"Accuracy on test set: {acc:.4f}")
print(f"Macro F1 on test set: {f1:.4f}")

# Per-class F1 and support
target_names = [lbl for lbl, idx in sorted(label2int.items(), key=lambda x: x[1])]
report = classification_report(Y_test, Y_pred, target_names=target_names, digits=3)
print("\nPer-class F1 and support:\n")
print(report)

# Optionally, print confusion matrix
# cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion matrix:\n", cm)

# --------------------------
# Save results with informative filename
# --------------------------
results_dir = "Syntactic_distances/Results/Narratives/LEACE/SD_on_Deplab"
os.makedirs(results_dir, exist_ok=True)

emb_name = os.path.splitext(os.path.basename(EMB_PROBE_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_deplab_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"Accuracy: {acc}\nMacro F1: {f1}\n\n")
    f.write("Per-class F1 and support:\n")
    f.write(report)
print(f"Results saved to {results_file}")