"""
str_probe_oracle_deplabs.py

Structural probe for dependency labels (oracle): probes only on the training set,
using TwoWordPSDProbe and a classifier on structural features.
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
ORIGINAL_EMB_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"
TRAIN_CONLLU = "data/ud_ewt/en_ewt-ud-train.conllu"
results_dir = "Syntactic_distances/Results/UD/Oracle/SD_on_Deplab_structural/"

# 1. Oracle LEACE (no PCA)
ORACLE_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec.pkl"
PCA_OBJ_FILE = None

# 2. Oracle LEACE + PCA
# ORACLE_EMB_FILE = "Syntactic_distances/Embeddings/Erased/UD/oracle_embeddings_synt_dist_vec_pca.pkl"
# PCA_OBJ_FILE = "Syntactic_distances/Eraser_objects/UD/oracle_pca_synt_dist_vec.pkl"

# 3. Original (no erasure)
# ORACLE_EMB_FILE = ORIGINAL_EMB_FILE
# PCA_OBJ_FILE = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Helper: Extract dependency labels from CoNLL-U using the same filtering as preprocessing
# --------------------------
def extract_filtered_deprel_from_conllu(conllu_path):
    sentences = []
    with open(conllu_path, "r", encoding="utf-8") as f:
        sent = []
        text = ""
        for line in f:
            if line.startswith("# text = "):
                text = line[len("# text = "):].strip()
            if line.startswith("#") or not line.strip():
                if sent:
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
    return sentences

def get_sentence_lengths_from_embeddings(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    return [sent["embeddings_by_layer"][layer].shape[0] for sent in data]

# --------------------------
# Main
# --------------------------
print("Extracting dependency labels from CoNLL-U file...")
deprel_sentences = extract_filtered_deprel_from_conllu(TRAIN_CONLLU)

print("Loading embeddings for alignment...")
if ORACLE_EMB_FILE.endswith(".pt"):
    # Original (not erased): load as list of dicts
    with open(ORACLE_EMB_FILE, "rb") as f:
        orig_data_probe = pickle.load(f)
    X_sents = [sent["embeddings_by_layer"][LAYER] for sent in orig_data_probe]
else:
    # Erased: load as dict with "train_erased"
    with open(ORACLE_EMB_FILE, "rb") as f:
        erased = pickle.load(f)
    X_sents = erased["train_erased"]

# Align by sentence length
aligned_embs = []
aligned_labels = []
for emb, lbls in zip(X_sents, deprel_sentences):
    if emb.shape[0] == len(lbls):
        aligned_embs.append(emb)
        aligned_labels.append(lbls)
    else:
        print(f"Skipping sentence with mismatched length: emb {emb.shape[0]}, labels {len(lbls)}")

# Flatten after alignment
Y_str = [lbl for sent in aligned_labels for lbl in sent]
X_flat = [x for sent in aligned_embs for x in sent]

# Map dependency labels to integer labels
unique_labels = sorted(set(Y_str))
label2int = {lbl: i for i, lbl in enumerate(unique_labels)}
Y = np.array([label2int[lbl] for lbl in Y_str])

# --------------------------
# Standardize X
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.stack(X_flat))

# Remove any rows with non-finite values in X or Y
mask = np.isfinite(X_scaled).all(axis=1)
if not mask.all():
    print(f"Filtered out {np.sum(~mask)} rows with non-finite values.")
    X_scaled = X_scaled[mask]
    Y = Y[mask]

# --------------------------
# Structural probe: TwoWordPSDProbe
# --------------------------
probe_args = {
    'probe': {'maximum_rank': min(128, X_scaled.shape[1])},
    'model': {'hidden_dim': X_scaled.shape[1]},
    'device': DEVICE,
}
probe = TwoWordPSDProbe(probe_args)

def get_structural_features(X, probe):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        D_pred = probe(X_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
    features = D_pred.mean(axis=1, keepdims=True)
    return features

print("Extracting structural features for all data...")
struct_features = []
idx = 0
for sent in aligned_embs:
    n = sent.shape[0]
    struct_features.append(get_structural_features(X_scaled[idx:idx+n], probe))
    idx += n
struct_features = np.vstack(struct_features)

# --------------------------
# Probe: Logistic regression (multi-class) on structural features
# --------------------------
from sklearn.linear_model import LogisticRegression

print("Fitting linear classifier on structural probe features for dependency labels (oracle, train only)...")
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf.fit(struct_features, Y)
Y_pred = clf.predict(struct_features)

acc = accuracy_score(Y, Y_pred)
f1 = f1_score(Y, Y_pred, average='macro')
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")

# Per-class F1 and support
target_names = [lbl for lbl, idx in sorted(label2int.items(), key=lambda x: x[1])]
labels = list(range(len(target_names)))
report = classification_report(Y, Y_pred, target_names=target_names, labels=labels, digits=3, zero_division=0)
print("\nPer-class F1 and support:\n")
print(report)

# --------------------------
# Save results
# --------------------------
os.makedirs(results_dir, exist_ok=True)
emb_name = os.path.splitext(os.path.basename(ORACLE_EMB_FILE))[0]
results_file = os.path.join(results_dir, f"probe_results_structural_oracle_deplab_{emb_name}.txt")

with open(results_file, "w") as f:
    f.write(f"Accuracy: {acc}\nMacro F1: {f1}\n\n")
    f.write("Per-class F1 and support:\n")
    f.write(report)
print(f"Results saved to {results_file}")