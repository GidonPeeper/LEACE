import pickle
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# --------------------------
# Settings
# --------------------------
LAYER = 8
EMBEDDING_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_train_synt_dist.pt"
TEST_FILE = "Syntactic_distances/Embeddings/Original/UD/gpt2_embeddings_test_synt_dist.pt"
TRAIN_CONLLU = "data/ud_ewt/en_ewt-ud-train.conllu"
TEST_CONLLU = "data/ud_ewt/en_ewt-ud-test.conllu"
results_dir = "Syntactic_distances/Results/UD/LEACE/SD_on_ALL/"

# --------------------------
# Label extraction
# --------------------------
def extract_labels(conllu_path):
    pos_sentences, deplab_sentences = [], []
    sent_pos, sent_deplab = [], []
    with open(conllu_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# text = ") or line.strip() == "":
                if sent_pos and sent_deplab:
                    pos_sentences.append(sent_pos)
                    deplab_sentences.append(sent_deplab)
                sent_pos, sent_deplab = [], []
                continue
            fields = line.strip().split("\t")
            if len(fields) > 7 and fields[0].isdigit():
                sent_pos.append(fields[3])
                sent_deplab.append(fields[7])
        if sent_pos and sent_deplab:
            pos_sentences.append(sent_pos)
            deplab_sentences.append(sent_deplab)
    return pos_sentences, deplab_sentences

# --------------------------
# Embedding utilities
# --------------------------
def get_sentence_lengths(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    return [sent["embeddings_by_layer"][layer].shape[0] for sent in data]

def align_labels(labels, lengths):
    aligned = []
    idx = 0
    for length in lengths:
        while idx < len(labels) and len(labels[idx]) != length:
            idx += 1
        if idx < len(labels):
            aligned.append(labels[idx])
            idx += 1
    return aligned

def load_embeddings(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    return [sent["embeddings_by_layer"][layer] for sent in data]

def flatten_embeddings_labels(emb_sents, label_sents):
    X, Y = [], []
    for emb, labels in zip(emb_sents, label_sents):
        if emb.shape[0] != len(labels):
            continue
        for i in range(emb.shape[0]):
            X.append(emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i])
            Y.append(labels[i])
    return np.stack(X), np.array(Y)

def pad_vector(vec, length, pad_value):
    vec = np.asarray(vec)
    return np.concatenate([vec, np.full(length - len(vec), pad_value)]) if len(vec) < length else vec[:length]

def load_distances(embedding_file, layer, max_len, pad_val):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    X, Z = [], []
    for sent in data:
        emb = sent["embeddings_by_layer"][layer]
        dist = sent.get("distance_matrix")
        for i in range(emb.shape[0]):
            X.append(emb[i].numpy())
            Z.append(pad_vector(dist[i], max_len, pad_val))
    return np.stack(X), np.stack(Z)

def encode_labels(labels):
    unique = sorted(set(labels))
    lbl2id = {lbl: i for i, lbl in enumerate(unique)}
    return np.array([lbl2id[lbl] for lbl in labels]), lbl2id

# --------------------------
# Main
# --------------------------
print("Extracting labels...")
train_pos, train_dep = extract_labels(TRAIN_CONLLU)
test_pos, test_dep = extract_labels(TEST_CONLLU)

print("Checking alignment with embeddings...")
train_lengths = get_sentence_lengths(EMBEDDING_FILE, LAYER)
test_lengths = get_sentence_lengths(TEST_FILE, LAYER)
aligned_train_pos = align_labels(train_pos, train_lengths)
aligned_test_pos = align_labels(test_pos, test_lengths)
aligned_train_dep = align_labels(train_dep, train_lengths)
aligned_test_dep = align_labels(test_dep, test_lengths)

print("Loading embeddings...")
train_embeds = load_embeddings(EMBEDDING_FILE, LAYER)
test_embeds = load_embeddings(TEST_FILE, LAYER)

print("Loading POS and dep labels...")
X_train_pos, Y_train_pos_str = flatten_embeddings_labels(train_embeds, aligned_train_pos)
X_test_pos, Y_test_pos_str = flatten_embeddings_labels(test_embeds, aligned_test_pos)
X_train_dep, Y_train_dep_str = flatten_embeddings_labels(train_embeds, aligned_train_dep)
X_test_dep, Y_test_dep_str = flatten_embeddings_labels(test_embeds, aligned_test_dep)

print("Standardizing each task separately...")
scaler_pos = StandardScaler().fit(X_train_pos)
scaler_dep = StandardScaler().fit(X_train_dep)
X_train_pos = scaler_pos.transform(X_train_pos)
X_test_pos = scaler_pos.transform(X_test_pos)
X_train_dep = scaler_dep.transform(X_train_dep)
X_test_dep = scaler_dep.transform(X_test_dep)

print("Encoding labels...")
Y_train_pos, pos2id = encode_labels(Y_train_pos_str)
Y_test_pos = np.array([pos2id.get(l, -1) for l in Y_test_pos_str])
Y_train_dep, dep2id = encode_labels(Y_train_dep_str)
Y_test_dep = np.array([dep2id.get(l, -1) for l in Y_test_dep_str])

# Filter out unseen labels
mask_pos = Y_test_pos != -1
mask_dep = Y_test_dep != -1
X_test_pos, Y_test_pos = X_test_pos[mask_pos], Y_test_pos[mask_pos]
X_test_dep, Y_test_dep = X_test_dep[mask_dep], Y_test_dep[mask_dep]

# POS Logistic Regression
print("Training POS probe...")
clf_pos = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf_pos.fit(X_train_pos, Y_train_pos)
acc_pos = accuracy_score(Y_test_pos, clf_pos.predict(X_test_pos))
print(f"POS accuracy: {acc_pos:.4f}")

# Dep Label Logistic Regression
print("Training dependency label probe...")
clf_dep = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf_dep.fit(X_train_dep, Y_train_dep)
acc_dep = accuracy_score(Y_test_dep, clf_dep.predict(X_test_dep))
print(f"Dependency label accuracy: {acc_dep:.4f}")

# Ridge regression for syntactic distances
print("Preparing syntactic distance probe...")
max_len = max(train_lengths + test_lengths)
pad_val = max_len - 1
X_train_sd, Z_train_sd = load_distances(EMBEDDING_FILE, LAYER, max_len, pad_val)
X_test_sd, Z_test_sd = load_distances(TEST_FILE, LAYER, max_len, pad_val)
scaler_sd = StandardScaler().fit(X_train_sd)
X_train_sd = scaler_sd.transform(X_train_sd)
X_test_sd = scaler_sd.transform(X_test_sd)

print("Training Ridge regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sd, Z_train_sd)
r2_sd = r2_score(Z_test_sd, ridge.predict(X_test_sd))
print(f"Syntactic distance R^2: {r2_sd:.4f}")

# --------------------------
# Save results
# --------------------------
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "probe_results_all.txt")
with open(out_path, "w") as f:
    f.write("=== Probe Results Table ===\n")
    f.write(f"{'Concept':<20} {'Score':<10}\n")
    f.write(f"{'-'*30}\n")
    f.write(f"{'Syntactic distance':<20} {r2_sd:.4f} (R^2)\n")
    f.write(f"{'POS':<20} {acc_pos:.4f} (accuracy)\n")
    f.write(f"{'Dependency label':<20} {acc_dep:.4f} (accuracy)\n")

print(f"Results saved to {out_path}")
