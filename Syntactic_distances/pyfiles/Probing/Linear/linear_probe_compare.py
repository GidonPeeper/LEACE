"""
linear_probe_compare.py

This script evaluates how well syntactic distance information can be recovered from
GPT-2 embeddings using linear probes (Ridge regression), under different conditions:
- Original embeddings
- LEACE-erased embeddings
- Oracle LEACE-erased embeddings

For each, it computes (on TRAINING DATA ONLY):
- R² score (variance explained)
- Pearson correlation
- Mean squared error (MSE)
- POS accuracy (Logistic regression)
- Dependency label accuracy (Logistic regression)

All evaluations are done on the **training set** for consistent comparison.
"""

import pickle
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os

LAYER = 8

probe_configs = [
    {
        "name": "original",
        "train_emb": "Syntactic_distances/Embeddings/Original/Narratives/gpt2_embeddings_train_synt_dist.pt",
    },
    {
        "name": "leace",
        "train_emb": "Syntactic_distances/Embeddings/Erased/Narratives/leace_embeddings_synt_dist_vec.pkl",
    },
    {
        "name": "oracle_leace",
        "train_emb": "Syntactic_distances/Embeddings/Erased/Narratives/oracle_embeddings_synt_dist_vec.pkl",
    }
]

TRAIN_CONLLU = "data/narratives/train_clean.conllu"
results_dir = "Syntactic_distances/Results/Narratives/LEACE/SD_on_SD_compare/"
os.makedirs(results_dir, exist_ok=True)

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

def get_max_sentence_length(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        arrs = data.get("train_erased", [])
        return max(arr.shape[0] for arr in arrs)
    else:
        return max(sent["embeddings_by_layer"][layer].shape[0] for sent in data)

def load_embeddings_and_distances(embedding_file, layer):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):  # LEACE/Oracle format
        embs = [np.array(e) for e in data["train_erased"]]
        return embs, None
    else:
        embs = [sent["embeddings_by_layer"][layer] for sent in data]
        dists = [sent["distance_matrix"] for sent in data]
        return embs, dists

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

def pad_vector(vec, length, pad_value):
    vec = np.asarray(vec)
    return np.concatenate([vec, np.full(length - len(vec), pad_value)]) if len(vec) < length else vec[:length]

def flatten_embeddings(emb_sents):
    return np.stack([emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i]
                     for emb in emb_sents for i in range(emb.shape[0])])

def flatten_labels(label_sents):
    return [lbl for sent in label_sents for lbl in sent]

def flatten_distances(dist_sents, max_len, pad_val):
    return np.stack([
        pad_vector(dist[i], max_len, pad_val)
        for dist in dist_sents for i in range(dist.shape[0])
    ])

def encode_labels(labels):
    unique = sorted(set(labels))
    lbl2id = {lbl: i for i, lbl in enumerate(unique)}
    return np.array([lbl2id[lbl] for lbl in labels]), lbl2id

def probe_all(X, Z, Y_pos, Y_dep, probe_name, results_dir):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SD probe
    mask = np.isfinite(X_scaled).all(axis=1) & np.isfinite(Z).all(axis=1)
    X_clean = X_scaled[mask]
    Z_clean = Z[mask]
    ridge = Ridge(alpha=1.0).fit(X_clean, Z_clean)
    Z_pred = ridge.predict(X_clean)
    r2 = r2_score(Z_clean, Z_pred)
    mse = mean_squared_error(Z_clean, Z_pred)
    corr = pearsonr(Z_clean.flatten(), Z_pred.flatten())[0]

    # POS probe
    Y_pos_enc, pos2id = encode_labels(Y_pos)
    clf_pos = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    clf_pos.fit(X_scaled, Y_pos_enc)
    acc_pos = accuracy_score(Y_pos_enc, clf_pos.predict(X_scaled))

    # Dep probe
    Y_dep_enc, dep2id = encode_labels(Y_dep)
    clf_dep = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    clf_dep.fit(X_scaled, Y_dep_enc)
    acc_dep = accuracy_score(Y_dep_enc, clf_dep.predict(X_scaled))

    # Save
    with open(os.path.join(results_dir, f"probe_results_{probe_name}.txt"), "w") as f:
        f.write("=== Probe Results (TRAIN only) ===\n")
        f.write(f"{'Concept':<20} {'Score':<20}\n")
        f.write(f"{'-'*40}\n")
        f.write(f"{'Syntactic distance':<20} R^2 = {r2:.4f}\n")
        f.write(f"{'':<20} PearsonR = {corr:.4f}\n")
        f.write(f"{'':<20} MSE = {mse:.4f}\n")
        f.write(f"{'POS':<20} Accuracy = {acc_pos:.4f}\n")
        f.write(f"{'Dependency label':<20} Accuracy = {acc_dep:.4f}\n")

    print(f"\n{probe_name.capitalize()} (train set only):")
    print(f"  Syntactic distance -> R^2 = {r2:.4f}, Pearson R = {corr:.4f}, MSE = {mse:.4f}")
    print(f"  POS accuracy       -> {acc_pos:.4f}")
    print(f"  Dep label accuracy -> {acc_dep:.4f}")

# --------------------------
# Main loop
# --------------------------
train_pos_labels, train_dep_labels = extract_labels(TRAIN_CONLLU)

for cfg in probe_configs:
    print(f"\n=== Probing: {cfg['name']} ===")
    max_len = get_max_sentence_length(cfg["train_emb"], LAYER)
    pad_val = max_len - 1

    emb_sents, dist_sents = load_embeddings_and_distances(cfg["train_emb"], LAYER)
    sent_lengths = [emb.shape[0] for emb in emb_sents]
    aligned_pos = align_labels(train_pos_labels, sent_lengths)
    aligned_dep = align_labels(train_dep_labels, sent_lengths)

    X = flatten_embeddings(emb_sents)
    Y_pos = flatten_labels(aligned_pos)
    Y_dep = flatten_labels(aligned_dep)

    if dist_sents is not None:
        Z = flatten_distances(dist_sents, max_len, pad_val)
    else:
        # Align original gold distances to erased embeddings by sentence length
        orig_embs, orig_dists = load_embeddings_and_distances(probe_configs[0]["train_emb"], LAYER)
        orig_lengths = [emb.shape[0] for emb in orig_embs]
        erased_lengths = [emb.shape[0] for emb in emb_sents]
        match_idx = []
        orig_idx = 0
        for e_len in erased_lengths:
            while orig_idx < len(orig_lengths) and orig_lengths[orig_idx] != e_len:
                orig_idx += 1
            if orig_idx == len(orig_lengths):
                break
            match_idx.append(orig_idx)
            orig_idx += 1
        # Only keep matching sentences
        matched_orig_dists = [orig_dists[i] for i in match_idx]
        Z = flatten_distances(matched_orig_dists, max_len, pad_val)
        # Also align emb_sents for X, Y_pos, Y_dep
        emb_sents = [emb_sents[i] for i in range(len(match_idx))]
        aligned_pos = [aligned_pos[i] for i in range(len(match_idx))]
        aligned_dep = [aligned_dep[i] for i in range(len(match_idx))]
        X = flatten_embeddings(emb_sents)
        Y_pos = flatten_labels(aligned_pos)
        Y_dep = flatten_labels(aligned_dep)

    # Optional: print L2 diff from original embeddings
    if cfg["name"] != "original":
        # Align original and erased embeddings by sentence length before flattening
        orig_embs, _ = load_embeddings_and_distances(probe_configs[0]["train_emb"], LAYER)
        orig_lengths = [emb.shape[0] for emb in orig_embs]
        erased_lengths = [emb.shape[0] for emb in emb_sents]
        match_idx = []
        orig_idx = 0
        for e_len in erased_lengths:
            while orig_idx < len(orig_lengths) and orig_lengths[orig_idx] != e_len:
                orig_idx += 1
            if orig_idx == len(orig_lengths):
                break
            match_idx.append(orig_idx)
            orig_idx += 1
        # Only keep matching sentences
        matched_orig_embs = [orig_embs[i] for i in match_idx]
        matched_erased_embs = [emb_sents[i] for i in range(len(match_idx))]
        flat_orig = np.vstack([emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i]
                               for emb in matched_orig_embs for i in range(emb.shape[0])])
        flat_erased = np.vstack([emb[i].numpy() if hasattr(emb[i], "numpy") else emb[i]
                                 for emb in matched_erased_embs for i in range(emb.shape[0])])
        if flat_orig.shape != flat_erased.shape:
            print(f"  ⚠️ Skipping Δ L2: shape mismatch {flat_orig.shape} vs {flat_erased.shape}")
        else:
            delta = np.linalg.norm(flat_orig - flat_erased) / flat_orig.shape[0]
            print(f"  Δ mean L2 (original vs {cfg['name']}): {delta:.4f}")

    probe_all(X, Z, Y_pos, Y_dep, cfg["name"], results_dir)

print(f"\n✅ All results saved to {results_dir}")

