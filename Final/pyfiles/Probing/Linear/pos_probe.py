import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def pad_vector(vec, length, pad_value):
    vec = np.asarray(vec)
    return np.concatenate([vec, np.full(length - len(vec), pad_value)]) if len(vec) < length else vec[:length]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--method", choices=["leace", "oracle"], required=True)
    parser.add_argument("--layer", type=int, default=8)
    args = parser.parse_args()

    dataset = "Narratives" if args.dataset == "narratives" else "UD"
    layer = args.layer
    method = args.method

    # File mapping - original files for all concepts and pos-erased
    concept_files = {
        "original_sd": f"Final/Embeddings/Original/{dataset}/Embed_{'nar' if dataset == 'Narratives' else 'ud'}_sd.pkl",
        "original_pos": f"Final/Embeddings/Original/{dataset}/Embed_{'nar' if dataset == 'Narratives' else 'ud'}_pos.pkl",
        "original_deplabs": f"Final/Embeddings/Original/{dataset}/Embed_{'nar' if dataset == 'Narratives' else 'ud'}_deplabs.pkl",
        "pos": f"Final/Embeddings/Erased/{dataset}/{method}_{'nar' if dataset == 'Narratives' else 'ud'}_pos_vec.pkl",
    }

    # Load embeddings
    data = {}
    for concept, path in concept_files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if concept.startswith("original"):
            data[concept] = obj
        else:
            if isinstance(obj, dict) and "all_erased" in obj:
                data[concept] = obj["all_erased"]  # Use erased embeddings directly
            else:
                data[concept] = obj

    # Get all POS and DepLab tags for encoding
    all_pos_tags = set()
    all_deplabs = set()
    for sent in data["original_pos"]:
        all_pos_tags.update(sent.get("pos_tags", []))
    for sent in data["original_deplabs"]:
        all_deplabs.update(sent.get("deplabs", []))
    all_pos_tags = sorted(list(all_pos_tags))
    all_deplabs = sorted(list(all_deplabs))
    pos_to_idx = {pos: i for i, pos in enumerate(all_pos_tags)}
    deplab_to_idx = {dep: i for i, dep in enumerate(all_deplabs)}

    print("\nUnique POS tags:", all_pos_tags)
    print("Unique DepLab tags:", all_deplabs)

    # Prepare results DataFrame
    results = pd.DataFrame(index=[
        "original", "pos-erased"
    ], columns=[
        "sd probe (r²)", "sd probe (pearson r)", "sd probe (mse)",
        "pos probe (r²)", "pos probe (accuracy)",
        "deplabs probe (r²)", "deplabs probe (accuracy)",
        "l2 distance to original"
    ])

    # Get original embeddings for L2 comparison
    X_orig = []
    for sent in data["original_pos"]:
        emb = sent["embeddings_by_layer"][layer]
        X_orig.append(emb if not hasattr(emb, "numpy") else emb.numpy())
    X_orig = np.vstack(X_orig)

    # For each embedding type, probe all concepts
    for emb_type in ["original_pos", "pos"]:
        if emb_type not in data:
            continue
        print(f"\n=== Probing on {emb_type} embeddings ===")
        sent_data = data[emb_type]
        
        # Prepare X
        X = []
        for sent in sent_data:
            if emb_type == "original_pos":
                emb = sent["embeddings_by_layer"][layer]
            else:
                emb = sent  # Use erased embeddings directly
            X.append(emb if not hasattr(emb, "numpy") else emb.numpy())
        X = np.vstack(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate L2 distance for pos-erased embeddings
        if emb_type == "pos":
            l2 = np.linalg.norm(X_orig - X, axis=1).mean()
            results.loc["pos-erased", "l2 distance to original"] = l2
            print(f"l2 distance to original: {l2:.4f}")

        # SD probe (regression)
        max_len = max(sent["embeddings_by_layer"][layer].shape[0] for sent in data["original_sd"])
        pad_val = max_len - 1
        
        y_sd = []
        for sent in data["original_sd"]:  # Use original SD embeddings for distance matrix
            dist = sent.get("distance_matrix")
            if dist is not None:
                dist = dist.numpy() if hasattr(dist, "numpy") else dist
                for i in range(dist.shape[0]):
                    y_sd.append(pad_vector(dist[i], max_len, pad_val))
        
        if y_sd:
            y_sd = np.vstack(y_sd)
            ridge = Ridge(alpha=1.0).fit(X_scaled, y_sd)
            y_pred = ridge.predict(X_scaled)
            r2 = r2_score(y_sd, y_pred)
            mse = mean_squared_error(y_sd, y_pred)
            corr = pearsonr(y_sd.flatten(), y_pred.flatten())[0]
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "sd probe (r²)"] = r2
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "sd probe (pearson r)"] = corr
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "sd probe (mse)"] = mse
            print(f"sd probe: r² = {r2:.4f}, pearson r = {corr:.4f}, mse = {mse:.4f}")

        # POS probe (both regression and classification)
        y_pos = []
        for sent in data["original_pos"]:  # Use POS embeddings for tags
            pos_tags = sent.get("pos_tags", [])
            if pos_tags:  # Only add if we have POS tags
                y_pos.extend([pos_to_idx[p] for p in pos_tags])
        
        if y_pos:
            print(f"Number of POS tags for {emb_type}: {len(y_pos)}")
            # Ridge regression (R²)
            y_pos_onehot = np.zeros((len(y_pos), len(all_pos_tags)), dtype=np.float32)
            for i, idx in enumerate(y_pos):
                y_pos_onehot[i, idx] = 1.0
            ridge_pos = Ridge(alpha=1.0).fit(X_scaled, y_pos_onehot)
            y_pos_pred = ridge_pos.predict(X_scaled)
            r2_pos = r2_score(y_pos_onehot, y_pos_pred)
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "pos probe (r²)"] = r2_pos
            print(f"pos probe (ridge): r² = {r2_pos:.4f}")
            
            # Logistic regression (accuracy)
            clf_pos = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
            clf_pos.fit(X_scaled, y_pos)
            acc_pos = accuracy_score(y_pos, clf_pos.predict(X_scaled))
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "pos probe (accuracy)"] = acc_pos
            print(f"pos probe (logistic): accuracy = {acc_pos:.4f}")

        # DepLab probe (both regression and classification)
        y_dep = []
        for sent in data["original_deplabs"]:  # Use DepLab embeddings for tags
            deplabs = sent.get("deplabs", [])
            if deplabs:  # Only add if we have DepLab tags
                y_dep.extend([deplab_to_idx[d] for d in deplabs])
        
        if y_dep:
            print(f"Number of DepLab tags for {emb_type}: {len(y_dep)}")
            # Ridge regression (R²)
            y_dep_onehot = np.zeros((len(y_dep), len(all_deplabs)), dtype=np.float32)
            for i, idx in enumerate(y_dep):
                y_dep_onehot[i, idx] = 1.0
            ridge_dep = Ridge(alpha=1.0).fit(X_scaled, y_dep_onehot)
            y_dep_pred = ridge_dep.predict(X_scaled)
            r2_dep = r2_score(y_dep_onehot, y_dep_pred)
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "deplabs probe (r²)"] = r2_dep
            print(f"deplabs probe (ridge): r² = {r2_dep:.4f}")
            
            # Logistic regression (accuracy)
            clf_dep = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
            clf_dep.fit(X_scaled, y_dep)
            acc_dep = accuracy_score(y_dep, clf_dep.predict(X_scaled))
            results.loc["original" if emb_type == "original_pos" else f"{emb_type}-erased", "deplabs probe (accuracy)"] = acc_dep
            print(f"deplabs probe (logistic): accuracy = {acc_dep:.4f}")

    # Save results
    print("\nResults Table:")
    print(results.round(4))
    
    # Save results
    results.to_csv(f"Final/Results/{args.dataset}_{args.method}_pos_probe_results.csv")
    print(f"\nResults saved to Final/Results/")

if __name__ == "__main__":
    main() 