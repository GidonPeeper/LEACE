import argparse
import pickle
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def probe_r2(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return r2, mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narratives", "ud"], required=True)
    parser.add_argument("--method", choices=["leace", "oracle"], required=True)
    parser.add_argument("--layer", type=int, default=8)
    args = parser.parse_args()

    dataset = "Narratives" if args.dataset == "narratives" else "UD"
    layer = args.layer
    method = args.method

    # File mapping
    concept_files = {
        "original": f"Final/Embeddings/Original/{dataset}/Embed_{'nar' if dataset == 'Narratives' else 'ud'}_sd.pkl",
        "sd": f"Final/Embeddings/Erased/{dataset}/{method}_{'nar' if dataset == 'Narratives' else 'ud'}_sd_vec.pkl",
        "pos": f"Final/Embeddings/Erased/{dataset}/{method}_{'nar' if dataset == 'Narratives' else 'ud'}_pos_vec.pkl",
        "deplabs": f"Final/Embeddings/Erased/{dataset}/{method}_{'nar' if dataset == 'Narratives' else 'ud'}_deplabs_vec.pkl",
    }

    # Load all embeddings
    data = {}
    for concept, path in concept_files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if concept == "original":
            data[concept] = obj
        else:
            if isinstance(obj, dict) and "all_erased" in obj:
                orig = data["original"]
                all_erased = obj["all_erased"]
                for i, sent in enumerate(orig):
                    sent[f"{concept}_erased"] = all_erased[i]
                data[concept] = orig
            else:
                data[concept] = obj

    # Get all POS and DepLab tags for encoding
    all_pos_tags = set()
    all_deplabs = set()
    for sent in data["original"]:
        all_pos_tags.update(sent.get("pos_tags", []))
        all_deplabs.update(sent.get("deplabs", []))
    all_pos_tags = sorted(list(all_pos_tags))
    all_deplabs = sorted(list(all_deplabs))
    pos_to_idx = {pos: i for i, pos in enumerate(all_pos_tags)}
    deplab_to_idx = {dep: i for i, dep in enumerate(all_deplabs)}

    # Prepare results
    results = {}

    # For each embedding type, probe all concepts (all as regression)
    for emb_type in ["original", "sd", "pos", "deplabs"]:
        if emb_type not in data:
            continue
        print(f"\n=== Probing on {emb_type.upper()} embeddings ===")
        sent_data = data[emb_type]
        # Prepare X
        X = []
        for sent in sent_data:
            if emb_type == "original":
                emb = sent["embeddings_by_layer"][layer]
            else:
                emb = sent.get(f"{emb_type}_erased", sent["embeddings_by_layer"][layer])
            X.append(emb if not hasattr(emb, "numpy") else emb.numpy())
        X = np.vstack(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # SD probe (regression)
        y_sd = []
        for sent in sent_data:
            dist = sent.get("distance_matrix")
            if dist is not None:
                for i in range(dist.shape[0]):
                    y_sd.append(dist[i].flatten())
        y_sd = np.vstack(y_sd)
        r2_sd, mse_sd = probe_r2(X_scaled, y_sd)
        results[f"{emb_type}_sd_r2"] = r2_sd
        results[f"{emb_type}_sd_mse"] = mse_sd
        print(f"SD probe: R^2 = {r2_sd:.4f}, MSE = {mse_sd:.2f}")

        # POS probe (one-hot regression)
        y_pos = []
        for sent in sent_data:
            y_pos.extend([pos_to_idx[p] for p in sent.get("pos_tags", [])])
        if y_pos:
            y_pos_onehot = np.zeros((len(y_pos), len(all_pos_tags)), dtype=np.float32)
            for i, idx in enumerate(y_pos):
                y_pos_onehot[i, idx] = 1.0
            r2_pos, mse_pos = probe_r2(X_scaled, y_pos_onehot)
            results[f"{emb_type}_pos_r2"] = r2_pos
            results[f"{emb_type}_pos_mse"] = mse_pos
            print(f"POS probe: R^2 = {r2_pos:.4f}, MSE = {mse_pos:.2f}")

        # DepLab probe (one-hot regression)
        y_dep = []
        for sent in sent_data:
            y_dep.extend([deplab_to_idx[d] for d in sent.get("deplabs", [])])
        if y_dep:
            y_dep_onehot = np.zeros((len(y_dep), len(all_deplabs)), dtype=np.float32)
            for i, idx in enumerate(y_dep):
                y_dep_onehot[i, idx] = 1.0
            r2_dep, mse_dep = probe_r2(X_scaled, y_dep_onehot)
            results[f"{emb_type}_deplab_r2"] = r2_dep
            results[f"{emb_type}_deplab_mse"] = mse_dep
            print(f"DepLab probe: R^2 = {r2_dep:.4f}, MSE = {mse_dep:.2f}")

        # L2 distance (only for erased embeddings)
        if emb_type != "original":
            X_orig = []
            for sent in data["original"]:
                emb = sent["embeddings_by_layer"][layer]
                X_orig.append(emb if not hasattr(emb, "numpy") else emb.numpy())
            X_orig = np.vstack(X_orig)
            l2 = np.linalg.norm(X_orig - X, axis=1).mean()
            results[f"{emb_type}_l2"] = l2
            print(f"L2 distance to original: {l2:.4f}")

    # Save results
    out_file = f"Final/Results/{args.dataset}_{args.method}_probe_r2_results.json"
    with open(out_file, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()