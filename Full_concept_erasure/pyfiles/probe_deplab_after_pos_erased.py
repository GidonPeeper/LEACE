import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import os
import json

# --------------------------
# Settings
# --------------------------
EMBEDDING_FILE = "Stage4/Embeddings/UD/Synt_deps/Original_embeddings/gpt2_embeddings.pt"
RESULTS_FILE = "Full_concept_erasure/Results/probe_deplab_after_pos_erased.json"
PROJECTION_FILE = "Full_concept_erasure/Eraser_objects/oracle_leace_transf_matrix_ALL_POS.pkl"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
SEED = 42
torch.manual_seed(SEED)

# --------------------------
# Load original embeddings and dependency labels
# --------------------------
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

LAYER = 8
X_all = [sent["embeddings_by_layer"][LAYER] for sent in data]
X = torch.cat(X_all, dim=0)

all_deplabs = sorted(list(data[0]["word_labels"].keys()))
labels_by_deplab = {
    feat: torch.tensor([label for sent in data for label in sent["word_labels"][feat]]).long()
    for feat in all_deplabs
}
y_deplab = torch.stack([labels_by_deplab[feat] for feat in all_deplabs], dim=1).argmax(dim=1)

# --------------------------
# Load Oracle LEACE projection matrix for POS
# --------------------------
with open(PROJECTION_FILE, "rb") as f:
    projection = pickle.load(f)  # shape: (num_features, num_tokens)

# --------------------------
# Apply the transformation to erase POS concept
# --------------------------
X_np = X.numpy()
X_erased_np = X_np - projection.T  # projection shape: (num_features, num_tokens)
X_erased = torch.from_numpy(X_erased_np)

# --------------------------
# Probe: dependency label prediction (train only, since Oracle LEACE is in-sample)
# --------------------------
scaler = StandardScaler()
X_erased_np_scaled = scaler.fit_transform(X_erased.numpy())

clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf.fit(X_erased_np_scaled, y_deplab.numpy())
train_acc = clf.score(X_erased_np_scaled, y_deplab.numpy())
print(f"Dependency label accuracy (train set, POS concept erased, oracle): {train_acc:.4f}")

# Baseline: most frequent class
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_erased_np_scaled, y_deplab.numpy())
baseline_acc = dummy.score(X_erased_np_scaled, y_deplab.numpy())
print(f"Baseline (most frequent dependency label): {baseline_acc:.4f}")

# Classification report
labels = list(range(len(all_deplabs)))
y_pred = clf.predict(X_erased_np_scaled)
print(classification_report(y_deplab.numpy(), y_pred, labels=labels, target_names=all_deplabs))

# Save results
results = {
    "deplab_accuracy_train": train_acc,
    "baseline_accuracy": baseline_acc,
    "classification_report": classification_report(y_deplab.numpy(), y_pred, labels=labels, target_names=all_deplabs, output_dict=True)
}
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE}")
