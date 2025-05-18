import argparse
import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data(embedding_file, layer_idx, feature_name):
    """
    Load word-level embeddings and corresponding binary labels for a specific layer and feature.

    Args:
        embedding_file (str): Path to the .pt file containing saved outputs.
        layer_idx (int): Index of the GPT-2 layer to extract.
        feature_name (str): Feature to probe (e.g., "function_content").

    Returns:
        X (np.ndarray): Array of word embeddings.
        y (np.ndarray): Array of binary labels.
    """
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)

    X = []
    y = []

    for sample in data:
        embeddings = sample["embeddings_by_layer"][layer_idx]  # Tensor [num_words, dim]
        labels = sample["word_labels"][feature_name]           # List[int]

        if embeddings.shape[0] != len(labels):
            raise ValueError("Mismatch between number of word embeddings and labels.")

        X.append(embeddings.numpy())
        y.append(np.array(labels))

    return np.vstack(X), np.hstack(y)

def train_and_evaluate(X, y, test_size=0.2, seed=42):
    """
    Train a logistic regression probe and evaluate it on a held-out test set.

    Args:
        X (np.ndarray): Word embeddings.
        y (np.ndarray): Binary feature labels.
        test_size (float): Proportion of data to hold out for testing.
        seed (int): Random seed for reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a linear probe on GPT-2 word-level embeddings.")
    parser.add_argument("--embedding-file", type=str, required=True, help="Path to saved .pt embeddings")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to probe (0–12 for GPT-2 small)")
    parser.add_argument("--feature", type=str, required=True, help="Feature name to probe (e.g., 'function_content')")
    args = parser.parse_args()

    print(f"Loading embeddings from: {args.embedding_file}")
    X, y = load_data(args.embedding_file, args.layer, args.feature)

    print(f"Training and evaluating probe on layer {args.layer} for feature '{args.feature}'")
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
