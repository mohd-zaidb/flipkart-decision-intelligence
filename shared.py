"""Shared data-loading, splitting, model, and output-formatting utilities."""
import csv
import math
import os
import numpy as np


def load_records(csv_path=os.path.join("data", "records.csv")):
    """Load data/records.csv in file order without modifying any row order."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    record_ids = [r[0] for r in rows]
    X = np.array([[float(r[j]) for j in range(1, 8)] for r in rows], dtype=np.float64)
    outcome = np.array([int(r[8]) for r in rows], dtype=np.int64)
    value = np.array([float(r[9]) for r in rows], dtype=np.float64)
    return header, record_ids, X, outcome, value


def sample_mean_sd(arr, axis=0):
    """Compute mean and sample standard deviation (divisor n - 1)."""
    n = arr.shape[axis]
    mean = np.sum(arr, axis=axis) / n
    diff = arr - np.expand_dims(mean, axis=axis) if arr.ndim > 1 else arr - mean
    var = np.sum(diff ** 2, axis=axis) / (n - 1)
    sd = np.sqrt(var)
    return mean, sd


def format_value(val):
    """Format values according to capstone results-file specification."""
    if isinstance(val, (bool, np.bool_)):
        return "yes" if bool(val) else "no"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.4f}"
    return str(val)


def write_results_txt(path, items):
    """Write key = value lines to outputs/partN_results.txt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for key, val in items:
            f.write(f"{key} = {format_value(val)}\n")


def write_csv(path, header, rows):
    """Write CSV file with exact header and 4-decimal float formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_value(v) for v in row])


def stratified_split_and_standardise(X, y):
    """Task 2.1 pinned 70/30 stratified split preserving file order, then train-only standardisation."""
    idx_0 = [i for i, val in enumerate(y) if val == 0]
    idx_1 = [i for i, val in enumerate(y) if val == 1]

    n_train_0 = (7 * len(idx_0)) // 10
    n_train_1 = (7 * len(idx_1)) // 10

    train_idx = sorted(idx_0[:n_train_0] + idx_1[:n_train_1])
    test_idx = sorted(idx_0[n_train_0:] + idx_1[n_train_1:])

    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    train_mean, train_sd = sample_mean_sd(X_train_raw, axis=0)
    X_train_std = (X_train_raw - train_mean) / train_sd
    X_test_std = (X_test_raw - train_mean) / train_sd

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "y_train": y_train,
        "y_test": y_test,
        "train_mean": train_mean,
        "train_sd": train_sd,
    }


def sigmoid(z):
    """Numerically safe logistic sigmoid."""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def binary_cross_entropy(y_true, y_prob):
    """Mean binary cross-entropy over records."""
    p = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    return float(-np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)) / len(y_true))


def train_logistic_regression(X_train, y_train, eta=0.1, epochs=1000, checkpoints=(0, 1, 10, 100, 1000)):
    """Batch gradient descent logistic regression from zero weights and bias."""
    n, d = X_train.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    checkpoint_set = set(checkpoints)
    losses = {}

    if 0 in checkpoint_set:
        p0 = sigmoid(X_train @ w + b)
        losses[0] = binary_cross_entropy(y_train, p0)

    for epoch in range(1, epochs + 1):
        z = X_train @ w + b
        p = sigmoid(z)
        err = p - y_train
        dw = (X_train.T @ err) / n
        db = float(np.sum(err) / n)
        w = w - eta * dw
        b = b - eta * db
        if epoch in checkpoint_set:
            p_after = sigmoid(X_train @ w + b)
            losses[epoch] = binary_cross_entropy(y_train, p_after)

    return w, b, losses


def confusion_matrix_counts(y_true, y_pred):
    """Compute (tp, fp, tn, fn) as Python ints."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def classification_metrics(tp, fp, tn, fn):
    """Compute accuracy, precision, recall (TPR), F1, specificity, and FPR with 0-denominator guard."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tpr": recall,
        "fpr": fpr,
    }


def knn_predict(X_ref, y_ref, X_query, k):
    """K-nearest neighbours with Euclidean distance and stable file-order tie-breaking."""
    threshold = (k + 1) // 2
    preds = np.zeros(X_query.shape[0], dtype=np.int64)
    probs = np.zeros(X_query.shape[0], dtype=np.float64)
    for i in range(X_query.shape[0]):
        diff = X_ref - X_query[i]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        order = np.argsort(dists, kind="stable")
        top_k = y_ref[order[:k]]
        pos_count = int(np.sum(top_k))
        probs[i] = pos_count / k
        preds[i] = 1 if pos_count >= threshold else 0
    return preds, probs
