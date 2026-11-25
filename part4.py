"""Part 4: Model-choice, decision trees, voting, AdaBoost, OOB, and neural network sizing."""
import math
import os
import numpy as np
from shared import (
    classification_metrics,
    confusion_matrix_counts,
    knn_predict,
    load_records,
    sigmoid,
    stratified_split_and_standardise,
    train_logistic_regression,
    write_csv,
    write_results_txt,
)


def node_entropy(positives, total):
    """Binary entropy in bits (0 for a pure or empty node)."""
    if total == 0 or positives == 0 or positives == total:
        return 0.0
    p1 = positives / total
    p0 = 1.0 - p1
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))


def node_gini(positives, total):
    """Binary Gini impurity (0 for empty node)."""
    if total == 0:
        return 0.0
    p1 = positives / total
    p0 = 1.0 - p1
    return 1.0 - (p1 ** 2 + p0 ** 2)


def adaboost_sign(val):
    """Sign function returning +1 or -1 (0 if exactly 0)."""
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def main():
    _, _, X, y, _ = load_records()
    split = stratified_split_and_standardise(X, y)
    X_train_raw = split["X_train_raw"]
    X_test_raw = split["X_test_raw"]
    X_train_std = split["X_train_std"]
    X_test_std = split["X_test_std"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    n_train = len(y_train)
    pos_train = int(np.sum(y_train))

    # Task 4.1: Split purity on 560 unstandardised training rows
    root_entropy = node_entropy(pos_train, n_train)
    root_gini = node_gini(pos_train, n_train)

    split_specs = [
        ("S1", lambda X_mat: X_mat[:, 5] == 1),
        ("S2", lambda X_mat: X_mat[:, 6] == 1),
        ("S3", lambda X_mat: X_mat[:, 2] > 0.5),
        ("S4", lambda X_mat: X_mat[:, 0] <= 24.0),
    ]

    split_rows = []
    split_rates = {}
    best_ig = -1.0
    best_split_ig = None
    best_gg = -1.0
    best_split_gg = None

    for name, cond_fn in split_specs:
        yes_mask = cond_fn(X_train_raw)
        no_mask = ~yes_mask
        n_yes = int(np.sum(yes_mask))
        pos_yes = int(np.sum(y_train[yes_mask]))
        n_no = int(np.sum(no_mask))
        pos_no = int(np.sum(y_train[no_mask]))

        w_ent = (n_yes / n_train) * node_entropy(pos_yes, n_yes) + (n_no / n_train) * node_entropy(pos_no, n_no)
        w_gini = (n_yes / n_train) * node_gini(pos_yes, n_yes) + (n_no / n_train) * node_gini(pos_no, n_no)
        info_gain = root_entropy - w_ent
        gini_gain = root_gini - w_gini

        split_rows.append([name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain])
        split_rates[name] = (pos_yes / n_yes, pos_no / n_no)

        if info_gain > best_ig:
            best_ig = info_gain
            best_split_ig = name
        if gini_gain > best_gg:
            best_gg = gini_gain
            best_split_gg = name

    write_csv(
        os.path.join("outputs", "part4_splits.csv"),
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        split_rows,
    )

    # Task 4.2: One-question stump + voting on 240 test rows
    stump_rate_yes, stump_rate_no = split_rates[best_split_ig]
    best_cond_fn = dict(split_specs)[best_split_ig]

    # Logistic (Task 2.2)
    w_log, b_log, _ = train_logistic_regression(X_train_std, y_train, eta=0.1, epochs=1000, checkpoints=())
    p_log = sigmoid(X_test_std @ w_log + b_log)
    pred_log = (p_log >= 0.3).astype(np.int64)

    # KNN (Task 2.7 best K via 5-fold CV)
    folds = np.arange(n_train) % 5
    best_k = None
    best_cv_f1 = -1.0
    for k in range(1, 22, 2):
        fold_f1s = []
        for f_idx in range(5):
            val_mask = folds == f_idx
            ref_mask = ~val_mask
            preds_val, _ = knn_predict(X_train_std[ref_mask], y_train[ref_mask], X_train_std[val_mask], k)
            tp, fp, tn, fn = confusion_matrix_counts(y_train[val_mask], preds_val)
            fold_f1s.append(classification_metrics(tp, fp, tn, fn)["f1"])
        mean_f1 = sum(fold_f1s) / 5.0
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_k = k
    pred_knn, p_knn = knn_predict(X_train_std, y_train, X_test_std, best_k)

    # Stump
    test_yes_mask = best_cond_fn(X_test_raw)
    p_stump = np.where(test_yes_mask, stump_rate_yes, stump_rate_no)
    pred_stump = (p_stump >= 0.3).astype(np.int64)

    # Hard & soft votes
    pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(np.int64)
    p_soft = (p_log + p_knn + p_stump) / 3.0
    pred_soft = (p_soft >= 0.3).astype(np.int64)

    voting_rows = []
    for model_name, pred_arr in [
        ("logistic", pred_log),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard),
        ("soft_vote", pred_soft),
    ]:
        tp, fp, tn, fn = confusion_matrix_counts(y_test, pred_arr)
        m = classification_metrics(tp, fp, tn, fn)
        voting_rows.append([model_name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])

    write_csv(
        os.path.join("outputs", "part4_voting.csv"),
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )

    # Task 4.3: Two rounds of AdaBoost by hand
    y_ada = [1, 1, 1, -1, -1, -1, 1, -1]
    h1 = [1, 1, -1, -1, -1, -1, -1, -1]
    h2 = [-1, 1, 1, -1, 1, -1, 1, -1]
    weights = [1.0 / 8.0] * 8
    ada_rows = []
    round_stats = {}

    for r_idx, h in [(1, h1), (2, h2)]:
        eps = sum(w_i for w_i, y_i, h_i in zip(weights, y_ada, h) if h_i != y_i)
        alpha = 0.5 * math.log((1.0 - eps) / eps)
        unnorm = [w_i * math.exp(-alpha * y_i * h_i) for w_i, y_i, h_i in zip(weights, y_ada, h)]
        total_u = sum(unnorm)
        after = [u_i / total_u for u_i in unnorm]
        for pt in range(8):
            ada_rows.append([r_idx, pt + 1, weights[pt], unnorm[pt], after[pt]])
        round_stats[r_idx] = (eps, alpha)
        weights = after

    write_csv(
        os.path.join("outputs", "part4_adaboost.csv"),
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        ada_rows,
    )

    alpha1 = round_stats[1][1]
    alpha2 = round_stats[2][1]
    correct_ada = sum(
        1 for i in range(8) if adaboost_sign(alpha1 * h1[i] + alpha2 * h2[i]) == y_ada[i]
    )
    ensemble_accuracy = correct_ada / 8.0

    # Task 4.4: Out-of-bag share
    oob_fraction = (1.0 - 1.0 / n_train) ** n_train

    # Task 4.5: Neural network parameter sizing
    # Model A Option 1: 784 -> 16 -> 16 -> 10
    mlp_hidden1 = (784 + 1) * 16
    mlp_hidden2 = (16 + 1) * 16
    mlp_output = (16 + 1) * 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # Model A Option 2: CNN on 28x28x1
    c1_out = (28 - 5 + 2 * 2) // 1 + 1  # 28
    p1_out = (c1_out - 2 + 0) // 2 + 1  # 14
    c2_out = (p1_out - 5 + 0) // 1 + 1  # 10
    p2_out = (c2_out - 2 + 0) // 2 + 1  # 5
    flat_dim = p2_out * p2_out * 16      # 400

    cnn_conv1 = (5 * 5 * 1 + 1) * 6
    cnn_conv2 = (5 * 5 * 6 + 1) * 16
    cnn_dense1 = (flat_dim + 1) * 32
    cnn_dense2 = (32 + 1) * 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    arch_rows = [
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2],
    ]
    write_csv(
        os.path.join("outputs", "part4_architecture.csv"),
        ["layer", "parameters"],
        arch_rows,
    )

    # Why convolution scales: 128x128x3 input
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    conv_layer_params = (3 * 3 * 3 + 1) * 64

    # Model B: RNN (x_dim=64, h_dim=128, y_dim=16)
    rnn_total = (128 * 64) + (128 * 128) + 128 + (16 * 128) + 16

    results = [
        ("root_entropy", root_entropy),
        ("root_gini", root_gini),
        ("best_split_by_info_gain", best_split_ig),
        ("best_split_by_gini_gain", best_split_gg),
        ("stump_rate_yes", stump_rate_yes),
        ("stump_rate_no", stump_rate_no),
        ("round1_error", round_stats[1][0]),
        ("round1_alpha", round_stats[1][1]),
        ("round2_error", round_stats[2][0]),
        ("round2_alpha", round_stats[2][1]),
        ("ensemble_accuracy", ensemble_accuracy),
        ("oob_fraction", oob_fraction),
        ("mlp_total", mlp_total),
        ("cnn_total", cnn_total),
        ("fc_layer_params", fc_layer_params),
        ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ]
    write_results_txt(os.path.join("outputs", "part4_results.txt"), results)


if __name__ == "__main__":
    main()
