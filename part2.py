"""Part 2: Classification and evaluation (logistic regression, KNN, and cost threshold)."""
import math
import os
import numpy as np
from shared import (
    binary_cross_entropy,
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


def main():
    _, _, X, y, _ = load_records()
    split = stratified_split_and_standardise(X, y)
    X_train = split["X_train_std"]
    X_test = split["X_test_std"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    train_rows = len(y_train)
    train_positives = int(np.sum(y_train))
    test_rows = len(y_test)
    test_positives = int(np.sum(y_test))

    # Task 2.2: Logistic regression from scratch (eta = 0.1, 1000 updates)
    w, b, losses = train_logistic_regression(
        X_train, y_train, eta=0.1, epochs=1000, checkpoints=(0, 1, 10, 100, 1000)
    )
    p_test = sigmoid(X_test @ w + b)
    test_bce = binary_cross_entropy(y_test, p_test)

    # Task 2.3: Learning-rate effect (200 updates)
    lr_losses = {}
    for eta in (0.01, 0.1, 1.0):
        _, _, l_map = train_logistic_regression(
            X_train, y_train, eta=eta, epochs=200, checkpoints=(200,)
        )
        lr_losses[eta] = l_map[200]

    # Task 2.4: Odds ratios
    odds_ratios = [math.exp(float(wj)) for wj in w]

    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    eval_05_pred = (p_test >= 0.5).astype(np.int64)
    tp05, fp05, tn05, fn05 = confusion_matrix_counts(y_test, eval_05_pred)
    m05 = classification_metrics(tp05, fp05, tn05, fn05)

    eval_03_pred = (p_test >= 0.3).astype(np.int64)
    tp03, fp03, tn03, fn03 = confusion_matrix_counts(y_test, eval_03_pred)
    m03 = classification_metrics(tp03, fp03, tn03, fn03)

    blind_accuracy = float(np.sum(y_test == 0) / test_rows)

    # Task 2.6: Threshold sweep tau = 0.1 .. 0.9
    sweep_rows = []
    for k in range(1, 10):
        tau = k / 10.0
        pred_t = (p_test >= tau).astype(np.int64)
        tp, fp, tn, fn = confusion_matrix_counts(y_test, pred_t)
        mt = classification_metrics(tp, fp, tn, fn)
        sweep_rows.append([tau, tp, fp, tn, fn, mt["tpr"], mt["fpr"]])
    write_csv(
        os.path.join("outputs", "part2_thresholds.csv"),
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        sweep_rows,
    )

    # Task 2.7: KNN with 5-fold CV (fold = index % 5)
    folds = np.arange(train_rows) % 5
    cv_rows = []
    best_k = None
    best_cv_f1 = -1.0
    for k in range(1, 22, 2):
        fold_f1s = []
        for f_idx in range(5):
            val_mask = folds == f_idx
            ref_mask = ~val_mask
            preds_val, _ = knn_predict(X_train[ref_mask], y_train[ref_mask], X_train[val_mask], k)
            tp, fp, tn, fn = confusion_matrix_counts(y_train[val_mask], preds_val)
            fold_f1s.append(classification_metrics(tp, fp, tn, fn)["f1"])
        mean_f1 = sum(fold_f1s) / 5.0
        cv_rows.append([k, mean_f1])
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_k = k

    write_csv(
        os.path.join("outputs", "part2_knn_cv.csv"),
        ["K", "mean_cv_f1"],
        cv_rows,
    )

    knn_test_pred, _ = knn_predict(X_train, y_train, X_test, best_k)
    knn_tp, knn_fp, knn_tn, knn_fn = confusion_matrix_counts(y_test, knn_test_pred)
    knn_m = classification_metrics(knn_tp, knn_fp, knn_tn, knn_fn)

    # Task 2.8: Cost-minimising alert threshold on training rows (cost = 50*fp + 400*fn)
    p_train = sigmoid(X_train @ w + b)
    best_tau = None
    best_train_cost = None
    for k in range(1, 20):
        tau = k / 20.0
        pred_tr = (p_train >= tau).astype(np.int64)
        _, fp_tr, _, fn_tr = confusion_matrix_counts(y_train, pred_tr)
        cost_tr = 50 * fp_tr + 400 * fn_tr
        if best_train_cost is None or cost_tr < best_train_cost:
            best_train_cost = cost_tr
            best_tau = tau

    pred_test_tau = (p_test >= best_tau).astype(np.int64)
    _, fp_te_tau, _, fn_te_tau = confusion_matrix_counts(y_test, pred_test_tau)
    cost_test_at_tau = 50 * fp_te_tau + 400 * fn_te_tau

    cost_test_at_05 = 50 * fp05 + 400 * fn05
    cost_test_flag_nobody = 400 * test_positives

    results = [
        ("train_rows", train_rows),
        ("train_positives", train_positives),
        ("test_rows", test_rows),
        ("test_positives", test_positives),
        ("loss_epoch_0", losses[0]),
        ("loss_epoch_1", losses[1]),
        ("loss_epoch_10", losses[10]),
        ("loss_epoch_100", losses[100]),
        ("loss_epoch_1000", losses[1000]),
    ]
    for j in range(7):
        results.append((f"weight_X{j + 1}", float(w[j])))
    results.extend([
        ("bias", float(b)),
        ("test_bce", test_bce),
        ("loss_epoch_200_eta_0.01", lr_losses[0.01]),
        ("loss_epoch_200_eta_0.1", lr_losses[0.1]),
        ("loss_epoch_200_eta_1.0", lr_losses[1.0]),
    ])
    for j in range(7):
        results.append((f"odds_ratio_X{j + 1}", odds_ratios[j]))
    results.extend([
        ("tau_0.5_tp", tp05),
        ("tau_0.5_fp", fp05),
        ("tau_0.5_tn", tn05),
        ("tau_0.5_fn", fn05),
        ("tau_0.5_accuracy", m05["accuracy"]),
        ("tau_0.5_precision", m05["precision"]),
        ("tau_0.5_recall", m05["recall"]),
        ("tau_0.5_f1", m05["f1"]),
        ("tau_0.5_specificity", m05["specificity"]),
        ("tau_0.3_tp", tp03),
        ("tau_0.3_fp", fp03),
        ("tau_0.3_tn", tn03),
        ("tau_0.3_fn", fn03),
        ("tau_0.3_accuracy", m03["accuracy"]),
        ("tau_0.3_precision", m03["precision"]),
        ("tau_0.3_recall", m03["recall"]),
        ("tau_0.3_f1", m03["f1"]),
        ("tau_0.3_specificity", m03["specificity"]),
        ("blind_accuracy", blind_accuracy),
        ("knn_best_k", best_k),
        ("knn_tp", knn_tp),
        ("knn_fp", knn_fp),
        ("knn_tn", knn_tn),
        ("knn_fn", knn_fn),
        ("knn_accuracy", knn_m["accuracy"]),
        ("knn_precision", knn_m["precision"]),
        ("knn_recall", knn_m["recall"]),
        ("knn_f1", knn_m["f1"]),
        ("cost_tau", best_tau),
        ("cost_train_at_tau", best_train_cost),
        ("cost_test_at_tau", cost_test_at_tau),
        ("cost_test_at_0.5", cost_test_at_05),
        ("cost_test_flag_nobody", cost_test_flag_nobody),
    ])
    write_results_txt(os.path.join("outputs", "part2_results.txt"), results)


if __name__ == "__main__":
    main()
