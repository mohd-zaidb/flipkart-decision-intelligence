"""Part 1: Estimation, inference, and a regression baseline."""
import math
import os
import numpy as np
from shared import load_records, sample_mean_sd, write_results_txt


def simple_ols(x, y):
    """Closed-form simple linear regression y ~ x using sample covariance and variance."""
    n = len(x)
    x_mean = float(np.sum(x) / n)
    y_mean = float(np.sum(y) / n)
    cov_xy = float(np.sum((x - x_mean) * (y - y_mean)) / (n - 1))
    var_x = float(np.sum((x - x_mean) ** 2) / (n - 1))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    tss = float(np.sum((y - y_mean) ** 2))
    ess = float(np.sum((fitted - y_mean) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.sum(np.abs(y - fitted)) / n)
    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def funnel_check(fitted, residuals):
    """Split residuals into lower 400 and upper 400 by stable sort of fitted values."""
    order = np.argsort(fitted, kind="stable")
    half = len(fitted) // 2
    lower_res = residuals[order[:half]]
    upper_res = residuals[order[half:]]
    _, sd_lower = sample_mean_sd(lower_res, axis=0)
    _, sd_upper = sample_mean_sd(upper_res, axis=0)
    return float(sd_lower), float(sd_upper), float(sd_upper / sd_lower)


def main():
    _, _, X, _, value = load_records()
    n = len(value)
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]
    x5 = X[:, 4]
    x6 = X[:, 5]
    x7 = X[:, 6]

    # Task 1.1: Estimate with an interval
    mean_value, sd_value = sample_mean_sd(value, axis=0)
    mean_value = float(mean_value)
    sd_value = float(sd_value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2: Test a planning claim (H0: mu = 8.0)
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96

    # Task 1.3: Size a future sample (E = 0.5)
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    # Task 1.4: Simple regression by closed form (value ~ X4)
    ols_raw = simple_ols(x4, value)
    tss_equals_ess_plus_rss = abs(ols_raw["tss"] - ols_raw["ess"] - ols_raw["rss"]) < 1e-6 * ols_raw["tss"]

    # Task 1.5: Fit the same line by gradient descent
    x4_mean, x4_sd = sample_mean_sd(x4, axis=0)
    x4_mean = float(x4_mean)
    x4_sd = float(x4_sd)
    z = (x4 - x4_mean) / x4_sd
    m = 0.0
    c = 0.0
    eta = 0.1

    def gd_cost(m_val, c_val):
        err = value - (m_val * z + c_val)
        return float(np.sum(err ** 2) / n)

    gd_losses = {0: gd_cost(m, c)}
    for step in range(1, 501):
        y_hat = m * z + c
        err = value - y_hat
        dm = float(-(2.0 / n) * np.sum(z * err))
        dc = float(-(2.0 / n) * np.sum(err))
        m = m - eta * dm
        c = c - eta * dc
        if step in (1, 10, 100, 500):
            gd_losses[step] = gd_cost(m, c)

    gd_slope = m / x4_sd
    gd_intercept = c - m * x4_mean / x4_sd
    gd_slope_gap = abs(gd_slope - ols_raw["slope"])

    # Task 1.6: Check for a funnel, then apply log remedy
    res_raw = value - ols_raw["fitted"]
    resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw = funnel_check(ols_raw["fitted"], res_raw)

    ln_x4 = np.log(x4)
    ln_value = np.log(value)
    ols_log = simple_ols(ln_x4, ln_value)
    res_log = ln_value - ols_log["fitted"]
    resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log = funnel_check(ols_log["fitted"], res_log)

    # Task 1.7: Multiple regression by the normal equation
    X_design = np.column_stack([np.ones(n, dtype=np.float64), ln_x4, x1, x2, x3, x5, x6, x7])
    beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ ln_value)
    mr_fitted = X_design @ beta
    ln_mean = float(np.sum(ln_value) / n)
    mr_tss = float(np.sum((ln_value - ln_mean) ** 2))
    mr_rss = float(np.sum((ln_value - mr_fitted) ** 2))
    mr_r2 = 1.0 - mr_rss / mr_tss
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
    loglog_adj_r2 = 1.0 - (1.0 - ols_log["r2"]) * (n - 1) / (n - 1 - 1)

    results = [
        ("n", n),
        ("mean_value", mean_value),
        ("sd_value", sd_value),
        ("se_value", se_value),
        ("ci95_low", ci95_low),
        ("ci95_high", ci95_high),
        ("z_stat", z_stat),
        ("reject_h0", reject_h0),
        ("n_required", n_required),
        ("reg_slope", ols_raw["slope"]),
        ("reg_intercept", ols_raw["intercept"]),
        ("tss", ols_raw["tss"]),
        ("ess", ols_raw["ess"]),
        ("rss", ols_raw["rss"]),
        ("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss),
        ("r2", ols_raw["r2"]),
        ("rmse", ols_raw["rmse"]),
        ("mae", ols_raw["mae"]),
        ("gd_J0", gd_losses[0]),
        ("gd_J1", gd_losses[1]),
        ("gd_J10", gd_losses[10]),
        ("gd_J100", gd_losses[100]),
        ("gd_J500", gd_losses[500]),
        ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap),
        ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw),
        ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", ols_log["intercept"]),
        ("loglog_slope", ols_log["slope"]),
        ("loglog_r2", ols_log["r2"]),
        ("loglog_rmse", ols_log["rmse"]),
        ("resid_sd_lower_log", resid_sd_lower_log),
        ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
        ("mr_beta_0", float(beta[0])),
        ("mr_beta_1", float(beta[1])),
        ("mr_beta_2", float(beta[2])),
        ("mr_beta_3", float(beta[3])),
        ("mr_beta_4", float(beta[4])),
        ("mr_beta_5", float(beta[5])),
        ("mr_beta_6", float(beta[6])),
        ("mr_beta_7", float(beta[7])),
        ("mr_r2", mr_r2),
        ("mr_adj_r2", mr_adj_r2),
        ("loglog_adj_r2", loglog_adj_r2),
    ]
    write_results_txt(os.path.join("outputs", "part1_results.txt"), results)


if __name__ == "__main__":
    main()
