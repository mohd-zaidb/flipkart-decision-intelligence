Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

End-to-end statistical inference, churn classification, customer segmentation, and model-architecture sizing built from scratch in Python and NumPy on synthetic e-commerce customer records.

## How to run

From a clean clone of the repository root:

```bash
python -m venv .venv
# On Windows PowerShell: .\.venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` writes `data/records.csv` and prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`. Running `part1.py` through `part4.py` regenerates all files in `outputs/`.

## Part 1 findings

- **R1.1 (Planning assumption test):** Null hypothesis `H0: mu = 8.0` is rejected (`reject_h0 = yes`) because `z_stat = 2.8930` exceeds the two-tailed critical value `1.96` (`alpha = 0.05`). A Type I error here means concluding that Flipkart's true average order value differs from `INR 800` when the true population mean actually equals `INR 800`, causing unnecessary category plan revisions. The observed gap `mean_value - 8.0` is `0.9382` column units (`INR 93.82`, `11.7%` above plan), which is large enough to matter in practice for category revenue and margin budgeting.
- **R1.2 (Sample sizing):** Estimating the mean within `E = 0.5` (`INR 50`) at 95% confidence requires `n_required = 1293` records, compared with `800` rows currently available (whose 95% CI half-width is `1.96 * 0.3243 = 0.6356` units). The planning team therefore needs `493` additional sampled orders to hit the `INR 50` precision target.
- **R1.3 (Gradient descent vs. closed form):** The final slope gap is `gd_slope_gap = 0.0000`. Gradient descent reached the exact closed-form OLS line because linear regression on a single standardised predictor has a smooth, strictly convex MSE objective with a unique global minimum that `eta = 0.1` reaches well within 500 batch updates.
- **R1.4 (Residual funnel and log remedy):** The raw linear model exhibits a clear heteroscedastic funnel, with `resid_sd_ratio_raw = 3.7525` (`resid_sd_upper_raw = 8.6891` vs. `resid_sd_lower_raw = 2.3155`). Switching to `ln(order_value_inr_hundreds) ~ ln(avg_order_value_inr_hundreds)` removes the funnel, bringing `resid_sd_ratio_log` down to `1.0396` (`0.5369` vs. `0.5165`).
- **R1.5 (Multiple regression):** Excluding the intercept `mr_beta_0 = -1.8613`, the predictors with `|beta| > 0.05` are `ln(X4)` (`1.1576`), `X2` (`complaint_count_90d`, `0.0940`), `X3` (`discount_order_share`, `0.1067`), and `X6` (`cod_flag`, `0.1518`). Adjusted R-squared rises from `loglog_adj_r2 = 0.6561` to `mr_adj_r2 = 0.7403`, showing that the additional predictors meaningfully improve explanatory fit even after penalising for 7 predictors.

## Part 2 findings

- **R2.1 (Learning-rate effect):** Ordering the 200-update training losses from lowest to highest gives `eta = 1.0` (`0.3703`) < `eta = 0.1` (`0.3776`) < `eta = 0.01` (`0.5091`). A learning rate that is too small (`0.01`) moves too slowly toward the minimum in 200 steps, while a well-scaled large step (`1.0`) reaches near-optimal convergence faster on this smooth standardised problem without diverging.
- **R2.2 (Key churn drivers and odds ratios):** The three predictors with the largest absolute weights are `X3` (`discount_order_share`, `weight_X3 = 0.9013`), `X1` (`customer_tenure_months`, `weight_X1 = -0.5554`), and `X2` (`complaint_count_90d`, `weight_X2 = 0.5505`). Interpreting the largest odds ratio (`odds_ratio_X3 = 2.4629`): holding all other features fixed, each one-SD increase in a customer's discount order share multiplies their odds of churning by `2.4629`.
- **R2.3 (Accuracy trap and threshold shift):** A blind model predicting zero churners achieves `blind_accuracy = 0.7875`, nearly matching `tau_0.5_accuracy = 0.8042` despite catching zero churning customers, so plain accuracy hides failure on the `21.25%` minority class. Lowering `tau` from `0.5` to `0.3` boosts recall from `0.4706` to `0.7451` at the cost of lowering precision from `0.5455` to `0.4634`.
- **R2.4 (Logistic vs. KNN):** On the 240 test rows, logistic regression at `tau = 0.3` substantially outperforms 5-fold-CV KNN (`knn_best_k = 5`), achieving `tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451` versus `knn_f1 = 0.3696` and `knn_recall = 0.3333`.
- **R2.5 (Deployment threshold and cost):** Deploy logistic regression at `cost_tau = 0.1500`, which achieves `cost_test_at_tau = INR 6050` compared with `cost_test_at_0.5 = INR 11800` at the default `0.5` threshold and `cost_test_flag_nobody = INR 20400` when no retention coupons are sent. The threshold was selected strictly on the 560 training rows so that test-set cost remains an unbiased, out-of-sample evaluation rather than an optimistically tuned figure.

## Part 3 findings

- **R3.1 (K-means elbow):** The four successive WCSS drops are `WCSS(2) - WCSS(3) = 659.8282`, `WCSS(3) - WCSS(4) = 108.0739`, `WCSS(4) - WCSS(5) = 171.0575`, and `WCSS(5) - WCSS(6) = 45.5466`. The elbow rule selects `elbow_k = 3` because `108.0739 < 0.5 * 659.8282 = 329.9141`; the drops do not shrink monotonically because the pinned starting rows give one reproducible local solution rather than necessarily the global minimum for every `K`.
- **R3.2 (Segment profiles):**
  - **Segment A — Premium Loyalists (`322` customers):** Longest tenure (`47.5028` months), lowest complaints (`1.9503`), lowest discount share (`0.2250`), highest loyalty (`62.1584`), `outcome_rate = 0.0186`, `mean_value = 3.5976` (`INR 359.76`). Action: Retain via Plus membership benefits and curated high-margin recommendations rather than discount coupons.
  - **Segment B — High-Value Friction-Prone Shoppers (`255` customers):** Mid tenure (`28.6071` months), highest complaints (`6.0706`), moderate discount share (`0.4534`), highest past AOV (`28.7192`) and current order value (`17.1248`, `INR 1,712.48`), `outcome_rate = 0.2353`. Action: Route complaints to priority customer-support resolution to prevent high-value churn.
  - **Segment C — Bargain-Focused Early-Tenure Shoppers (`223` customers):** Shortest tenure (`14.3892` months), highest discount share (`0.7118`), lowest loyalty (`38.5215`), `mean_value = 7.2883` (`INR 728.83`), and the **highest churn rate (`outcome_rate = 0.4664`)**. Action: Target with proactive retention coupons and early onboarding incentives.
- **R3.3 (Hierarchical clustering disagreement):** At the largest-gap cut, single linkage yields `4` clusters (`{E2, E3, E6}`, `{E1}`, `{E4}`, `{E5}`) whereas complete linkage yields `2` clusters (`{E1, E4, E5}`, `{E2, E3, E6}`). Single linkage judges cluster proximity by the closest pair of members (chaining `{E2, E3, E6}` at `1.4000` and leaving a `1.4000` gap before `{E4, E5}` merge at `2.8000`), while complete linkage merges by the farthest pair so the between-group diameter jump (`7.9000` vs. `2.9000`) dominates and creates two compact groups.
- **R3.4 (PCA):** After standardising `X1`–`X5`, the mean vector in Task 3.5 Step 1 is all zeros (`[0, 0, 0, 0, 0]`). PC1 (`eigenvalue_1 = 2.6136`, `52.27%` of variance) contrasts high tenure (`pc1_loading_X1 = 0.5317`) and loyalty (`pc1_loading_X5 = 0.4418`) against high discount share (`pc1_loading_X3 = -0.5187`), complaints (`pc1_loading_X2 = -0.4100`), and past AOV (`pc1_loading_X4 = -0.2916`). Retaining at least 80% of variance requires `components_for_80pct = 3` components (`cumulative_ratio_3 = 0.8616`). Segment PC1 means confirm clean separation along this loyalty-vs-discount axis: Segment A averages `+1.7773`, Segment B `-0.7831`, and Segment C `-1.6709`.

## Part 4 findings

- **R4.1 (Split purity and overfitting):** Split `S4` (`customer_tenure_months <= 24`) is best by both information gain (`0.1252`) and Gini gain (`0.0606`) because it isolates a high-churn early-tenure group (`86` positives out of `188`, `45.74%`) from low-churn tenured buyers (`33` out of `372`, `8.87%`). Growing a tree until every leaf is pure memorises random label noise in tiny sub-regions and degrades generalisation; imposing a stopping rule such as `max_depth` or `min_samples_leaf` prevents excessive branching.
- **R4.2 (Single models vs. voting):** Among the five Task 4.2 models, I would ship `logistic` (`f1 = 0.5714`, `recall = 0.7451`). Combining models did not help: `hard_vote` (`f1 = 0.5344`, `recall = 0.6863`) and `soft_vote` (`f1 = 0.5507`, `recall = 0.7451`) both scored lower F1 than `logistic` alone because the weaker `knn` (`f1 = 0.3696`, `recall = 0.3333`) and `stump` (`f1 = 0.5532`, `recall = 0.7647`) models lowered vote precision.
- **R4.3 (AdaBoost weighting):** After round 1 normalisation, the two misclassified points (points 3 and 7) carry `0.2500` each for a total weight of `0.5000`. Round 2 assigns a higher weight to `h2` (`round2_alpha = 0.8047`) than round 1 gave `h1` (`round1_alpha = 0.5493`) because `h2` achieves a lower weighted error (`round2_error = 0.1667`) on the reweighted points than `h1` achieved (`round1_error = 0.2500`).
- **R4.4 (Neural network sizing and depth):**

| Layer (Model A Option 2 CNN) | Output Shape (H × W × D) | Parameters |
|---|---|---|
| Input | `28 × 28 × 1` | `0` |
| `cnn_conv1` (5×5, 6 filters, s1, p2) + ReLU | `28 × 28 × 6` | `156` |
| MaxPool 1 (2×2, s2) | `14 × 14 × 6` | `0` |
| `cnn_conv2` (5×5, 16 filters, s1, p0) + ReLU | `10 × 10 × 16` | `2416` |
| MaxPool 2 (2×2, s2) | `5 × 5 × 16` | `0` |
| Flatten | `400` | `0` |
| `cnn_dense1` (32 neurons) + ReLU | `32` | `12832` |
| `cnn_dense2` (10 neurons) | `10` | `330` |

For Model B, the matrix shapes are `W_xh: 128 × 64`, `W_hh: 128 × 128`, and `W_hy: 16 × 128`, and `rnn_total = 26768` does not grow with sequence length because the exact same recurrent matrices and biases are reused at every time step. For 10-class image classification, use `ReLU` in hidden layers and `Softmax` on the output layer paired with categorical cross-entropy loss. A very deep plain network can stall in its early layers because repeated multiplication of small derivatives shrinks gradients toward zero (vanishing gradients) during backpropagation; adding skip connections creates a direct identity path for gradients to flow unhindered back to early layers.

## Model-choice memo

**To:** Head of Retention Team  
**Subject:** Production Churn Model Recommendation  

I recommend deploying **logistic regression** (`logistic` from Task 4.2) for Flipkart's 90-day churn retention campaign. On the 240-customer test set, it achieved the highest F1 score (`0.5714`, `recall = 0.7451` at `tau = 0.3` and `test_bce = 0.3818`), outperforming 5-nearest-neighbours (`f1 = 0.3696`), the single-question stump (`f1 = 0.5532`), hard voting (`f1 = 0.5344`), and soft voting (`f1 = 0.5507`).

For coupon targeting, I recommend setting the probability alert threshold to **`cost_tau = 0.1500`**, selected purely on the 560 training rows. At `tau = 0.1500`, total test-set cost is **`INR 6050`**, cutting retention-campaign losses by nearly half compared with the logistic regression comparator at the default `0.5` threshold (`INR 11800`) and saving `INR 14350` versus flagging nobody (`INR 20400`).

**Key limitation:** All evidence comes from a synthetic 800-row dataset evaluated on a single 240-row test split containing only `51` churn positives, and the `0.1500` threshold assumes fixed unit costs of `INR 50` per wasted coupon and `INR 400` per missed churner.

Neither Model A (product-image CNN, `15734` parameters) nor Model B (review-text RNN, `26768` parameters) should be built before the structured logistic model is in production, because structured churn targeting delivers immediate, measurable rupee savings with minimal engineering cost.

## Limitations

1. **Synthetic data assumptions:** All 800 records come from `generate_data.py` with smooth Gaussian/log-normal latent groups and stationary relationships, which may not reflect real Flipkart seasonality, catalogue shifts, or missing-data noise.
2. **Single 70/30 split and small positive test count:** Test metrics and cost comparisons rest on one 240-row split with only `51` churn positives, so confusion-matrix counts and threshold costs carry sampling variance.
3. **Fixed cost ratio and static features:** Threshold selection at `cost_tau = 0.1500` treats every churner's future margin loss as exactly `INR 400` and every coupon as `INR 50`, ignoring customer-level order value differences across Segments A, B, and C.
4. **No out-of-time or live A/B validation:** Model performance was measured on held-out static rows rather than future time windows or a live controlled coupon experiment to verify true uplift.
