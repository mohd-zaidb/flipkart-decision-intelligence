Domain: E-commerce

# Flipkart E-commerce Decision Intelligence

Analytics and ML pipeline built from scratch in Python + NumPy on synthetic customer data to look at order values, 90-day churn prediction, customer segments, and neural net sizing.

## How to run

Clone the repo, go to the project root, and run these commands in order:

```bash
python -m venv .venv
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` creates `data/records.csv` and prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`. Running `part1.py` through `part4.py` generates all the files inside `outputs/`.

## Part 1 findings

- R1.1: We reject H0 (`reject_h0 = yes`) because `z_stat = 2.8930` is bigger than the critical value `1.96` (`alpha = 0.05`). In this context, a Type I error would mean deciding Flipkart's true average order value is not `INR 800` when it actually is `INR 800`, so the team would change category plans when nothing really changed. The actual gap `mean_value - 8.0` is `0.9382` units (`INR 93.82`, around `11.7%` higher than planned), which is big enough to matter in real budgeting.
- R1.2: To estimate the average within `E = 0.5` (`INR 50`) at 95% confidence, we need `n_required = 1293` rows, whereas we only have `800` right now (our current 95% CI half-width is `1.96 * 0.3243 = 0.6356`). So the planning team needs `493` more records before they hit that `INR 50` precision target.
- R1.3: The difference between the two slopes is `gd_slope_gap = 0.0000`. Gradient descent matched the closed-form line because MSE on a single standardised variable is a single smooth bowl shape with one global minimum, and `eta = 0.1` walks right to it within 500 updates.
- R1.4: The raw regression clearly has a funnel (`resid_sd_ratio_raw = 3.7525`, since the upper half residual SD is `8.6891` compared to `2.3155` in the lower half). Logging both variables (`ln(order_value_inr_hundreds)` on `ln(avg_order_value_inr_hundreds)`) removes that funnel, dropping `resid_sd_ratio_log` down to `1.0396` (`0.5369` vs `0.5165`).
- R1.5: Excluding the constant `mr_beta_0 = -1.8613`, the predictors in Task 1.7 with `|beta| > 0.05` are `ln(X4)` (`1.1576`), `X2` (`complaint_count_90d`, `0.0940`), `X3` (`discount_order_share`, `0.1067`), and `X6` (`cod_flag`, `0.1518`). Also `mr_adj_r2 = 0.7403` beats `loglog_adj_r2 = 0.6561`, which tells us the extra predictors add real explanatory value even after adjusting for having 7 features (though `0.05` is just a mechanical cutoff since the columns are in different units).

## Part 2 findings

- R2.1: Ordering the Task 2.3 training losses after 200 updates from lowest to highest gives `eta = 1.0` (`0.3703`) < `eta = 0.1` (`0.3776`) < `eta = 0.01` (`0.5091`). When `eta` is too small (`0.01`), weights barely move in 200 steps, while on well-scaled standardised features a bigger step (`1.0`) gets close to the minimum much faster without blowing up.
- R2.2: The top three predictors by absolute weight are `X3` (`discount_order_share`, `weight_X3 = 0.9013`), `X1` (`customer_tenure_months`, `weight_X1 = -0.5554`), and `X2` (`complaint_count_90d`, `weight_X2 = 0.5505`). Looking at the largest odds ratio (`odds_ratio_X3 = 2.4629`): holding everything else fixed, a 1-SD jump in a customer's discount order share multiplies their odds of churning by `2.4629`.
- R2.3: Predicting 0 for everyone already gets `blind_accuracy = 0.7875`, which is almost as high as `tau_0.5_accuracy = 0.8042` even though it catches zero churners, so plain accuracy is misleading on imbalanced data. Dropping `tau` from `0.5` to `0.3` pushes recall up from `0.4706` to `0.7451` while precision dips from `0.5455` to `0.4634`.
- R2.4: On the 240 test rows, logistic regression at `tau = 0.3` beats KNN (`knn_best_k = 5`) easily, getting `tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451` compared to `knn_f1 = 0.3696` and `knn_recall = 0.3333`.
- R2.5: I would deploy logistic regression at `cost_tau = 0.1500` because its test cost is `cost_test_at_tau = INR 6050`, compared with `cost_test_at_0.5 = INR 11800` at `tau = 0.5` and `cost_test_flag_nobody = INR 20400` if we ignore everyone. We picked `cost_tau` using only the 560 training rows so the test-set cost stays an honest out-of-sample check instead of overfitting the threshold to the test data.

## Part 3 findings

- R3.1: The four successive WCSS drops are `WCSS(2) - WCSS(3) = 659.8282`, `WCSS(3) - WCSS(4) = 108.0739`, `WCSS(4) - WCSS(5) = 171.0575`, and `WCSS(5) - WCSS(6) = 45.5466`. The elbow rule picks `elbow_k = 3` because `108.0739 < 0.5 * 659.8282 = 329.9141`; the drops do not shrink smoothly every step because the pinned starting centroids give one reproducible local solution, not always the global minimum for every `K`.
- R3.2: From `part3_segments.csv`, the three groups look like:
  - Segment A (Premium Loyalists, `322` users): Longest tenure (`47.5028` months), fewest complaints (`1.9503`), lowest discount share (`0.2250`), highest loyalty (`62.1584`), `outcome_rate = 0.0186`, `mean_value = 3.5976` (`INR 359.76`). Action: Keep them happy with Plus perks and cross-sells instead of wasting discount coupons.
  - Segment B (High-Value Friction Shoppers, `255` users): Mid tenure (`28.6071` months), highest complaints (`6.0706`), moderate discounts (`0.4534`), highest past AOV (`28.7192`) and current order value (`17.1248`, `INR 1,712.48`), `outcome_rate = 0.2353`. Action: Give them fast priority support when they complain so we do not lose big-basket shoppers.
  - Segment C (Bargain-Focused New Shoppers, `223` users): Shortest tenure (`14.3892` months), highest discount share (`0.7118`), lowest loyalty (`38.5215`), `mean_value = 7.2883` (`INR 728.83`), and the highest churn rate (`outcome_rate = 0.4664`). Action: Send targeted retention coupons and early onboarding nudges before they drop off.
- R3.3: At the largest-gap cut, single linkage gives `4` clusters (`{E2, E3, E6}`, `{E1}`, `{E4}`, `{E5}`) while complete linkage gives `2` clusters (`{E1, E4, E5}`, `{E2, E3, E6}`). Single linkage only cares about the closest pair between groups (chaining `{E2, E3, E6}` at `1.4000` before a `1.4000` jump to `2.8000`), whereas complete linkage looks at the farthest pair, so the huge diameter jump (`7.9000` vs `2.9000`) splits the table cleanly into two tight clusters.
- R3.4: After standardising `X1` to `X5`, the Step 1 mean vector is all zeros (`[0, 0, 0, 0, 0]`). PC1 (`eigenvalue_1 = 2.6136`, `52.27%` of variance) loads positively on tenure (`pc1_loading_X1 = 0.5317`) and loyalty (`pc1_loading_X5 = 0.4418`), and negatively on discount share (`pc1_loading_X3 = -0.5187`), complaints (`pc1_loading_X2 = -0.4100`), and past AOV (`pc1_loading_X4 = -0.2916`). We need `components_for_80pct = 3` components to clear 80% variance (`cumulative_ratio_3 = 0.8616`). The segment PC1 averages line up nicely with the clusters: Segment A is `+1.7773`, Segment B is `-0.7831`, and Segment C is `-1.6709`.

## Part 4 findings

- R4.1: Split `S4` (`customer_tenure_months <= 24`) wins both info gain (`0.1252`) and Gini gain (`0.0606`) because it separates newer customers (`45.74%` churn, `86` of `188`) from older ones (`8.87%` churn, `33` of `372`). Growing a tree until every leaf is pure makes it memorize random noise in tiny pockets and overfit, so using a stopping rule like `max_depth` or `min_samples_leaf` keeps it generalising.
- R4.2: Out of the five Task 4.2 models, I would ship `logistic` (`f1 = 0.5714`, `recall = 0.7451`). Combining models actually hurt here: `hard_vote` (`f1 = 0.5344`, `recall = 0.6863`) and `soft_vote` (`f1 = 0.5507`, `recall = 0.7451`) both scored lower F1 than `logistic` alone because weak `knn` (`f1 = 0.3696`, `recall = 0.3333`) and `stump` (`f1 = 0.5532`, `recall = 0.7647`) dragged down precision.
- R4.3: After round 1 normalisation, the two misclassified points (points 3 and 7) carry `0.2500` each, so their combined weight is `0.5000`. Round 2 trusts `h2` (`round2_alpha = 0.8047`) more than round 1 trusted `h1` (`round1_alpha = 0.5493`) because `h2` makes a smaller weighted error (`round2_error = 0.1667`) on the reweighted points than `h1` made (`round1_error = 0.2500`).
- R4.4: Layer breakdown for Model A Option 2 (CNN):

| Layer | Output Shape (H × W × D) | Parameters |
|---|---|---|
| Input | `28 × 28 × 1` | `0` |
| `cnn_conv1` (5×5, 6 filters, s1, p2) + ReLU | `28 × 28 × 6` | `156` |
| MaxPool 1 (2×2, s2) | `14 × 14 × 6` | `0` |
| `cnn_conv2` (5×5, 16 filters, s1, p0) + ReLU | `10 × 10 × 16` | `2416` |
| MaxPool 2 (2×2, s2) | `5 × 5 × 16` | `0` |
| Flatten | `400` | `0` |
| `cnn_dense1` (32 neurons) + ReLU | `32` | `12832` |
| `cnn_dense2` (10 neurons) | `10` | `330` |

For Model B, the matrix shapes are `W_xh: 128 × 64`, `W_hh: 128 × 128`, and `W_hy: 16 × 128`, and `rnn_total = 26768` stays the same even for longer text sequences because the RNN reuses the exact same weight matrices and biases at every step. For 10-class image classification, I would use `ReLU` on hidden layers and `Softmax` on the output layer with categorical cross-entropy loss. Very deep plain networks can stall in early layers because multiplying lots of small derivatives shrinks gradients close to zero during backprop (vanishing gradients); skip connections fix this by giving gradients an easy identity shortcut straight back to the early layers.

## Model-choice memo

**To:** Head of Retention Team  
**Subject:** Churn Model Recommendation for Production  

I recommend shipping the **logistic regression** model (`logistic` from Task 4.2) to flag customers likely to churn in the next 90 days. On our 240-customer test set, it gave the best F1 score (`0.5714`, with `recall = 0.7451` at `tau = 0.3` and `test_bce = 0.3818`), beating 5-nearest-neighbours (`f1 = 0.3696`), the single-split stump (`f1 = 0.5532`), hard voting (`f1 = 0.5344`), and soft voting (`f1 = 0.5507`).

For sending retention coupons, I recommend using the probability cutoff **`cost_tau = 0.1500`** (chosen purely on the 560 training rows). At `tau = 0.1500`, total test cost drops to **`INR 6050`**, which is almost half the logistic regression comparator cost at `tau = 0.5` (`INR 11800`) and saves `INR 14350` compared to flagging nobody (`INR 20400`).

**Main risk:** These numbers come from a synthetic 800-row dataset tested on one 240-row holdout split with just `51` churners, and assume fixed costs of `INR 50` per wasted coupon and `INR 400` per missed churner.

Neither Model A (product-image CNN, `15734` parameters) nor Model B (review-text RNN, `26768` parameters) is worth building before this structured churn model is live, since tabular customer signals already deliver clear rupee savings right away.

## Limitations

1. **Synthetic data:** All 800 rows are generated from fixed formulas in `generate_data.py`, so there are no real Flipkart holiday spikes, missing fields, or messy real-world shifts.
2. **Single train/test split:** Test scores come from just one 70/30 split with only `51` positive churn rows, so a few predictions flipping can noticeably move F1 or cost.
3. **Fixed rupee costs:** Using `INR 50` for every false alarm and `INR 400` for every missed churner ignores that Segment B big-basket customers lose way more margin than low-spend buyers.
4. **No live A/B test yet:** Everything here is offline prediction; we have not tested in a live campaign whether a `INR 50` coupon actually convinces flagged users to stay.
