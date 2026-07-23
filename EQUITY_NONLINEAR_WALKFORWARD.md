# Non-linear models: same kill as Ridge and Kalman

**Question:** Ridge (linear regression) and the Kalman filter (linear,
adaptive) both failed the random-selection control — is that a statement
about linear models, or about the whole "these engineered features
predict forward returns" premise? Tests two structurally different
non-linear families — `HistGradientBoostingRegressor` (histogram-based
gradient boosting, sklearn's built-in LightGBM analog) and
`RandomForestRegressor` (bagged trees) — on the **exact same** candidate
pool, features, threshold-based selection rule, chronological folds, and
purging as the Ridge walk-forward. Only the model class changes, so a
different result is attributable to the model, not a methodology change.

## Result: kill, same as Ridge and Kalman

| Model | Real-label expectancy | Shuffled-label expectancy | Percentile vs. 100-seed random control |
|---|---|---|---|
| HistGradientBoosting | 15.1 bps | 18.9 bps | **19.0th** |
| RandomForest | 47.3 bps | 42.3 bps | **63.0th** |
| *(reference)* Ridge | — | — | 2nd–10th |
| *(reference)* simple_trend | — | — | 99th (deployed) |

**The tell, not just the percentile:** for both models, real-label
performance is statistically indistinguishable from shuffled-label
performance — HistGradientBoosting's shuffled labels actually *outperform*
its real labels (18.9 vs 15.1 bps), and RandomForest's real/shuffled gap
(47.3 vs 42.3) is well within noise. If either model had learned a real
feature→return relationship, real labels should meaningfully beat
permuted ones. They don't. Whatever expectancy these models produce
is coming from the same structural pool-level drift every selector in
this project sees (the unconditional bull-market bias in the candidate
pool), not from the model finding real signal in the features.

## Verdict

This closes the "model or premise" question this project has had open
since the Ridge kill: **it's the premise — or more precisely, this
specific 20-ish-feature set (returns, ATR-normalized trend/spread
measures, volume, relative strength, candlestick geometry) — not the
model family.** Four different model types (Ridge, Kalman, gradient
boosting, random forest) have now all failed the same random-selection
control on the same features. The only selector that has ever passed is
`simple_trend`, which doesn't use these features for prediction at all —
it just picks the strongest relative-momentum candidate. The edge in this
whole project has consistently been the exit mechanics + candidate pool
structure, not feature-based stock-picking, regardless of how
sophisticated the picker is.

**Not recommended as a next step:** more model families on this same
feature set (SVM, neural nets, etc.) — four kills across two model
families sharing the same result pattern is strong enough evidence this
door is closed, not just under-explored. If ML is worth revisiting, it
needs genuinely different features/information (the European lead signal
is the actual example of that working — a different information source
entirely, not a fancier model on the same inputs).

Script: `experiments/run_equity_nonlinear_walkforward.py`
