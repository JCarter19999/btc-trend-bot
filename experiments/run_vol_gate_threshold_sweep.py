import numpy as np
import pandas as pd

# Regime built from the canonical intraday_matrix_50000 feature frame (has
# atr_5m), exactly as run_ema_pullback_full_backtest.py and
# run_vol_gate_broaden_search.py did -- NOT popular_matrix_50000's own
# feature_frame.csv (different ATR column names, would be a different
# regime definition).
ff = pd.read_csv("/home/joey/btc-paper-5m/outputs/intraday_matrix_50000/feature_frame.csv", parse_dates=["timestamp"])
ff = ff.sort_values("timestamp").reset_index(drop=True)
ff["norm_atr"] = ff["atr_5m"] / ff["close"]
pct = ff["norm_atr"].rolling(2016, min_periods=2016).apply(lambda w: (w < w.iloc[-1]).mean(), raw=False)
ff["vol_percentile"] = pct.shift(1)  # no-lookahead

te = pd.read_csv("/home/joey/btc-paper-5m/outputs/popular_matrix_50000/trade_episodes.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
ep = te[(te.strategy_id == "ema_pullback_15m_4h") & (te.is_open == False)].copy().sort_values("entry_timestamp").reset_index(drop=True)

# Merge the CONTINUOUS percentile once (not a pre-thresholded boolean) --
# every threshold below is then a plain float comparison on a clean numeric
# column, so there is no risk of the dropna-after-merge dtype trap (a bool
# column with NaN silently upcasting to object, which previously broke `~`
# into bitwise complement).
ff_idx = ff.set_index("timestamp")
merged = pd.merge_asof(ep, ff_idx[["vol_percentile"]].reset_index(),
                        left_on="entry_timestamp", right_on="timestamp", direction="backward")
merged_before = len(merged)
merged = merged.dropna(subset=["vol_percentile"]).reset_index(drop=True)
print(f"ema_pullback_15m_4h trades merged with regime percentile: {len(merged)} / {len(ep)} "
      f"(dropped {merged_before - len(merged)}, entries before first classifiable bar -- expected)")
print(f"vol_percentile dtype: {merged['vol_percentile'].dtype} (must be float, confirms no bool-upcast risk)\n")

merged_sorted = merged.sort_values("entry_timestamp").reset_index(drop=True)
mid = len(merged_sorted) // 2

print("=== LEVER 1: volatility threshold sensitivity sweep ===\n")
print(f"{'Threshold':<10} {'n_hv':>5} {'n_other':>7} {'hv_mean%':>9} {'other_mean%':>11} {'diff_pp':>8} "
      f"{'OOS_1st_pp':>10} {'OOS_2nd_pp':>10} {'sign_flip':>9}")

results = []
for thresh in (0.50, 0.60, 0.65, 0.70, 0.75):
    is_hv = merged["vol_percentile"] >= thresh
    hv = merged[is_hv]
    other = merged[~is_hv]
    diff = hv.net_portfolio_return_pct.mean() - other.net_portfolio_return_pct.mean()

    is_hv_sorted = merged_sorted["vol_percentile"] >= thresh
    first_half, second_half = merged_sorted.iloc[:mid].copy(), merged_sorted.iloc[mid:].copy()
    first_half["is_hv"] = is_hv_sorted.iloc[:mid]
    second_half["is_hv"] = is_hv_sorted.iloc[mid:]

    oos = {}
    for label, half in (("first", first_half), ("second", second_half)):
        h, o = half[half.is_hv], half[~half.is_hv]
        oos[label] = (h.net_portfolio_return_pct.mean() - o.net_portfolio_return_pct.mean()) if len(h) and len(o) else np.nan

    flip = "N/A"
    if not (np.isnan(oos["first"]) or np.isnan(oos["second"])):
        flip = "YES" if np.sign(oos["first"]) != np.sign(oos["second"]) else "no"

    print(f">= {thresh:<7} {len(hv):>5} {len(other):>7} {hv.net_portfolio_return_pct.mean()*100:>8.3f} "
          f"{other.net_portfolio_return_pct.mean()*100:>10.3f} {diff*100:>7.3f} "
          f"{oos['first']*100 if not np.isnan(oos['first']) else float('nan'):>9.3f} "
          f"{oos['second']*100 if not np.isnan(oos['second']) else float('nan'):>9.3f} {flip:>9}")

    results.append(dict(threshold=thresh, n_hv=len(hv), n_other=len(other),
                         hv_mean=hv.net_portfolio_return_pct.mean(), diff=diff,
                         oos_first=oos["first"], oos_second=oos["second"], sign_flip=flip))

print("\n=== Random-date control at each threshold (1000 draws) ===")
for r in results:
    thresh = r["threshold"]
    is_hv = merged["vol_percentile"] >= thresh
    n_hv = is_hv.sum()
    rng_diffs = []
    for seed in range(1000):
        sample = merged.sample(n=n_hv, random_state=seed, replace=False)
        rest = merged.drop(sample.index)
        rng_diffs.append(sample.net_portfolio_return_pct.mean() - rest.net_portfolio_return_pct.mean())
    rng_diffs = np.array(rng_diffs)
    pctile = (rng_diffs < r["diff"]).mean() * 100
    r["random_control_pctile"] = pctile
    print(f">= {thresh}: real diff {r['diff']*100:+.3f}pp sits at {pctile:.1f}th percentile of random-draw null "
          f"(mean {rng_diffs.mean()*100:+.3f}pp, std {rng_diffs.std()*100:.3f}pp)")

pd.DataFrame(results).to_csv("/tmp/vol_gate_threshold_sweep.csv", index=False)
print("\nSaved: /tmp/vol_gate_threshold_sweep.csv")
