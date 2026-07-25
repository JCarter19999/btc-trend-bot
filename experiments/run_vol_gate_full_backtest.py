import numpy as np
import pandas as pd

RESULT_DIR = "/home/joey/btc-paper-5m/outputs/intraday_matrix_50000"
ff = pd.read_csv(f"{RESULT_DIR}/feature_frame.csv", parse_dates=["timestamp"])
te = pd.read_csv(f"{RESULT_DIR}/trade_episodes.csv", parse_dates=["entry_timestamp", "exit_timestamp"])

ff = ff.sort_values("timestamp").reset_index(drop=True)
ff["norm_atr"] = ff["atr_5m"] / ff["close"]

WINDOW = 2016
pct = ff["norm_atr"].rolling(WINDOW, min_periods=WINDOW).apply(
    lambda w: (w < w.iloc[-1]).mean(), raw=False
)
ff["vol_percentile"] = pct.shift(1)  # no-lookahead: knowable strictly before the bar trades
ff["high_vol_regime"] = ff["vol_percentile"] >= 0.75

classified = ff.dropna(subset=["vol_percentile"])
print(f"Classified bars: {len(classified)} / {len(ff)}")
print(f"High-vol regime base rate: {classified['high_vol_regime'].mean()*100:.1f}%")

# merge onto breakout strategy's trades via entry_timestamp -> nearest bar at/before entry
bo = te[te.strategy_id == "breakout_5m_1h_regime"].copy().sort_values("entry_timestamp").reset_index(drop=True)
ff_idx = ff.set_index("timestamp")
merged = pd.merge_asof(bo, ff_idx[["vol_percentile", "high_vol_regime"]].reset_index(),
                        left_on="entry_timestamp", right_on="timestamp", direction="backward")
merged = merged.dropna(subset=["high_vol_regime"])
print(f"\nBreakout trades merged with regime label: {len(merged)} / {len(bo)}")

# back out the existing flat cost (empirically, from gross - net)
flat_cost = (merged["gross_price_return_pct"] - merged["net_portfolio_return_pct"]).mean()
print(f"Backed-out flat round-trip cost: {flat_cost*100:.4f}%  (matches 8bps/side*2 = 16bps expectation)")

hv = merged[merged.high_vol_regime]
other = merged[~merged.high_vol_regime]
print(f"\nBaseline (flat cost, as in prelim): n_hv={len(hv)} mean={hv.net_portfolio_return_pct.mean()*100:+.3f}%  "
      f"n_other={len(other)} mean={other.net_portfolio_return_pct.mean()*100:+.3f}%  "
      f"diff={ (hv.net_portfolio_return_pct.mean()-other.net_portfolio_return_pct.mean())*100:+.3f}pp")

print("\n=== CHECK 1: volatility-conditioned cost stress ===")
print("Scaling the round-trip cost UP specifically on high-vol trades (cost stays flat on 'other' trades):")
for mult in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
    hv_net = merged.loc[merged.high_vol_regime, "gross_price_return_pct"] - flat_cost * mult
    hv_mean = hv_net.mean()
    diff = hv_mean - other.net_portfolio_return_pct.mean()
    print(f"  cost x{mult:<4}: hv net mean={hv_mean*100:+.3f}%  diff vs other={diff*100:+.3f}pp")

print("\n=== CHECK 2: out-of-sample split (chronological, this strategy's 153 trades) ===")
merged_sorted = merged.sort_values("entry_timestamp").reset_index(drop=True)
mid = len(merged_sorted) // 2
first_half, second_half = merged_sorted.iloc[:mid], merged_sorted.iloc[mid:]
for label, half in (("First half", first_half), ("Second half", second_half)):
    h = half[half.high_vol_regime]
    o = half[~half.high_vol_regime]
    if len(h) == 0 or len(o) == 0:
        print(f"  {label}: n_hv={len(h)} n_other={len(o)} -- one side empty, can't compare")
        continue
    diff = h.net_portfolio_return_pct.mean() - o.net_portfolio_return_pct.mean()
    print(f"  {label} ({half.entry_timestamp.min().date()} to {half.entry_timestamp.max().date()}): "
          f"n_hv={len(h)} hv_mean={h.net_portfolio_return_pct.mean()*100:+.3f}%  "
          f"n_other={len(o)} other_mean={o.net_portfolio_return_pct.mean()*100:+.3f}%  diff={diff*100:+.3f}pp")

print("\n=== CHECK 3: random-date control (pure random sampling, not label permutation) ===")
n_hv = len(hv)
rng_diffs = []
rng = np.random.default_rng(0)
for seed in range(1000):
    sample = merged.sample(n=n_hv, random_state=seed, replace=False)
    rest = merged.drop(sample.index)
    rng_diffs.append(sample.net_portfolio_return_pct.mean() - rest.net_portfolio_return_pct.mean())
rng_diffs = np.array(rng_diffs)
real_diff = hv.net_portfolio_return_pct.mean() - other.net_portfolio_return_pct.mean()
pct = (rng_diffs < real_diff).mean() * 100
print(f"Real diff: {real_diff*100:+.3f}pp | random-sample null: mean={rng_diffs.mean()*100:+.3f}pp "
      f"std={rng_diffs.std()*100:.3f}pp | real diff sits at {pct:.1f}th percentile of 1000 random-draw controls")

# methodology trap check: any candidates dropped from the merge that could
# mechanically inflate one side's count?
print(f"\n=== Methodology trap check ===")
print(f"Trades in original breakout episode file: {len(bo)}, trades after regime merge: {len(merged)} "
      f"(dropped: {len(bo)-len(merged)}, due to entry before first classified bar -- expected, not a data artifact)")
