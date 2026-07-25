import numpy as np
import pandas as pd

ff = pd.read_csv("/home/joey/btc-paper-5m/outputs/intraday_matrix_50000/feature_frame.csv", parse_dates=["timestamp"])
ff = ff.sort_values("timestamp").reset_index(drop=True)
ff["norm_atr"] = ff["atr_5m"] / ff["close"]
pct = ff["norm_atr"].rolling(2016, min_periods=2016).apply(lambda w: (w < w.iloc[-1]).mean(), raw=False)
ff["vol_percentile"] = pct.shift(1)
ff["high_vol_regime"] = ff["vol_percentile"] >= 0.75

te = pd.read_csv("/home/joey/btc-paper-5m/outputs/popular_matrix_50000/trade_episodes.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
rsi = te[(te.strategy_id == "rsi2_reversion_5m_4h_filter") & (te.is_open == False)].copy().sort_values("entry_timestamp").reset_index(drop=True)
print(f"rsi2_reversion_5m_4h_filter total closed trades: {len(rsi)}")

ff_idx = ff.set_index("timestamp")
merged = pd.merge_asof(rsi, ff_idx[["vol_percentile", "high_vol_regime"]].reset_index(),
                        left_on="entry_timestamp", right_on="timestamp", direction="backward")
merged_before = len(merged)
merged = merged.dropna(subset=["high_vol_regime"]).reset_index(drop=True)
print(f"merged with regime label: {len(merged)} / {len(rsi)} (dropped {merged_before - len(merged)}, expected)")
print(f"high_vol_regime dtype: {merged['high_vol_regime'].dtype}\n")

flat_cost = (merged["gross_price_return_pct"] - merged["net_portfolio_return_pct"]).mean()
print(f"Backed-out flat round-trip cost: {flat_cost*100:.4f}%")

hv = merged[merged.high_vol_regime]
other = merged[~merged.high_vol_regime]
baseline_diff = hv.net_portfolio_return_pct.mean() - other.net_portfolio_return_pct.mean()
print(f"\nBaseline: n_hv={len(hv)} mean={hv.net_portfolio_return_pct.mean()*100:+.3f}%  "
      f"n_other={len(other)} mean={other.net_portfolio_return_pct.mean()*100:+.3f}%  diff={baseline_diff*100:+.3f}pp")

print("\n=== CHECK 1: volatility-conditioned cost stress ===")
for mult in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
    hv_net = merged.loc[merged.high_vol_regime, "gross_price_return_pct"] - flat_cost * mult
    hv_mean = hv_net.mean()
    diff = hv_mean - other.net_portfolio_return_pct.mean()
    print(f"  cost x{mult:<4}: hv net mean={hv_mean*100:+.3f}%  diff vs other={diff*100:+.3f}pp")

print("\n=== CHECK 2: out-of-sample split (formal, chronological) ===")
merged_sorted = merged.sort_values("entry_timestamp").reset_index(drop=True)
mid = len(merged_sorted) // 2
first_half, second_half = merged_sorted.iloc[:mid], merged_sorted.iloc[mid:]
oos_diffs = {}
for label, half in (("First half", first_half), ("Second half", second_half)):
    h = half[half.high_vol_regime]
    o = half[~half.high_vol_regime]
    if len(h) == 0 or len(o) == 0:
        print(f"  {label}: n_hv={len(h)} n_other={len(o)} -- one side empty, can't compare")
        continue
    diff = h.net_portfolio_return_pct.mean() - o.net_portfolio_return_pct.mean()
    oos_diffs[label] = diff
    print(f"  {label} ({half.entry_timestamp.min().date()} to {half.entry_timestamp.max().date()}): "
          f"n_hv={len(h)} hv_mean={h.net_portfolio_return_pct.mean()*100:+.3f}%  "
          f"n_other={len(o)} other_mean={o.net_portfolio_return_pct.mean()*100:+.3f}%  diff={diff*100:+.3f}pp")
sign_flip = None
if len(oos_diffs) == 2:
    vals = list(oos_diffs.values())
    sign_flip = np.sign(vals[0]) != np.sign(vals[1])
    print(f"  Sign flip: {sign_flip}")

print("\n=== CHECK 3: random-date control (1000 draws) ===")
n_hv = len(hv)
rng_diffs = []
for seed in range(1000):
    sample = merged.sample(n=n_hv, random_state=seed, replace=False)
    rest = merged.drop(sample.index)
    rng_diffs.append(sample.net_portfolio_return_pct.mean() - rest.net_portfolio_return_pct.mean())
rng_diffs = np.array(rng_diffs)
pctile = (rng_diffs < baseline_diff).mean() * 100
print(f"Real diff: {baseline_diff*100:+.3f}pp | random-sample null: mean={rng_diffs.mean()*100:+.3f}pp "
      f"std={rng_diffs.std()*100:.3f}pp | real diff sits at {pctile:.1f}th percentile")

print("\n=== COMBINED with ema_pullback_15m_4h (both strategies, same regime, $10,000 sequential compounding) ===")
ep = te[(te.strategy_id == "ema_pullback_15m_4h") & (te.is_open == False)].copy().sort_values("entry_timestamp").reset_index(drop=True)
ep_merged = pd.merge_asof(ep, ff_idx[["vol_percentile", "high_vol_regime"]].reset_index(),
                           left_on="entry_timestamp", right_on="timestamp", direction="backward").dropna(subset=["high_vol_regime"])
ep_hv = ep_merged[ep_merged.high_vol_regime].copy()
ep_hv["source"] = "ema_pullback_15m_4h"
rsi_hv = hv.copy()
rsi_hv["source"] = "rsi2_reversion_5m_4h_filter"
combined = pd.concat([ep_hv, rsi_hv]).sort_values("entry_timestamp").reset_index(drop=True)
n_days = (combined.entry_timestamp.max() - combined.entry_timestamp.min()).days

capital = 10000.0
for x in combined["net_portfolio_return_pct"]:
    capital *= (1 + x)
print(f"Combined trade count: {len(combined)} ({len(ep_hv)} ema_pullback + {len(rsi_hv)} rsi2_reversion)")
print(f"Window: {combined.entry_timestamp.min().date()} to {combined.entry_timestamp.max().date()} ({n_days} days)")
print(f"Combined trades/day: {len(combined)/n_days:.3f} (~1 every {n_days/len(combined):.1f} days) "
      f"vs ema_pullback-alone's ~1 every {n_days/len(ep_hv):.1f} days")
print(f"Combined $10,000 sequential compounding -> ${capital:,.2f} ({(capital/10000-1)*100:+.3f}%)")
print(f"Combined win rate: {(combined.net_portfolio_return_pct>0).mean()*100:.1f}%")
