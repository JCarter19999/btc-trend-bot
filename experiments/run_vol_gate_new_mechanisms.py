import numpy as np
import pandas as pd

COST_RT_PCT = 0.001599  # 0.1599% round trip, as a fraction (matches existing flat 8bps/side convention)
N_NULL = 1000

def load_features():
    df = pd.read_csv("outputs/intraday_matrix_50000/feature_frame.csv", parse_dates=["timestamp", "bar_end"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["norm_atr"] = df["atr_5m"] / df["close"]
    pct = df["norm_atr"].rolling(2016, min_periods=2016).apply(lambda x: (x < x.iloc[-1]).mean(), raw=False)
    df["regime_pctile"] = pct.shift(1)
    df["high_vol"] = df["regime_pctile"] >= 0.75
    return df

def shuffled_null_pct(n_hv, values, real_diff, seeds=N_NULL):
    vals = values.to_numpy()
    n = len(vals)
    diffs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        hv_idx, other_idx = idx[:n_hv], idx[n_hv:]
        diffs.append(vals[hv_idx].mean() - vals[other_idx].mean())
    diffs = np.array(diffs)
    return float((diffs < real_diff).mean() * 100), float(diffs.mean()), float(diffs.std())

def summarize(trades_df, label, ret_col="net_return"):
    trades_df = trades_df.dropna(subset=["high_vol", ret_col]).copy()
    trades_df["high_vol"] = trades_df["high_vol"].astype(bool)
    hv = trades_df[trades_df.high_vol]
    other = trades_df[~trades_df.high_vol]
    if len(hv) < 5 or len(other) < 5:
        print(f"{label}: too thin (n_hv={len(hv)}, n_other={len(other)})")
        return
    diff = hv[ret_col].mean() - other[ret_col].mean()
    pct, null_mean, null_std = shuffled_null_pct(len(hv), trades_df[ret_col], diff)
    trades_sorted = trades_df.sort_values("entry_timestamp")
    mid = len(trades_sorted) // 2
    def half_diff(half):
        h, o = half[half.high_vol], half[~half.high_vol]
        if len(h) < 3 or len(o) < 3:
            return None
        return float(h[ret_col].mean() - o[ret_col].mean())
    oos1, oos2 = half_diff(trades_sorted.iloc[:mid]), half_diff(trades_sorted.iloc[mid:])
    print(f"{label}: n_total={len(trades_df)} n_hv={len(hv)} n_other={len(other)} "
          f"hv_mean={hv[ret_col].mean()*100:+.3f}% other_mean={other[ret_col].mean()*100:+.3f}% "
          f"diff={diff*100:+.3f}pp null_pctile={pct:.1f} "
          f"oos1={oos1*100 if oos1 is not None else float('nan'):+.3f} "
          f"oos2={oos2*100 if oos2 is not None else float('nan'):+.3f}")
    # ALSO: is the high-vol-restricted version itself net profitable (not just relatively better)?
    print(f"    high-vol-only strategy: mean={hv[ret_col].mean()*100:+.4f}% win_rate={float((hv[ret_col]>0).mean())*100:.1f}% n={len(hv)}")

df = load_features()
print(f"Total bars: {len(df)}, classifiable: {df['regime_pctile'].notna().sum()}, high_vol share: {df['high_vol'].mean():.3f}")

# ============ Mechanism A: vol-spike mean-reversion fade ============
# Entry: |vwap_zscore| >= threshold, execute at NEXT bar open (per project convention).
# Direction: fade the extension (short if very positive z, long if very negative z).
# Hold: fixed N bars, exit at that bar's close (approximation, consistent with ad-hoc prelim conventions).
Z_THRESH = 2.0
HOLD_BARS = 6  # 30 minutes

sig = df[df["feature_valid"] & df["vwap_zscore"].abs().ge(Z_THRESH)].copy()
sig["direction"] = -np.sign(sig["vwap_zscore"])  # fade: short if z>0, long if z<0
entry_idx = sig.index + 1  # next bar open
valid = entry_idx < len(df) - HOLD_BARS
sig = sig[valid]
entry_idx = sig.index.to_numpy() + 1
exit_idx = entry_idx + HOLD_BARS

opens = df["open"].to_numpy()
closes = df["close"].to_numpy()
entry_price = opens[entry_idx]
exit_price = closes[exit_idx]
gross = sig["direction"].to_numpy() * (exit_price / entry_price - 1)
net = gross - COST_RT_PCT

fade_trades = pd.DataFrame({
    "entry_timestamp": df["timestamp"].to_numpy()[entry_idx],
    "high_vol": df["high_vol"].to_numpy()[entry_idx],
    "net_return": net,
})
print("\n--- Mechanism A: vol-spike mean-reversion fade (|vwap_zscore|>=2, 6-bar hold) ---")
summarize(fade_trades, "vol_fade_z2_6bar")

# ============ Mechanism B: wide-stop breakout, designed FOR high-vol regime ============
# Entry: breakout above prior_breakout_high (long) w/ bullish 1h momentum, or below
# prior_breakout_low (short) w/ bearish 1h momentum. Stop/target sized as ATR multiples
# -- the actual hypothesis under test: does a stop sized for the regime's own typical
# range do better than a signal not designed with vol-regime in mind.
STOP_ATR_MULT = 2.0
TARGET_ATR_MULT = 3.0
MAX_HOLD_BARS = 24  # 2 hours

long_sig = (df["close"] > df["prior_breakout_high"]) & (df["one_hour_momentum_bps"] > 0) & df["feature_valid"]
short_sig = (df["close"] < df["prior_breakout_low"]) & (df["one_hour_momentum_bps"] < 0) & df["feature_valid"]
sig_idx = df.index[long_sig | short_sig].to_numpy()
directions = np.where(long_sig.to_numpy()[sig_idx], 1, -1)

highs = df["high"].to_numpy()
lows = df["low"].to_numpy()
atrs = df["atr_5m"].to_numpy()

records = []
for i, direction in zip(sig_idx, directions):
    entry_i = i + 1
    if entry_i + MAX_HOLD_BARS >= len(df):
        continue
    entry_p = opens[entry_i]
    atr = atrs[i]
    if not np.isfinite(atr) or atr <= 0:
        continue
    stop_p = entry_p - direction * STOP_ATR_MULT * atr
    target_p = entry_p + direction * TARGET_ATR_MULT * atr
    outcome_ret = None
    for j in range(entry_i, entry_i + MAX_HOLD_BARS):
        hi, lo = highs[j], lows[j]
        if direction == 1:
            hit_stop = lo <= stop_p
            hit_target = hi >= target_p
        else:
            hit_stop = hi >= stop_p
            hit_target = lo <= target_p
        if hit_stop and hit_target:
            outcome_ret = direction * (stop_p / entry_p - 1)  # conservative: assume stop hit first if both in one bar
            break
        elif hit_stop:
            outcome_ret = direction * (stop_p / entry_p - 1)
            break
        elif hit_target:
            outcome_ret = direction * (target_p / entry_p - 1)
            break
    if outcome_ret is None:
        exit_p = closes[entry_i + MAX_HOLD_BARS - 1]
        outcome_ret = direction * (exit_p / entry_p - 1)
    records.append({
        "entry_timestamp": df["timestamp"].iloc[entry_i],
        "high_vol": df["high_vol"].iloc[i],
        "net_return": outcome_ret - COST_RT_PCT,
    })

breakout_trades = pd.DataFrame(records)
print(f"\n--- Mechanism B: wide-stop breakout ({STOP_ATR_MULT}x/{TARGET_ATR_MULT}x ATR, max {MAX_HOLD_BARS} bars) ---")
print(f"Total signals: {len(breakout_trades)}")
summarize(breakout_trades, "wide_stop_breakout")
