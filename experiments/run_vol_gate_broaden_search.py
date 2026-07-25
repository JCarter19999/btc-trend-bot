import numpy as np
import pandas as pd

N_NULL = 1000
COST_RT_PCT = 0.1599  # matches the existing flat 8bps/side round-trip, backed out empirically in the full backtest

def build_regime(intraday_feat_path):
    df = pd.read_csv(intraday_feat_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["norm_atr"] = df["atr_5m"] / df["close"]
    pct = df["norm_atr"].rolling(2016, min_periods=2016).apply(
        lambda x: (x < x.iloc[-1]).mean(), raw=False)
    df["regime_pctile"] = pct.shift(1)  # no-lookahead shift, matches prelim exactly
    df["high_vol"] = df["regime_pctile"] >= 0.75
    return df[["timestamp", "high_vol", "regime_pctile"]].dropna(subset=["regime_pctile"])

def shuffled_null_pct(hv_mask, values, real_diff, seeds=N_NULL):
    vals = values.to_numpy()
    n_hv = int(hv_mask.sum())
    n = len(vals)
    diffs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        hv_idx, other_idx = idx[:n_hv], idx[n_hv:]
        diffs.append(vals[hv_idx].mean() - vals[other_idx].mean())
    diffs = np.array(diffs)
    return float((diffs < real_diff).mean() * 100), float(diffs.mean()), float(diffs.std())

def check_candidate(trades, regime, label, return_col="net_portfolio_return_pct"):
    trades = trades.copy()
    trades["entry_timestamp"] = pd.to_datetime(trades["entry_timestamp"])
    trades = trades.sort_values("entry_timestamp")
    merged = pd.merge_asof(trades, regime, left_on="entry_timestamp", right_on="timestamp", direction="backward")
    before = len(trades)
    merged = merged.dropna(subset=["high_vol", return_col])
    merged["high_vol"] = merged["high_vol"].astype(bool)
    dropped = before - len(merged)
    if len(merged) < 10 or merged["high_vol"].sum() < 5 or (~merged["high_vol"]).sum() < 5:
        return {"label": label, "n": len(merged), "dropped": dropped, "status": "too_thin"}

    hv = merged[merged.high_vol]
    other = merged[~merged.high_vol]
    diff = hv[return_col].mean() - other[return_col].mean()
    pct, null_mean, null_std = shuffled_null_pct(merged.high_vol, merged[return_col], diff)

    # Early OOS split -- checked alongside significance, not after
    merged_sorted = merged.sort_values("entry_timestamp")
    mid = len(merged_sorted) // 2
    first, second = merged_sorted.iloc[:mid], merged_sorted.iloc[mid:]
    def half_diff(half):
        h, o = half[half.high_vol], half[~half.high_vol]
        if len(h) < 3 or len(o) < 3:
            return None
        return float(h[return_col].mean() - o[return_col].mean())
    oos_first, oos_second = half_diff(first), half_diff(second)
    oos_flip = (oos_first is not None and oos_second is not None and
                np.sign(oos_first) != np.sign(oos_second) and abs(oos_first) > 0.01 and abs(oos_second) > 0.01)

    return {
        "label": label, "n_total": len(merged), "n_hv": int(hv.shape[0]), "n_other": int(other.shape[0]),
        "dropped_in_merge": dropped, "hv_mean": float(hv[return_col].mean()), "other_mean": float(other[return_col].mean()),
        "diff_pp": diff, "null_percentile": pct, "null_mean": null_mean, "null_std": null_std,
        "oos_first_half_diff": oos_first, "oos_second_half_diff": oos_second, "oos_sign_flip": oos_flip,
        "status": "ok",
    }

regime = build_regime("outputs/intraday_matrix_50000/feature_frame.csv")
print(f"Regime rows usable: {len(regime)}, high_vol share: {regime['high_vol'].mean():.3f}")

results = []
for matrix_name, path in [("popular_matrix_50000", "outputs/popular_matrix_50000/trade_episodes.csv"),
                           ("slope_matrix_50000", "outputs/slope_matrix_50000/trade_episodes.csv")]:
    trades_all = pd.read_csv(path)
    for sid in trades_all.strategy_id.unique():
        if sid in ("buy_hold_5m", "strategy_id"):
            continue
        sub = trades_all[(trades_all.strategy_id == sid) & (trades_all.is_open == False)]
        if sub.empty:
            continue
        res = check_candidate(sub, regime, f"{matrix_name}::{sid}")
        results.append(res)

for r in results:
    print(r)
