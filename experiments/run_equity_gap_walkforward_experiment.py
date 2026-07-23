from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from btc_trend_bot.v1.candlestick_geometry import CONTINUOUS_GEOMETRY_COLUMNS, add_candlestick_geometry
from run_equity_three_symbol_experiment import (
    BARS_PER_SESSION,
    SYMBOLS,
    SPECS,
    BASE_FEATURES,
    GEOMETRY_FEATURES,
    candidate_mask,
    trading_index,
    regime_for_session,
)


@dataclass(frozen=True)
class ShockSpec:
    frequency_sessions: int
    positive_probability: float
    mean_abs_gap: float
    sigma_gap: float


SHOCKS = {
    "AAPL": ShockSpec(63, 0.56, 0.035, 0.018),
    "TSLA": ShockSpec(63, 0.52, 0.065, 0.035),
    "MSFT": ShockSpec(63, 0.58, 0.030, 0.015),
}


def generate_market_with_shocks(sessions: int, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = trading_index("2019-01-02", sessions)
    market_returns = np.zeros(len(index))
    for i in range(len(index)):
        session = i // BARS_PER_SESSION
        drift, vol_factor = regime_for_session(session, sessions)
        opening_bar = i % BARS_PER_SESSION == 0
        sigma = 0.0024 * vol_factor * (1.4 if opening_bar else 1.0)
        market_returns[i] = drift / BARS_PER_SESSION + rng.normal(0.0, sigma)

    shock_rows: list[dict] = []
    shock_maps: dict[str, dict[int, float]] = {}
    for symbol in SYMBOLS:
        spec = SHOCKS[symbol]
        shocks: dict[int, float] = {}
        offset = {"AAPL": 12, "TSLA": 31, "MSFT": 48}[symbol]
        for session in range(offset, sessions, spec.frequency_sessions):
            positive = rng.random() < spec.positive_probability
            magnitude = abs(rng.normal(spec.mean_abs_gap, spec.sigma_gap))
            gap = magnitude if positive else -magnitude
            shocks[session] = gap
            shock_rows.append({"symbol": symbol, "session": session, "gap_return": gap, "positive": positive})
        shock_maps[symbol] = shocks

    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        spec = SPECS[symbol]
        rows = []
        previous_close = spec.start_price
        latent_edge = 0.0
        for i, timestamp in enumerate(index):
            bar_in_session = i % BARS_PER_SESSION
            opening_bar = bar_in_session == 0
            session = i // BARS_PER_SESSION
            _, vol_factor = regime_for_session(session, sessions)
            routine_gap = rng.normal(0.0, spec.overnight_sigma * vol_factor) if opening_bar else 0.0
            event_gap = shock_maps[symbol].get(session, 0.0) if opening_bar else 0.0
            gap = routine_gap + event_gap
            open_price = previous_close * math.exp(gap)

            if bar_in_session in (2, 7) and rng.random() < 0.17:
                quality = rng.normal()
                # Stronger cross-sectional planted relationship, but with symbol-specific scaling.
                latent_edge = spec.edge_scale * (0.0048 * quality)
            else:
                latent_edge *= 0.88

            # Shock sessions remain noisy after the open so gaps cannot be treated as free profits.
            shock_vol = 1.8 if session in shock_maps[symbol] else 1.0
            idio = rng.normal(0.0, spec.intraday_sigma * vol_factor * 0.72 * shock_vol)
            ret = spec.beta * market_returns[i] + idio + latent_edge
            close = open_price * math.exp(ret)
            base_range = abs(ret) + spec.intraday_sigma * vol_factor * 0.55 * shock_vol
            upper_share = np.clip(0.50 - np.sign(latent_edge) * 0.16 + rng.normal(0, 0.10), 0.08, 0.92)
            extra = open_price * base_range
            high = max(open_price, close) + extra * upper_share
            low = min(open_price, close) - extra * (1.0 - upper_share)
            volume = (
                (1.0 + 0.40 * abs(latent_edge) / 0.002)
                * (1.0 + 0.55 * (opening_bar or bar_in_session == 12))
                * (2.2 if session in shock_maps[symbol] else 1.0)
                * rng.lognormal(12.0 if symbol != "TSLA" else 11.5, 0.32)
            )
            rows.append((timestamp, open_price, high, low, close, volume, session, bar_in_session, event_gap))
            previous_close = close
        frames[symbol] = pd.DataFrame(
            rows,
            columns=["open_time", "open", "high", "low", "close", "volume", "session", "bar_in_session", "event_gap"],
        )
    return frames, pd.DataFrame(shock_rows)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat([
        out["high"] - out["low"],
        (out["high"] - previous_close).abs(),
        (out["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = true_range.ewm(span=20, adjust=False, min_periods=20).mean()
    for bars in (1, 2, 4, 8, 13, 26):
        out[f"return_{bars}"] = close.pct_change(bars)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=36, adjust=False).mean()
    out["ema_spread_atr"] = (fast - slow) / out["atr"].replace(0, np.nan)
    out["ema_slope_atr"] = fast.diff(3) / out["atr"].replace(0, np.nan)
    out["atr_pct"] = out["atr"] / close
    out["realized_vol_13"] = out["return_1"].rolling(13).std(ddof=0)
    volume_mean = out["volume"].rolling(26).mean()
    volume_std = out["volume"].rolling(26).std(ddof=0)
    out["relative_volume"] = out["volume"] / volume_mean.replace(0, np.nan)
    out["volume_zscore"] = (out["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    out["distance_session_high_atr"] = (out.groupby("session")["high"].cummax() - close) / out["atr"]
    out["distance_session_low_atr"] = (close - out.groupby("session")["low"].cummin()) / out["atr"]
    out["opening_bar"] = (out["bar_in_session"] == 0).astype(float)
    out["closing_bar"] = (out["bar_in_session"] == 12).astype(float)
    out["bar_sin"] = np.sin(2 * np.pi * out["bar_in_session"] / BARS_PER_SESSION)
    out["bar_cos"] = np.cos(2 * np.pi * out["bar_in_session"] / BARS_PER_SESSION)
    out["overnight_gap_abs"] = out["open"].div(previous_close).sub(1.0).abs().fillna(0.0)
    out["shock_volume_flag"] = (out["relative_volume"] > 2.0).astype(float)
    return add_candlestick_geometry(out)


EXTRA_FEATURES = ["overnight_gap_abs", "shock_volume_flag"]
GEOMETRY_EXTENDED = GEOMETRY_FEATURES + EXTRA_FEATURES
BASE_EXTENDED = BASE_FEATURES + EXTRA_FEATURES


def simulate_trade(frame: pd.DataFrame, index: int, max_bars: int) -> dict:
    entry_index = index + 1
    if entry_index >= len(frame):
        return {"valid": False}
    entry = float(frame.iloc[entry_index].open)
    atr = float(frame.iloc[index].atr)
    stop = entry - 1.35 * atr
    target = entry + 2.15 * atr
    dynamic_stop = stop
    peak = entry
    mfe = 0.0
    mae = 0.0
    exit_price = float(frame.iloc[min(entry_index + max_bars - 1, len(frame) - 1)].close)
    exit_reason = "time_exit"
    bars_held = 0
    gap_through_stop = False
    gap_through_target = False

    for j in range(entry_index, min(entry_index + max_bars, len(frame))):
        row = frame.iloc[j]
        bars_held = j - entry_index + 1
        open_price = float(row.open)
        high = float(row.high)
        low = float(row.low)
        close = float(row.close)

        # At a new session open, stops and targets fill at the opening print if gapped through.
        if int(row.bar_in_session) == 0 and bars_held > 1:
            if open_price <= dynamic_stop:
                exit_price = open_price
                exit_reason = "gap_through_stop"
                gap_through_stop = True
                mae = min(mae, open_price / entry - 1.0)
                break
            if open_price >= target:
                exit_price = open_price
                exit_reason = "gap_through_target"
                gap_through_target = True
                mfe = max(mfe, open_price / entry - 1.0)
                break

        peak = max(peak, high)
        mfe = max(mfe, high / entry - 1.0)
        mae = min(mae, low / entry - 1.0)
        if low <= dynamic_stop and high >= target:
            exit_price = dynamic_stop
            exit_reason = "ambiguous_stop_first"
            break
        if low <= dynamic_stop:
            exit_price = dynamic_stop
            exit_reason = "stop"
            break
        if high >= target:
            exit_price = target
            exit_reason = "target"
            break

        current_atr = float(row.atr) if np.isfinite(row.atr) else atr
        vol_factor = float(np.clip(current_atr / max(atr, 1e-12), 1.0, 3.2))
        if peak >= entry + 1.10 * atr:
            dynamic_stop = max(dynamic_stop, entry + 0.0002 * entry)
        if peak >= entry + 1.70 * atr:
            trailing_distance = 1.15 * atr * (vol_factor ** 0.95)
            dynamic_stop = max(dynamic_stop, peak - trailing_distance)

        minimum_soft_hold = int(round(5 * vol_factor))
        if bars_held >= minimum_soft_hold:
            recent_start = max(entry_index, j - 5)
            momentum = close / float(frame.iloc[recent_start].close) - 1.0
            threshold = -0.0014 * (vol_factor ** 0.9)
            if momentum < threshold and close < float(frame.iloc[j - 1].close):
                exit_price = close
                exit_reason = "momentum_decay"
                break
        exit_price = close

    gross = exit_price / entry - 1.0
    net = gross - 0.0006
    return {
        "valid": True,
        "entry_time": frame.iloc[entry_index].open_time,
        "exit_time": frame.iloc[min(entry_index + bars_held - 1, len(frame)-1)].open_time,
        "entry_price": entry,
        "exit_price": exit_price,
        "net_return": net,
        "gross_return": gross,
        "bars_held": bars_held,
        "exit_reason": exit_reason,
        "mfe": mfe,
        "mae": mae,
        "gap_through_stop": gap_through_stop,
        "gap_through_target": gap_through_target,
    }


def build_candidates(frames: dict[str, pd.DataFrame], max_bars: int) -> pd.DataFrame:
    records: list[dict] = []
    for symbol, raw in frames.items():
        frame = add_features(raw)
        mask = candidate_mask(frame)
        for idx in np.flatnonzero(mask.to_numpy()):
            outcome = simulate_trade(frame, int(idx), max_bars)
            if not outcome["valid"]:
                continue
            row = frame.iloc[idx]
            rec = {column: float(row[column]) for column in sorted(set(GEOMETRY_EXTENDED))}
            rec.update(outcome)
            rec.update({"symbol": symbol, "signal_time": row.open_time, "session": int(row.session)})
            records.append(rec)
    return pd.DataFrame(records).sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame, features: list[str], model_kind: str) -> Pipeline:
    transformer = ColumnTransformer([
        ("continuous", StandardScaler(), features),
        ("symbol", OneHotEncoder(handle_unknown="ignore"), ["symbol"]),
    ])
    if model_kind == "ridge":
        estimator = Ridge(alpha=20.0)
    else:
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.045,
            max_iter=260,
            max_leaf_nodes=11,
            min_samples_leaf=32,
            l2_regularization=2.5,
            random_state=73,
        )
    model = Pipeline([("transform", transformer), ("estimator", estimator)])
    model.fit(train[features + ["symbol"]], train["net_return"])
    return model


def select_portfolio(test: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ranked = test.sort_values(["signal_time", "predicted_net_return"], ascending=[True, False])
    ranked = ranked.groupby("signal_time", as_index=False).head(1)
    ranked = ranked[ranked.predicted_net_return > threshold].copy()
    accepted = []
    next_free = pd.Timestamp.min.tz_localize("UTC")
    for _, row in ranked.sort_values("signal_time").iterrows():
        if pd.Timestamp(row.signal_time) < next_free:
            continue
        accepted.append(row)
        next_free = pd.Timestamp(row.exit_time)
    return pd.DataFrame(accepted, columns=ranked.columns)


def portfolio_metrics(selected: pd.DataFrame) -> tuple[float, float]:
    equity = 1.0
    peak = 1.0
    dd = 0.0
    for ret in selected.net_return.to_numpy(float):
        equity *= 1.0 + ret
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / peak)
    return equity - 1.0, dd


def walk_forward(candidates: pd.DataFrame, features: list[str], model_kind: str, threshold_bps: float, folds: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = np.sort(candidates.signal_time.unique())
    boundaries = [int(len(times) * x) for x in (0.40, 0.55, 0.70, 0.85, 1.00)]
    fold_rows = []
    prediction_frames = []
    for fold in range(folds):
        train_end = times[boundaries[fold]]
        test_end = times[boundaries[fold + 1] - 1]
        train_all = candidates[candidates.signal_time < train_end].copy()
        test = candidates[(candidates.signal_time >= train_end) & (candidates.signal_time <= test_end)].copy()
        if len(train_all) < 150 or len(test) < 25:
            continue
        cut = int(len(train_all) * 0.80)
        train = train_all.iloc[:cut].copy()
        calibration = train_all.iloc[cut:].copy()
        model = fit_model(train, features, model_kind)
        cal_raw = model.predict(calibration[features + ["symbol"]])
        slope, intercept = np.polyfit(cal_raw, calibration.net_return.to_numpy(float), 1)
        raw = model.predict(test[features + ["symbol"]])
        test["predicted_net_return"] = intercept + slope * raw
        selected = select_portfolio(test, threshold_bps / 10_000.0)
        ret, dd = portfolio_metrics(selected)
        fold_rows.append({
            "fold": fold + 1,
            "model_kind": model_kind,
            "threshold_bps": threshold_bps,
            "train_candidates": len(train),
            "calibration_candidates": len(calibration),
            "test_candidates": len(test),
            "selected_trades": len(selected),
            "selected_expectancy_bps": selected.net_return.mean() * 10_000 if len(selected) else np.nan,
            "selected_win_rate": (selected.net_return > 0).mean() if len(selected) else np.nan,
            "portfolio_return": ret,
            "max_drawdown": dd,
            "test_mae_bps": mean_absolute_error(test.net_return, test.predicted_net_return) * 10_000,
            "test_r2": r2_score(test.net_return, test.predicted_net_return),
            "gap_stop_count": int(selected.gap_through_stop.sum()) if len(selected) else 0,
            "symbol_counts": json.dumps(selected.symbol.value_counts().to_dict()),
        })
        test["fold"] = fold + 1
        test["selected"] = test.index.isin(selected.index)
        prediction_frames.append(test)
    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def buy_hold(frames: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    curves = []
    for symbol, frame in frames.items():
        subset = frame[(frame.open_time >= start) & (frame.open_time <= end)]
        normalized = subset.close.astype(float) / float(subset.iloc[0].open)
        rows.append({
            "benchmark": f"{symbol}_buy_hold",
            "return": float(normalized.iloc[-1] - 1.0 - 0.0006),
            "max_drawdown": float(((normalized.cummax() - normalized) / normalized.cummax()).max()),
        })
        curves.append(normalized.reset_index(drop=True))
    basket = pd.concat(curves, axis=1).mean(axis=1)
    rows.append({
        "benchmark": "equal_weight_buy_hold",
        "return": float(basket.iloc[-1] - 1.0 - 0.0006),
        "max_drawdown": float(((basket.cummax() - basket) / basket.cummax()).max()),
    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=1300)
    parser.add_argument("--seed", type=int, default=9381)
    parser.add_argument("--output", default="outputs/equity_gap_walkforward")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    frames, shocks = generate_market_with_shocks(args.sessions, args.seed)
    all_summaries = []
    all_folds = []
    for hold_name, max_bars in {"1_session": 13, "2_sessions": 26, "5_sessions": 65}.items():
        candidates = build_candidates(frames, max_bars)
        candidates.to_csv(output / f"candidates_{hold_name}.csv", index=False)
        for feature_name, features in {"baseline": BASE_EXTENDED, "geometry": GEOMETRY_EXTENDED}.items():
            for model_kind in ("ridge", "tree"):
                for threshold in (5.0, 10.0, 15.0):
                    folds, predictions = walk_forward(candidates, features, model_kind, threshold)
                    if folds.empty:
                        continue
                    folds.insert(0, "hold", hold_name)
                    folds.insert(1, "feature_set", feature_name)
                    all_folds.append(folds)
                    summary = {
                        "hold": hold_name,
                        "feature_set": feature_name,
                        "model_kind": model_kind,
                        "threshold_bps": threshold,
                        "folds": len(folds),
                        "positive_folds": int((folds.portfolio_return > 0).sum()),
                        "total_selected_trades": int(folds.selected_trades.sum()),
                        "mean_selected_expectancy_bps": float(folds.selected_expectancy_bps.mean()),
                        "median_selected_expectancy_bps": float(folds.selected_expectancy_bps.median()),
                        "mean_portfolio_return": float(folds.portfolio_return.mean()),
                        "compounded_fold_return": float(np.prod(1.0 + folds.portfolio_return) - 1.0),
                        "worst_fold_return": float(folds.portfolio_return.min()),
                        "max_fold_drawdown": float(folds.max_drawdown.max()),
                        "mean_test_r2": float(folds.test_r2.mean()),
                        "gap_stop_count": int(folds.gap_stop_count.sum()),
                    }
                    all_summaries.append(summary)
                    predictions.to_csv(output / f"pred_{hold_name}_{feature_name}_{model_kind}_{int(threshold)}bps.csv", index=False)

    summary_df = pd.DataFrame(all_summaries).sort_values(
        ["compounded_fold_return", "worst_fold_return"], ascending=[False, False]
    )
    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df.to_csv(output / "summary.csv", index=False)
    folds_df.to_csv(output / "fold_results.csv", index=False)
    shocks.to_csv(output / "earnings_like_shocks.csv", index=False)

    earliest = folds_df.fold.min() if not folds_df.empty else 1
    times = np.sort(build_candidates(frames, 26).signal_time.unique())
    start = pd.Timestamp(times[int(len(times) * 0.40)])
    end = pd.Timestamp(times[-1])
    benchmarks = buy_hold(frames, start, end)
    benchmarks.to_csv(output / "benchmarks.csv", index=False)

    best = summary_df.iloc[0].to_dict()
    report = {"best_configuration": best, "benchmarks": benchmarks.to_dict(orient="records")}
    (output / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(summary_df.head(15).to_string(index=False))
    print("\nBenchmarks")
    print(benchmarks.to_string(index=False))


if __name__ == "__main__":
    main()
