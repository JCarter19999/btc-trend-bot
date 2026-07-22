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
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.v1.candlestick_geometry import (
    CONTINUOUS_GEOMETRY_COLUMNS,
    add_candlestick_geometry,
)

SYMBOLS = ("AAPL", "TSLA", "MSFT")
BARS_PER_SESSION = 13  # 30-minute regular-session bars, 9:30–16:00 ET.


@dataclass(frozen=True)
class SymbolSpec:
    start_price: float
    beta: float
    intraday_sigma: float
    overnight_sigma: float
    edge_scale: float


SPECS = {
    "AAPL": SymbolSpec(145.0, 1.00, 0.0032, 0.0080, 1.00),
    "TSLA": SymbolSpec(220.0, 1.45, 0.0060, 0.0160, 1.25),
    "MSFT": SymbolSpec(265.0, 0.88, 0.0028, 0.0070, 0.92),
}


def trading_index(start: str, sessions: int) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, periods=sessions, tz="America/New_York")
    stamps: list[pd.Timestamp] = []
    for day in days:
        opening = day.normalize() + pd.Timedelta(hours=9, minutes=30)
        stamps.extend(opening + pd.to_timedelta(np.arange(BARS_PER_SESSION) * 30, unit="min"))
    return pd.DatetimeIndex(stamps).tz_convert("UTC")


def regime_for_session(session: int, sessions: int) -> tuple[float, float]:
    phase = session / sessions
    if phase < 0.20:  # moderate bull
        return 0.00045, 0.85
    if phase < 0.38:  # sideways
        return 0.00002, 0.95
    if phase < 0.55:  # correction
        return -0.00038, 1.20
    if phase < 0.72:  # volatile recovery
        return 0.00020, 1.65
    return 0.00028, 1.00


def generate_market(sessions: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = trading_index("2019-01-02", sessions)
    market_returns = np.zeros(len(index))
    market_vol = np.zeros(len(index))
    for i in range(len(index)):
        session = i // BARS_PER_SESSION
        drift, vol_factor = regime_for_session(session, sessions)
        opening_bar = i % BARS_PER_SESSION == 0
        sigma = 0.0024 * vol_factor * (1.4 if opening_bar else 1.0)
        market_returns[i] = drift / BARS_PER_SESSION + rng.normal(0.0, sigma)
        market_vol[i] = vol_factor

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
            gap = rng.normal(0.0, spec.overnight_sigma * vol_factor) if opening_bar else 0.0
            open_price = previous_close * math.exp(gap)

            # Edge episodes are deliberately observable through continuous
            # geometry, trend, relative volume and symbol identity. They are
            # not exposed as a hidden label to the model.
            if bar_in_session in (2, 7) and rng.random() < 0.16:
                quality = rng.normal()
                latent_edge = spec.edge_scale * (0.0042 * quality)
            else:
                latent_edge *= 0.90

            idio = rng.normal(0.0, spec.intraday_sigma * vol_factor * 0.72)
            ret = spec.beta * market_returns[i] + idio + latent_edge
            close = open_price * math.exp(ret)
            base_range = abs(ret) + spec.intraday_sigma * vol_factor * 0.55
            upper_share = np.clip(0.50 - np.sign(latent_edge) * 0.16 + rng.normal(0, 0.10), 0.08, 0.92)
            extra = open_price * base_range
            high = max(open_price, close) + extra * upper_share
            low = min(open_price, close) - extra * (1.0 - upper_share)
            volume = (
                (1.0 + 0.35 * abs(latent_edge) / 0.002)
                * (1.0 + 0.45 * (opening_bar or bar_in_session == 12))
                * rng.lognormal(12.0 if symbol != "TSLA" else 11.5, 0.32)
            )
            rows.append((timestamp, open_price, high, low, close, volume, session, bar_in_session))
            previous_close = close
        frames[symbol] = pd.DataFrame(
            rows,
            columns=["open_time", "open", "high", "low", "close", "volume", "session", "bar_in_session"],
        )
    return frames


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
    out = add_candlestick_geometry(out)
    return out


BASE_FEATURES = [
    "return_1", "return_2", "return_4", "return_8", "return_13", "return_26",
    "ema_spread_atr", "ema_slope_atr", "atr_pct", "realized_vol_13",
    "relative_volume", "volume_zscore", "distance_session_high_atr",
    "distance_session_low_atr", "opening_bar", "closing_bar", "bar_sin", "bar_cos",
]
GEOMETRY_FEATURES = BASE_FEATURES + CONTINUOUS_GEOMETRY_COLUMNS


def candidate_mask(frame: pd.DataFrame) -> pd.Series:
    green_2 = (frame["close"] > frame["open"]) & (frame["close"].shift(1) > frame["open"].shift(1))
    return (
        green_2
        & (frame["ema_spread_atr"] > -0.15)
        & (frame["relative_volume"] > 1.05)
        & frame["atr"].notna()
        & (frame["bar_in_session"] <= 9)
    )


def simulate_trade(frame: pd.DataFrame, index: int, max_bars: int = 26) -> dict:
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
    exit_price = float(frame.iloc[min(entry_index + max_bars, len(frame) - 1)].close)
    exit_reason = "time_exit"
    bars_held = 0

    for j in range(entry_index, min(entry_index + max_bars, len(frame))):
        row = frame.iloc[j]
        bars_held = j - entry_index + 1
        high = float(row.high)
        low = float(row.low)
        close = float(row.close)
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

        # Volatility-aware management. Wider and slower in volatile periods.
        current_atr = float(row.atr) if np.isfinite(row.atr) else atr
        vol_factor = float(np.clip(current_atr / max(atr, 1e-12), 1.0, 2.8))
        if peak >= entry + 1.05 * atr:
            dynamic_stop = max(dynamic_stop, entry + 0.0002 * entry)
        if peak >= entry + 1.55 * atr:
            trailing_distance = 1.00 * atr * (vol_factor ** 0.85)
            dynamic_stop = max(dynamic_stop, peak - trailing_distance)

        minimum_soft_hold = int(round(4 * vol_factor))
        if bars_held >= minimum_soft_hold:
            recent_start = max(entry_index, j - 5)
            momentum = close / float(frame.iloc[recent_start].close) - 1.0
            threshold = -0.0012 * (vol_factor ** 0.8)
            if momentum < threshold and close < float(frame.iloc[j - 1].close):
                exit_price = close
                exit_reason = "momentum_decay"
                break
        exit_price = close

    gross = exit_price / entry - 1.0
    # Research assumption: 6 bps round trip for highly liquid US equities.
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
    }


def build_candidates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for symbol, raw in frames.items():
        frame = add_features(raw)
        mask = candidate_mask(frame)
        feature_columns = sorted(set(GEOMETRY_FEATURES))
        for index in np.flatnonzero(mask.to_numpy()):
            outcome = simulate_trade(frame, int(index))
            if not outcome["valid"]:
                continue
            record = {column: float(frame.iloc[index][column]) for column in feature_columns}
            record.update(outcome)
            record["symbol"] = symbol
            record["signal_time"] = frame.iloc[index].open_time
            record["session"] = int(frame.iloc[index].session)
            records.append(record)
    return pd.DataFrame(records).sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def fit_model(train: pd.DataFrame, features: list[str], model_kind: str) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("continuous", StandardScaler(), features),
            ("symbol", OneHotEncoder(handle_unknown="ignore"), ["symbol"]),
        ],
        remainder="drop",
    )
    estimator = (
        Ridge(alpha=18.0)
        if model_kind == "ridge"
        else HistGradientBoostingRegressor(
            learning_rate=0.055,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=28,
            l2_regularization=1.5,
            random_state=19,
        )
    )
    model = Pipeline([("transform", transformer), ("estimator", estimator)])
    model.fit(train[features + ["symbol"]], train["net_return"])
    return model


def portfolio_from_selected(selected: pd.DataFrame, starting: float = 1.0) -> tuple[float, float]:
    equity = starting
    peak = starting
    maximum_drawdown = 0.0
    for value in selected["net_return"].to_numpy(float):
        equity *= 1.0 + value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
    return equity - starting, maximum_drawdown


def evaluate(candidates: pd.DataFrame, features: list[str], label: str, model_kind: str) -> tuple[dict, pd.DataFrame]:
    unique_times = np.sort(candidates["signal_time"].unique())
    train_end = unique_times[int(len(unique_times) * 0.60)]
    calibration_end = unique_times[int(len(unique_times) * 0.80)]
    train = candidates[candidates.signal_time < train_end].copy()
    calibration = candidates[(candidates.signal_time >= train_end) & (candidates.signal_time < calibration_end)].copy()
    test = candidates[candidates.signal_time >= calibration_end].copy()

    model = fit_model(train, features, model_kind)
    calibration_prediction = model.predict(calibration[features + ["symbol"]])
    slope, intercept = np.polyfit(calibration_prediction, calibration.net_return.to_numpy(float), 1)
    raw_prediction = model.predict(test[features + ["symbol"]])
    test["predicted_net_return"] = intercept + slope * raw_prediction

    # Cross-sectional selection: at most one position per timestamp, and only
    # when expected net return clears a conservative 5-bps hurdle.
    ranked = test.sort_values(["signal_time", "predicted_net_return"], ascending=[True, False])
    ranked = ranked.groupby("signal_time", as_index=False).head(1)
    ranked = ranked[ranked.predicted_net_return > 0.0005].copy()
    # Enforce the research portfolio rule: 100% allocation and at most one
    # open position across all three symbols.
    accepted_rows = []
    next_free_time = pd.Timestamp.min.tz_localize("UTC")
    for _, row in ranked.sort_values("signal_time").iterrows():
        signal_time = pd.Timestamp(row.signal_time)
        if signal_time < next_free_time:
            continue
        accepted_rows.append(row)
        next_free_time = pd.Timestamp(row.exit_time)
    selected = pd.DataFrame(accepted_rows, columns=ranked.columns)
    total_return, maximum_drawdown = portfolio_from_selected(selected)

    metrics = {
        "feature_set": label,
        "model_kind": model_kind,
        "train_candidates": len(train),
        "calibration_candidates": len(calibration),
        "test_candidates": len(test),
        "selected_trades": len(selected),
        "selection_rate": len(selected) / max(test.signal_time.nunique(), 1),
        "test_mae_bps": mean_absolute_error(test.net_return, test.predicted_net_return) * 10_000,
        "test_r2": r2_score(test.net_return, test.predicted_net_return),
        "all_test_expectancy_bps": test.net_return.mean() * 10_000,
        "selected_expectancy_bps": selected.net_return.mean() * 10_000 if len(selected) else None,
        "selected_win_rate": float((selected.net_return > 0).mean()) if len(selected) else None,
        "portfolio_return": total_return,
        "max_drawdown": maximum_drawdown,
        "median_hold_bars": float(selected.bars_held.median()) if len(selected) else None,
        "symbol_counts": selected.symbol.value_counts().to_dict(),
    }
    return metrics, test


def buy_hold_benchmarks(frames: dict[str, pd.DataFrame], test_start: pd.Timestamp) -> list[dict]:
    rows = []
    returns = []
    equity_series = []
    for symbol, frame in frames.items():
        subset = frame[frame.open_time >= test_start].copy()
        gross = float(subset.iloc[-1].close / subset.iloc[0].open - 1.0)
        net = gross - 0.0006
        normalized = subset.close.astype(float) / float(subset.iloc[0].open)
        drawdown = float(((normalized.cummax() - normalized) / normalized.cummax()).max())
        rows.append({"benchmark": f"{symbol}_buy_hold", "return": net, "max_drawdown": drawdown})
        returns.append(net)
        equity_series.append(normalized.reset_index(drop=True))
    basket = pd.concat(equity_series, axis=1).mean(axis=1)
    basket_return = float(basket.iloc[-1] - 1.0 - 0.0006)
    basket_drawdown = float(((basket.cummax() - basket) / basket.cummax()).max())
    rows.append({"benchmark": "equal_weight_buy_hold", "return": basket_return, "max_drawdown": basket_drawdown})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=1300)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", default="outputs/equity_three_symbol_synthetic")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    frames = generate_market(args.sessions, args.seed)
    candidates = build_candidates(frames)
    baseline, baseline_predictions = evaluate(candidates, BASE_FEATURES, "baseline", "ridge")
    geometry_ridge, geometry_ridge_predictions = evaluate(candidates, GEOMETRY_FEATURES, "continuous_geometry", "ridge")
    geometry_tree, geometry_tree_predictions = evaluate(candidates, GEOMETRY_FEATURES, "continuous_geometry", "tree")
    test_start = min(
        baseline_predictions.signal_time.min(),
        geometry_ridge_predictions.signal_time.min(),
        geometry_tree_predictions.signal_time.min(),
    )
    benchmarks = buy_hold_benchmarks(frames, pd.Timestamp(test_start))

    summary = pd.DataFrame([baseline, geometry_ridge, geometry_tree])
    benchmark_frame = pd.DataFrame(benchmarks)
    summary.to_csv(output / "model_summary.csv", index=False)
    benchmark_frame.to_csv(output / "benchmarks.csv", index=False)
    candidates.to_csv(output / "candidates.csv", index=False)
    baseline_predictions.to_csv(output / "baseline_test_predictions.csv", index=False)
    geometry_ridge_predictions.to_csv(output / "geometry_ridge_test_predictions.csv", index=False)
    geometry_tree_predictions.to_csv(output / "geometry_tree_test_predictions.csv", index=False)
    for symbol, frame in frames.items():
        frame.iloc[:3000].to_csv(output / f"{symbol.lower()}_candles_sample.csv", index=False)

    result = {"models": [baseline, geometry_ridge, geometry_tree], "benchmarks": benchmarks}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(summary.to_string(index=False))
    print("\nBenchmarks")
    print(benchmark_frame.to_string(index=False))


if __name__ == "__main__":
    main()
