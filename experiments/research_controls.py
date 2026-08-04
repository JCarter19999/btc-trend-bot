"""The controls that actually caught things. Import these; don't re-derive them.

Three major promotion decisions have been made in this project. Two were
reversed, both by controls that were cheap and were not run at promotion time:

- **Ridge** (halted 2026-07-23): passed a backtest, then landed in the 2nd-10th
  percentile of a random-selection distribution -- worse than its own
  shuffled-label control.
- **DAX / ES / NQ lead-lag options** (void 2026-07-26): "+469.6%, the strongest
  single options result in this project's history" was a 30-minute lookahead.
  A window-overlap assertion would have caught it before it was written up,
  let alone deployed.
- **simple_trend** (standing): reproduces its 96th-percentile random control,
  but loses to volatility-matched buy-and-hold of the four stocks it trades --
  a benchmark nobody ran because the gate specified SPY.

Every one of those is a control failure, not a modelling failure. This module
collects the ones that have earned their place, so the next candidate gets
tested against them by default rather than by remembering to.

See PROMOTION_CONTROLS.md for which are mandatory before a shadow deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_SHUFFLE = 1000
SEED = 0
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Window-overlap assertion -- would have caught the lead-lag lookahead
# ---------------------------------------------------------------------------

def assert_no_window_overlap(signal_end: pd.Series, target_start: pd.Series,
                             label: str = "") -> None:
    """The signal must be fully knowable before the window it predicts begins.

    Pass the END of each signal window and the START of each target window,
    indexed alike. Bars are labelled by their START by most vendors
    (yfinance included), so a 60m bar tagged 13:00 runs to 14:00 -- the defect
    that produced a +0.379 correlation which was really -0.008.
    """
    joined = pd.concat([signal_end.rename("end"), target_start.rename("start")],
                       axis=1).dropna()
    bad = joined[joined["end"] > joined["start"]]
    if len(bad):
        raise AssertionError(
            f"{label}: signal window ends after target window starts on "
            f"{len(bad)} of {len(joined)} dates, e.g.\n{bad.head()}"
        )


# ---------------------------------------------------------------------------
# 2. Shuffled null -- is the correlation more than reshuffled pairings give?
# ---------------------------------------------------------------------------

def shuffled_null_percentile(x: pd.Series, y: pd.Series,
                             n_shuffle: int = N_SHUFFLE, seed: int = SEED) -> dict:
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(j) < 30:
        return {"n": int(len(j)), "note": "too thin"}
    corr = float(np.corrcoef(j["x"], j["y"])[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(j["x"].to_numpy(), rng.permutation(j["y"].to_numpy()))[0, 1]
                     for _ in range(n_shuffle)])
    return {
        "n": int(len(j)),
        "corr": corr,
        "percentile": float((null < corr).mean() * 100),
        "passes_95th": bool((null < corr).mean() * 100 >= 95.0),
    }


# ---------------------------------------------------------------------------
# 3. Negative control -- a case where the effect CANNOT exist
# ---------------------------------------------------------------------------

def negative_control(effect_fn, cases: dict) -> dict:
    """Run the same measurement where the mechanism is impossible.

    The lead-lag audit's decisive evidence was not the effect shrinking, it
    was Nikkei and Hang Seng being unchanged to four decimal places -- those
    markets are shut at 13:00 UTC, so nothing could leak. An effect that also
    shows up where it cannot exist is measuring the harness, not the market.

    `cases` maps label -> input; each is passed to `effect_fn`. Every case
    here should come back null. Any that doesn't is an indictment.
    """
    return {label: effect_fn(payload) for label, payload in cases.items()}


# ---------------------------------------------------------------------------
# 4. Mechanical control -- a relationship that MUST hold if the join is right
# ---------------------------------------------------------------------------

def mechanical_control(x: pd.Series, y: pd.Series, label: str = "") -> dict:
    """A definitional relationship, used to prove the plumbing before reading
    the result.

    The auction studies would have been thrown away without this: the opening
    imbalance's post-open numbers were flat, and the only reason that could be
    trusted as a real null rather than a broken join was the cross-vs-book
    identity coming in at 84.8% direction accuracy on the same records.

    A mechanical control that FAILS means stop and fix the data. Note it may
    also reveal a sign convention -- Databento's imbalance `side` reads
    opposite to its enum, which this control is what established.
    """
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(j) < 30:
        return {"label": label, "n": int(len(j)), "note": "too thin"}
    return {
        "label": label,
        "n": int(len(j)),
        "corr": float(np.corrcoef(j["x"], j["y"])[0, 1]),
        "direction_accuracy": float((np.sign(j["x"]) == np.sign(j["y"])).mean()),
    }


# ---------------------------------------------------------------------------
# 5. In-sample vs out-of-sample gate -- is the threshold doing the work?
# ---------------------------------------------------------------------------

def expanding_gate(values: pd.Series, quantile: float = 0.75,
                   min_periods: int = 60) -> pd.Series:
    """Threshold from prior observations only. Compare results under this to
    the same gate fitted in-sample; if the OOS version is weaker, the gate was
    the edge. (The closing-cross fade came back *stronger* under this --
    5.86 vs 5.04 bp -- which is why it was written up.)
    """
    return values.shift(1).expanding(min_periods=min_periods).quantile(quantile)


# ---------------------------------------------------------------------------
# 6. Day-clustered significance -- N symbols on one day is not N observations
# ---------------------------------------------------------------------------

def day_clustered_t(returns: pd.Series, dates: pd.Series) -> dict:
    """Average within session first, then t-test the daily series.

    A plain t-stat over symbol-days treats eight correlated names as eight
    independent draws and overstates significance, sometimes by a lot.
    """
    daily = pd.Series(returns.to_numpy(), index=pd.Index(dates)).groupby(level=0).mean()
    if len(daily) < 30 or daily.std(ddof=1) == 0:
        return {"days": int(len(daily)), "note": "too thin"}
    return {
        "days": int(len(daily)),
        "mean": float(daily.mean()),
        "t_stat": float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))),
    }


# ---------------------------------------------------------------------------
# 7. The right benchmark -- vol-matched, on the instruments actually traded
# ---------------------------------------------------------------------------

def vol_matched_benchmark(strategy_equity: pd.Series,
                          benchmark_returns: pd.Series) -> dict:
    """Scale the benchmark to the strategy's realized volatility, then compare.

    Beating SPY while trading four megacap growth names is not evidence.
    Benchmark against what is actually held, at matched risk -- simple_trend
    beats SPY comfortably and loses to this.
    """
    s_rets = strategy_equity.pct_change().dropna()
    b = benchmark_returns.dropna()
    if len(s_rets) < 30 or b.std(ddof=1) == 0:
        return {"note": "too thin"}
    scale = float(s_rets.std(ddof=1) / b.std(ddof=1))
    matched = (1 + b * scale).cumprod()
    s_total = float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1)
    b_total = float(matched.iloc[-1] / matched.iloc[0] - 1)
    return {
        "scale_applied": scale,
        "strategy_total_return": s_total,
        "benchmark_vol_matched_total_return": b_total,
        "strategy_wins": bool(s_total > b_total),
    }


# ---------------------------------------------------------------------------
# 8. Single-year dependence -- is the whole case one period?
# ---------------------------------------------------------------------------

def drop_one_year(per_year: dict[str, float], benchmark_per_year: dict[str, float]) -> dict:
    """Recompound with each year removed in turn.

    simple_trend beats its basket in 2 of 9 years; drop 2022 and it goes from
    roughly level to hopelessly behind. A record that survives only with one
    period included is a claim about that period.
    """
    out = {}
    for drop in per_year:
        keep = [y for y in per_year if y != drop]
        s = float(np.prod([1 + per_year[y] for y in keep]) - 1)
        b = float(np.prod([1 + benchmark_per_year[y] for y in keep]) - 1)
        out[f"ex_{drop}"] = {"strategy": s, "benchmark": b, "diff": s - b}
    return out
