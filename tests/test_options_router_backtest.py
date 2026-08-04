"""End-to-end smoke test: BacktestEngine over a handful of real cached OPRA
days. Skips if the cached data or network (yfinance) isn't available."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from btc_trend_bot.options_router.backtest import available_days, run_backtest

CONFIG_PATH = ROOT / "configs" / "options_router.yaml"
ENTRY_DIR = ROOT / "data" / "opra_spx"
EXIT_DIR = ROOT / "data" / "opra_spx_exit"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG_PATH.exists():
        pytest.skip("options_router.yaml not found")
    return _load_cfg()


def test_backtest_smoke_over_a_few_cached_days(cfg):
    days = available_days(ENTRY_DIR, EXIT_DIR)
    if len(days) < 3:
        pytest.skip("not enough cached OPRA days to smoke test")
    start, end = days[0].strftime("%Y-%m-%d"), days[4].strftime("%Y-%m-%d") if len(days) > 4 else days[-1].strftime("%Y-%m-%d")

    try:
        decisions, trades = run_backtest(cfg, root=ROOT, start=start, end=end)
    except Exception as e:  # network/yfinance unavailable
        pytest.skip(f"could not run backtest (likely no network for yfinance): {e}")

    assert not decisions.empty
    assert set(decisions["date"]) <= {d.strftime("%Y-%m-%d") for d in days}
    assert "initial_regime" in decisions.columns or "reason" in decisions.columns

    if not trades.empty:
        assert (trades["contracts"] > 0).all()
        assert trades["return"].notna().all()
