import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / 'experiments' / 'run_equity_extreme_bull_overlay.py'
spec = importlib.util.spec_from_file_location('bull_overlay', P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_bull_regime_requires_four_conditions_and_option_gate():
    row = {
        'return_26': 0.07, 'return_13': 0.05, 'return_8': 0.04,
        'ema_slope_atr': 1.0, 'three_bar_close_progress': 1.0,
        'realized_option_return': 0.30, 'predicted_net_return': 0.006,
        'expected_option_return': 0.04, 'option_probability_profit': 0.60,
    }
    out = mod.add_bull_regime_fields(pd.DataFrame([row]))
    assert out.loc[0, 'bull_regime_score'] == 4
    assert bool(out.loc[0, 'bull_option_eligible'])


def test_tiered_overlay_uses_15_and_30_percent():
    scores = pd.Series([4, 5, 5])
    eligible = pd.Series([True, True, False])
    got = mod.overlay_fraction(scores, eligible, 'tiered')
    np.testing.assert_allclose(got, [0.15, 0.30, 0.0])
