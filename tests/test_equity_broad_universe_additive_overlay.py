import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'experiments' / 'run_equity_broad_universe_additive_overlay.py'
spec = importlib.util.spec_from_file_location('broad_overlay', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_additive_overlay_keeps_full_stock_exposure():
    trades = pd.DataFrame({
        'option_overlay_eligible': [True], 'bull_regime_score': [5],
        'predicted_net_return': [0.01], 'option_utility': [0.03],
        'net_return': [0.02], 'realized_option_return': [0.50],
    })
    result, fraction = mod.additive_returns(trades, 'fixed_20')
    assert np.isclose(fraction[0], 0.20)
    assert np.isclose(result[0], 0.02 + 0.20 * 0.50)


def test_no_eligible_option_equals_stock_only():
    trades = pd.DataFrame({
        'option_overlay_eligible': [False], 'bull_regime_score': [5],
        'predicted_net_return': [0.01], 'option_utility': [0.03],
        'net_return': [-0.01], 'realized_option_return': [0.80],
    })
    result, fraction = mod.additive_returns(trades, 'fixed_30')
    assert fraction[0] == 0.0
    assert result[0] == -0.01
