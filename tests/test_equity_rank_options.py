import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('rank_options', ROOT / 'experiments' / 'run_equity_ranking_options_experiment.py')
MOD = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_black_scholes_call_is_positive_and_monotone():
    low = MOD.bs_call(100.0, 105.0, 30 / 252, 0.30)
    high = MOD.bs_call(110.0, 105.0, 30 / 252, 0.30)
    assert low > 0
    assert high > low


def test_long_option_loss_is_capped():
    import pandas as pd
    row = pd.Series({
        'entry_price': 100.0, 'exit_price': 50.0, 'bars_held': 13,
        'realized_vol_13': 0.01, 'symbol': 'AAPL', 'shock_volume_flag': 0.0,
        'mae': -0.5, 'mfe': 0.0,
    })
    result = MOD.option_return(row, MOD.OPTION_POLICIES[0])
    assert -1.0 <= result['option_return']
