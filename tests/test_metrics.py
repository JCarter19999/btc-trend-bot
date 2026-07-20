import pandas as pd

from btc_trend_bot.metrics import moving_block_bootstrap_mean_ci


def test_block_bootstrap_is_deterministic():
    returns = pd.Series([0.01, -0.005, 0.002, 0.004] * 50)
    first = moving_block_bootstrap_mean_ci(returns, 200, 8, seed=42)
    second = moving_block_bootstrap_mean_ci(returns, 200, 8, seed=42)
    assert first == second
    assert first["probability_positive"] > 0.5
