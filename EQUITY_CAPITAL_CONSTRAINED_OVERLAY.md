# Capital-Constrained Stock + Call Overlay

This experiment applies whole-contract option affordability and Schwab online fee assumptions to the v2.7 selected trades.

## Execution assumptions

- Online listed stock commission: $0.
- Fractional stock purchases: permitted from $1.
- Options: $0 online base commission plus $0.65 per contract on entry and exit.
- Standard equity option multiplier: 100.
- Variable regulatory and exchange fees are not separately modeled.
- Call overlay target: 30% of current account equity.
- Monthly contribution: configurable; primary experiment uses $50.

## Routes

### Cash-safe

When an eligible whole call contract is affordable within the 30% budget, its exact opening cost is reserved and all remaining equity is placed in fractional stock. When no call is affordable, the stock receives the full account balance.

### Additive margin

The stock notional remains 100% of equity and the call is additional exposure. This reproduces the intended 100% stock + up to 30% premium route but requires margin or equivalent external buying power. Margin interest is not modeled, so this route is an upper-bound research comparison rather than a deployable cash-account result.
