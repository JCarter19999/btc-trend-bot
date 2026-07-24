# Taiwan → US semiconductors: Phase 3 candidate A, tested

Per Joey's "Phase 3: find a second independent edge" framing (2026-07-24)
— ranked ahead of bond/rates, FX, and overnight-futures-structure
candidates as the sharpest, most mechanistically distinct hypothesis
(Taiwan/TSMC dominates global leading-edge chip supply, a genuinely
different transmission channel than the DAX-based general risk-sentiment
signal). Reused the exact validated lead-lag methodology from
`EUROPEAN_LEAD_US_FIRST_HOUR_STUDY.txt` unchanged where it applies —
Taiwan's session (09:00-13:30 Taipei = 01:00-05:30 UTC) closes 8 hours
before US open, well before DAX's own session even opens, so no lookahead
concerns.

`experiments/run_taiwan_semiconductor_lead_lag_study.py`.

## What was tested

Signal: `^TWII` (Taiwan Weighted Index) own-session cumulative return.
`TSM` (Taiwan Semiconductor's US-listed ADR) was also attempted as an
alternative proxy but produced 0 usable observations — it trades on NYSE
during US hours, not Taipei hours, so it simply isn't present in the
pre-US-open window this method looks at. Dropped, not a bug.

Target: SOXX and SMH (semiconductor sector ETFs) first-hour US return.

## Results (448 days, 2023-08-24 to 2026-07-23)

| | SOXX | SMH |
|---|---|---|
| Directional correlation | -0.137 | -0.124 |
| Directional percentile vs. 1000-shuffle | **0.1** (i.e. more negative than 99.9% of shuffles) | 0.3 |
| Magnitude correlation | 0.194 | 0.177 |
| Magnitude percentile vs. shuffle | **100.0** | 100.0 |

**Both effects are real** (nowhere near the shuffled-control range), but
the shape isn't what the DAX→SPY signal has:

- **Direction is contrarian, not confirming, and weak in magnitude**
  (-0.12 to -0.14). This is the same pattern already found for broad
  Asian markets vs. SPY in the European-lead study (Nikkei/HSI/Shanghai
  → opposite-sign, not same-sign) — this result mostly *replicates* that
  known effect in a sharper, sector-specific form rather than being a
  wholly new mechanism. Compare to DAX→SPY's +0.28 same-sign correlation
  — meaningfully stronger and in the intuitive direction.
- **Magnitude is the stronger, cleaner signal** (100th percentile both
  targets): a big Taiwan move — either direction — reliably precedes a
  bigger-than-average US semiconductor first-hour move. This is the
  same "volatility clustering across markets" shape as the
  DAX-magnitude-gates-Asia finding already in
  `EUROPEAN_LEAD_ASIAN_MARKETS_AND_OPTIONS_OVERLAY.txt`.

## Independence check (the actual point of Phase 3)

TWII signal vs. the already-validated DAX signal, same 681 overlapping
days: **correlation = -0.080**. Genuinely low — this would be a real
diversifier if it's tradeable, not the same risk factor under a
different ticker.

## Honest verdict

Real, independent, but not yet a strategy. The magnitude effect is
strong and clean; the directional effect is real but weak and
contrarian, meaning a naive "Taiwan up → buy semis" rule would lose
money on direction alone. The DAX study's own best result used the
same shape of finding (Asia magnitude gating DAX direction, not Asia
direction predicting SPY direction) — the natural next step here is the
analogous move: use Taiwan's move MAGNITUDE as a filter/gate on some
other directional signal (DAX itself, or SOXX's own momentum) rather
than trading Taiwan's direction directly. Not yet built — this is a
lead worth pursuing, not a finding worth trading as-is.
