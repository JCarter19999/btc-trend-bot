"""CPU-only intraday options strategy router (Phase 1).

Classifies the 9:30-10:30 ET SPX regime from free data, routes to a
defined-risk options structure, and evaluates a binary exit policy
(hold-to-close vs. exit-at-noon) against real SPXW 0DTE quotes already
cached in data/opra_spx/ and data/opra_spx_exit/.

See equity_v2_4_research/OPTIONS_ROUTER_PHASE1.md for what is and isn't in
scope for this phase.
"""
