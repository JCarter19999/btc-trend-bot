"""Regime -> structure routing (spec section 4) with an expectancy comparison
across candidate structures (spec: "evaluate multiple candidate structures
where possible and choose the structure with the highest estimated net
expectancy").

CPU-only mode has no calibrated probability model, so the expectancy proxy
here is a documented heuristic, not a forecast: it uses the regime
classifier's confidence score as a stand-in win probability, and for
uncapped structures (straddle/strangle) assumes the profit potential is
`assumed_uncapped_multiple` x max_loss. This is only used to break ties
between candidate structures within one regime (e.g. straddle vs. strangle
under VOLATILITY_EXPANSION) -- it is not a claim about real trade
expectancy, which is what the backtest (using real fills) actually measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .structures import OptionStructure, build_structure

ASSUMED_UNCAPPED_MULTIPLE = 3.0


@dataclass
class CandidateEval:
    kind: str
    structure: OptionStructure | None
    expectancy_proxy: float | None
    rejected_reason: str | None = None


@dataclass
class RoutingDecision:
    regime: str
    confidence: float
    candidates: list[CandidateEval] = field(default_factory=list)
    selected_kind: str | None = None
    selected_structure: OptionStructure | None = None
    trade: bool = False
    reason: str = ""


def _max_leg_spread_pct(structure: OptionStructure, chains: dict) -> float:
    worst = 0.0
    for leg in structure.legs:
        chain = chains.get(leg.cp)
        if chain is None or chain.empty:
            continue
        match = chain[chain["strike"] == leg.strike]
        if match.empty:
            continue
        bid = float(match.iloc[0]["bid_px_00"])
        ask = float(match.iloc[0]["ask_px_00"])
        mid = (bid + ask) / 2.0
        if mid > 0:
            worst = max(worst, (ask - bid) / mid)
    return worst


def _expectancy_proxy(structure: OptionStructure, confidence: float, cfg: dict) -> float:
    max_profit = structure.max_profit
    if max_profit is None:
        max_profit = structure.max_loss * cfg.get("assumed_uncapped_multiple", ASSUMED_UNCAPPED_MULTIPLE)
    return confidence * max_profit - (1.0 - confidence) * structure.max_loss


def route(regime: str, confidence: float, entry_chains: dict, spot: float, cfg: dict,
          fill_mode: str = "natural") -> RoutingDecision:
    decision = RoutingDecision(regime=regime, confidence=confidence)

    if regime in ("CHOP_RANGE", "NO_TRADE"):
        decision.reason = f"regime {regime} routes to no strategies by default (spec section 4)"
        return decision

    kinds = cfg["strategy_map"].get(regime, [])
    if not kinds:
        decision.reason = f"no strategies configured for regime {regime}"
        return decision

    nt = cfg["no_trade_filters"]
    for kind in kinds:
        structure = build_structure(kind, entry_chains, spot, cfg, fill_mode)
        if structure is None:
            decision.candidates.append(CandidateEval(kind, None, None, "could not price structure (missing strikes/quotes)"))
            continue
        spread_pct = _max_leg_spread_pct(structure, entry_chains)
        if spread_pct > nt["max_entry_spread_pct_of_mid"]:
            decision.candidates.append(CandidateEval(kind, structure, None, f"leg spread {spread_pct:.1%} exceeds max {nt['max_entry_spread_pct_of_mid']:.1%}"))
            continue
        expectancy = _expectancy_proxy(structure, confidence, cfg)
        if expectancy < nt["min_net_expectancy"]:
            decision.candidates.append(CandidateEval(kind, structure, expectancy, f"expectancy {expectancy:.2f} below min {nt['min_net_expectancy']}"))
            continue
        decision.candidates.append(CandidateEval(kind, structure, expectancy, None))

    viable = [c for c in decision.candidates if c.rejected_reason is None]
    if not viable:
        decision.reason = "no candidate structure cleared the no-trade filters"
        return decision

    best = max(viable, key=lambda c: c.expectancy_proxy)
    decision.selected_kind = best.kind
    decision.selected_structure = best.structure
    decision.trade = True
    decision.reason = f"selected {best.kind} (expectancy proxy {best.expectancy_proxy:.2f}) among {len(viable)} viable candidate(s)"
    return decision
