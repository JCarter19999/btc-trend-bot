"""Deterministic daily checkup of the three live paper-trading deployments --
runs at 7am via equity-daily-checkup.timer, emails a summary.

Originally designed to run as a headless `claude -p` agentic session (see
git history), but that requires the /home/joey/equity_v2_4 workspace to have
passed Claude Code's interactive trust dialog, and spawning a second
full-permission Claude Code instance from within an already-running session
is (correctly) refused by the sandbox classifier -- not something to route
around. A plain script covers the same known failure patterns without
either problem, and is more predictable for an unattended job anyway.

The one qualitative piece -- "Claude's Hot Take" -- is rule-based commentary
grounded in this project's actual validation history (see CLAUDE.md and the
research-branch docs), not a live model call.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_equity_paper_step as stock_bot  # noqa: E402
import run_equity_call_paper_step as call_bot  # noqa: E402
import run_european_signal_shadow_step as eu_signal_bot  # noqa: E402
from send_email_smtp import send  # noqa: E402

BOTS = [
    ("simple_trend", "primary stock", ROOT / "config/settings_equity_paper_yfinance_simpletrend.yaml"),
    ("Ridge", "control arm", ROOT / "config/settings_equity_paper_yfinance.yaml"),
]
TIMERS = [
    "equity-paper-yfinance.timer", "equity-paper-yfinance-simpletrend.timer",
    "equity-paper-calls.timer", "equity-quote-refresh-simpletrend.timer",
    "equity-quote-refresh-ridge.timer", "european-signal-shadow-entry.timer",
    "european-signal-shadow-exit.timer",
]
SERVICES_TO_CHECK_LOGS = [
    "equity-paper-yfinance.service", "equity-paper-yfinance-simpletrend.service", "equity-paper-calls.service",
    "european-signal-shadow-entry.service", "european-signal-shadow-exit.service",
]


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout


def check_timers() -> tuple[str, list[str]]:
    out = _run(["systemctl", "list-timers", *TIMERS, "--all"])
    active = _run(["systemctl", "is-active", *TIMERS, "btc-dashboard.service"]).splitlines()
    issues = [name for name, state in zip([*TIMERS, "btc-dashboard.service"], active) if state != "active"]
    return out, issues


def check_recent_errors() -> list[str]:
    """Crude substring match on 'error'/'traceback'/'exception' -- excludes
    lines that are actually Python warnings (DeprecationWarning etc.),
    which routinely say things like "will raise an error in the future"
    as normal warning text, not a real error. Caught a false positive from
    exactly this pattern on 2026-07-23 (a DeprecationWarning line got
    reported as an "issue" in the daily email)."""
    findings = []
    for svc in SERVICES_TO_CHECK_LOGS:
        out = _run(["journalctl", "-u", svc, "--since", "-24 hours", "--no-pager"])
        for line in out.splitlines():
            lowered = line.lower()
            if "warning" in lowered:
                continue
            if any(tok in lowered for tok in ("error", "traceback", "exception")):
                findings.append(f"{svc}: {line.strip()}")
    return findings


def check_dashboard() -> bool:
    try:
        out = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8501"],
                              capture_output=True, text=True, timeout=10).stdout
        return out.strip() == "200"
    except Exception:
        return False


def fix_stuck_pending_entry(config_path: Path, status: dict) -> str | None:
    pos = status.get("open_position")
    if not pos or pos.get("status") != "pending_entry":
        return None
    last_processed = status.get("last_processed_bar_date")
    if not last_processed:
        return None
    age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(last_processed).replace(tzinfo=timezone.utc)).days
    if age_days < 1:
        return None
    stock_bot.run_step(config_path)
    new_status = stock_bot.get_status(config_path)
    new_pos = new_status.get("open_position") or {}
    if new_pos.get("status") == "open":
        return f"Advanced stuck pending_entry for {pos.get('symbol')} -> filled at ${new_pos.get('entry_price'):.2f}"
    return f"Tried to advance stuck pending_entry for {pos.get('symbol')}, still not filled -- needs a look"


def hot_take(stock_statuses: dict, call_status: dict) -> str:
    """Rule-based commentary, not a live model call -- see module docstring."""
    lines = []
    for name, label, status in stock_statuses:
        pos = status.get("open_position")
        completed = status.get("completed_trades", 0)
        if completed == 0 and pos and pos.get("current_price") and pos.get("entry_price"):
            unrl = pos["current_price"] / pos["entry_price"] - 1
            if name == "simple_trend":
                lines.append(
                    f"<b>simple_trend</b> is {completed} trades in -- {unrl*100:+.2f}% on the open {pos.get('symbol')} "
                    "position is noise, not signal, at this stage. What actually matters here is the backtest it's "
                    "carrying into live trading: 243.3 bps/trade, 99th percentile against 100 random-selection seeds, "
                    "stable across 2018-2026 including 2022. This is the one bot with real validated backing -- "
                    "the honest expectation is boring, grindy, medium-horizon drift capture, not big daily swings. "
                    "I'd want to see dozens of completed trades before reading anything into the live number either way."
                )
            elif name == "Ridge":
                lines.append(
                    f"<b>Ridge</b> is running as a deliberate control arm, not a live candidate -- it failed "
                    "re-validation (2nd-10th percentile of the same random-selection distribution simple_trend beat), "
                    "so I'm watching it expecting it to underwhelm relative to simple_trend over time. If it "
                    "consistently keeps pace or beats simple_trend live, that would actually be the interesting "
                    "result -- it'd mean the backtest correction missed something. Right now, day one, there's nothing to read."
                )
        elif completed == 0:
            lines.append(f"<b>{name}</b> ({label}): no fills to comment on yet.")

    call_pos = call_status.get("open_position") if call_status else None
    if call_pos and call_status.get("completed_trades", 0) == 0:
        entry, mark = call_pos.get("entry_premium"), call_pos.get("current_mark")
        if entry and mark:
            unrl = mark / entry - 1
            lines.append(
                f"<b>Call options</b> on {call_pos.get('underlying_symbol')}: premium down {unrl*100:.1f}% so far, "
                "which for a single-digit-days-old 30-DTE option is well within normal decay/noise -- not a signal. "
                "This is genuinely the one I'm most curious about: the synthetic backtest found calls roughly tied "
                "stock on return but was very sensitive to an assumed spread it couldn't pin down. Live real quotes "
                "are the actual answer to that question, and we don't have enough trades yet to know it."
            )

    if not lines:
        lines.append("Nothing live enough yet to have a take on. Check back once a few trades have completed.")
    return "<br><br>".join(lines)


def main() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    fixes, issues = [], []

    stock_statuses = []
    for name, label, cfg in BOTS:
        status = stock_bot.get_status(cfg)
        fix = fix_stuck_pending_entry(cfg, status)
        if fix:
            fixes.append(f"{name}: {fix}")
            status = stock_bot.get_status(cfg)  # re-read after fix
        if status.get("halted"):
            issues.append(f"{name} is HALTED: {status.get('halt_reason')}")
        stock_statuses.append((name, label, status))

    call_status = call_bot.get_status(ROOT / "config/settings_equity_paper_calls.yaml")
    if call_status.get("halted"):
        issues.append(f"Call options is HALTED: {call_status.get('halt_reason')}")

    eu_status = eu_signal_bot.status()
    if eu_status.get("halted"):
        issues.append(f"European signal is HALTED: {eu_status.get('halt_reason')}")

    timers_out, inactive_timers = check_timers()
    if inactive_timers:
        issues.append(f"Inactive: {', '.join(inactive_timers)}")
    errors = check_recent_errors()
    if errors:
        issues.extend(errors[:5])
    if not check_dashboard():
        issues.append("Dashboard (btc-dashboard.service, :8501) not responding with 200")

    verdict = "ISSUES FOUND" if issues else "ALL CLEAR"
    subject = f"Equity/BTC paper-trading daily check -- {today} -- {verdict}"

    def bot_row(name, label, status):
        pos = status.get("open_position") or {}
        symbol = pos.get("symbol", "flat")
        entry = pos.get("entry_price")
        cur = pos.get("current_price")
        pnl = f"{(cur/entry-1)*100:+.2f}%" if entry and cur else "—"
        return name, label, symbol, f"${entry:,.2f}" if entry else "—", f"${cur:,.2f}" if cur else "—", pnl, status.get("completed_trades", 0)

    rows = [bot_row(n, l, s) for n, l, s in stock_statuses]
    call_pos = call_status.get("open_position") or {}
    call_pnl = "—"
    if call_pos.get("entry_premium") and call_pos.get("current_mark"):
        call_pnl = f"{(call_pos['current_mark']/call_pos['entry_premium']-1)*100:+.2f}%"
    rows.append(("Call options", "primary options", call_pos.get("underlying_symbol", "flat"),
                 f"${call_pos.get('entry_premium'):.2f}" if call_pos.get("entry_premium") else "—",
                 f"${call_pos.get('current_mark'):.2f}" if call_pos.get("current_mark") else "—",
                 call_pnl, call_status.get("completed_trades", 0)))

    eu_primary = eu_status.get("primary_book", {})
    eu_signal = eu_status.get("most_recent_signal") or {}
    eu_direction = {1: "LONG", -1: "SHORT", 0: "FLAT", None: "—"}.get(eu_signal.get("dax_direction"), "—")

    text_lines = [f"Equity/BTC paper-trading daily check -- {today} -- {verdict}", ""]
    for n, l, sym, entry, cur, pnl, completed in rows:
        text_lines.append(f"{n} ({l}): {sym} | entry {entry} | current {cur} | unrealized {pnl} | completed trades {completed}")
    text_lines.append(
        f"European signal (SPY, DAX-top-quartile): equity ${eu_primary.get('current_equity', 2500.0):,.2f} "
        f"({eu_primary.get('total_return_pct', 0.0):+.2f}%) | trades {eu_primary.get('trades_taken', 0)} | "
        f"today's signal {eu_direction}"
    )
    if fixes:
        text_lines += ["", "Fixed automatically:"] + [f"- {f}" for f in fixes]
    if issues:
        text_lines += ["", "Issues found:"] + [f"- {i}" for i in issues]
    text_lines += ["", "Claude's Hot Take:", hot_take(stock_statuses, call_status).replace("<br><br>", "\n\n").replace("<b>", "").replace("</b>", "")]
    text_body = "\n".join(text_lines)

    status_color = "#d62728" if issues else "#2ca02c"
    table_rows = "".join(
        f"<tr><td style='padding:8px 12px;border-bottom:1px solid #333'>{n}<br><span style='color:#888;font-size:12px'>{l}</span></td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #333'>{sym}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #333'>{entry}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #333'>{cur}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #333;color:{'#2ca02c' if pnl.startswith('+') else '#d62728' if pnl.startswith('-') else '#888'}'>{pnl}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #333'>{completed}</td></tr>"
        for n, l, sym, entry, cur, pnl, completed in rows
    )
    fixes_html = ("<h3 style='color:#2ca02c'>Fixed automatically</h3><ul>" + "".join(f"<li>{f}</li>" for f in fixes) + "</ul>") if fixes else ""
    issues_html = ("<h3 style='color:#d62728'>Issues found</h3><ul>" + "".join(f"<li>{i}</li>" for i in issues) + "</ul>") if issues else ""
    eu_return_color = "#2ca02c" if eu_primary.get("total_return_pct", 0.0) >= 0 else "#d62728"
    eu_html = f"""
  <h3 style="margin-top:24px;color:#a855f7">🌍 European signal (SPY, DAX-top-quartile)</h3>
  <p style="line-height:1.6;color:#c9d1d9">
    Equity <b>${eu_primary.get('current_equity', 2500.0):,.2f}</b>
    (<span style="color:{eu_return_color}">{eu_primary.get('total_return_pct', 0.0):+.2f}%</span>)
    &middot; {eu_primary.get('trades_taken', 0)} trades &middot; today's signal: <b>{eu_direction}</b>
  </p>
"""
    html_body = f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:0 auto;background:#0d1117;color:#e6edf3;padding:24px;border-radius:12px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    <span style="background:{status_color};color:#0d1117;font-weight:700;font-size:12px;padding:4px 10px;border-radius:999px">{verdict}</span>
    <span style="color:#888;font-size:13px">{today}</span>
  </div>
  <h2 style="margin:12px 0 16px 0">Equity / BTC paper-trading daily check</h2>
  <table style="width:100%;border-collapse:collapse;font-size:14px">
    <tr style="color:#888;text-align:left;font-size:12px;text-transform:uppercase">
      <th style="padding:0 12px 8px 12px">Bot</th><th style="padding:0 12px 8px 12px">Symbol</th>
      <th style="padding:0 12px 8px 12px">Entry</th><th style="padding:0 12px 8px 12px">Current</th>
      <th style="padding:0 12px 8px 12px">Unrealized</th><th style="padding:0 12px 8px 12px">Trades</th>
    </tr>
    {table_rows}
  </table>
  {eu_html}
  {fixes_html}
  {issues_html}
  <h3 style="margin-top:24px">🔥 Claude's Hot Take</h3>
  <p style="line-height:1.6;color:#c9d1d9">{hot_take(stock_statuses, call_status)}</p>
  <p style="color:#555;font-size:11px;margin-top:24px">Automated daily check -- equity_v2_4 / btc-dashboard, sent {datetime.now(timezone.utc).isoformat()}</p>
</div>
"""

    send(subject, text_body, html_body=html_body)
    print(f"Sent: {subject}")
    if issues:
        print("ISSUES:", issues)


if __name__ == "__main__":
    main()
