"""Send a plain-text email via Gmail SMTP using an app password.

Exists because the account's Gmail MCP connector (used interactively in
Claude Code) can only create drafts, not send -- no good for an unattended
cron job with nobody there to click Send. This is the deterministic,
always-actually-delivers path the daily checkup relies on.

Reads GMAIL_SMTP_USER / GMAIL_SMTP_APP_PASSWORD / GMAIL_SMTP_TO from the
environment (loaded from /home/joey/.config/btc-trend-bot/gmail_smtp.env by
the systemd service's EnvironmentFile -- never committed to git).
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send(subject: str, body: str, to: str | None = None, html_body: str | None = None) -> None:
    user = os.environ["GMAIL_SMTP_USER"]
    app_password = os.environ["GMAIL_SMTP_APP_PASSWORD"]
    recipient = to or os.environ["GMAIL_SMTP_TO"]
    cc = [addr.strip() for addr in os.environ.get("GMAIL_SMTP_CC", "").split(",") if addr.strip()]

    if html_body:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
    else:
        msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = recipient
    if cc:
        msg["Cc"] = ", ".join(cc)

    # Port 465 (implicit TLS) is blocked outbound on this VM -- 587
    # (STARTTLS) is open, confirmed via a direct /dev/tcp connectivity check.
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(user, app_password)
        server.sendmail(user, [recipient, *cc], msg.as_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a plain-text email via Gmail SMTP")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", help="Path to read the plain-text body from (default: stdin)")
    parser.add_argument("--html-body-file", help="Optional path to an HTML alternative body")
    parser.add_argument("--to", help="Override recipient (default: GMAIL_SMTP_TO)")
    args = parser.parse_args()

    body = open(args.body_file).read() if args.body_file else sys.stdin.read()
    html_body = open(args.html_body_file).read() if args.html_body_file else None
    send(args.subject, body, args.to, html_body)
    print(f"Sent: {args.subject!r} to {args.to or os.environ['GMAIL_SMTP_TO']}")


if __name__ == "__main__":
    main()
