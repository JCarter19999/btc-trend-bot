# Apply five-minute paper lab v0.2

From the five-minute worktree:

```bash
cd /home/joey/btc-paper-5m
unzip -o /home/joey/btc-paper-5m-v0.2.0-overlay.zip

docker compose -f compose.paper-5m.yaml build --no-cache

docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m pytest -q
```

Then rerun research into a new output directory:

```bash
docker compose -f compose.paper-5m.yaml run --rm \
  --entrypoint python \
  btc-paper-5m \
  -m btc_trend_bot.paper_lab \
  --config config/settings_paper_5m.yaml \
  research --bars 10000 --output outputs/paper_5m_swing_10000
```

Do not create the Supabase tables or enable the timer until the new research result
has been reviewed. See `PAPER_LAB_5M_V02.md`.
