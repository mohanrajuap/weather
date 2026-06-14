# PolyWeather Monitor — Railway Deployment Guide

Continuous weather-market monitor that sends Telegram alerts when a city
crosses 70% win probability, and another alert when that signal collapses.

## Files
- `polyweather_predict.py` — the prediction engine (your existing bot)
- `monitor.py` — the continuous monitor + Telegram alerts
- `requirements.txt`, `Procfile`, `railway.json`, `runtime.txt` — Railway config

---

## Step 1 — Create a Telegram Bot (5 minutes)

1. Open Telegram, search for **@BotFather**
2. Send `/newbot`, follow prompts, pick a name
3. BotFather gives you a **token** like `123456:ABC-DEF...` → save it
4. Search for **@userinfobot**, send any message → it replies with your **chat id** (a number)
5. Send any message to YOUR new bot first (so it can message you back)

---

## Step 2 — Push to GitHub

```bash
cd /your/folder
git init
git add polyweather_predict.py monitor.py requirements.txt Procfile railway.json runtime.txt .gitignore
git commit -m "PolyWeather monitor"
git branch -M main
git remote add origin https://github.com/YOURNAME/polyweather-monitor.git
git push -u origin main
```

(Do NOT commit your .env file — .gitignore already excludes it.)

---

## Step 3 — Deploy on Railway

1. Go to https://railway.app → New Project → Deploy from GitHub repo
2. Select your repo
3. Railway auto-detects Python and starts building

### Add a Volume (so state survives restarts)
1. In your service → Settings → Volumes → New Volume
2. Mount path: `/data`
3. This keeps the alert-state DB so you don't get duplicate alerts after restart

### Set Environment Variables
In Railway → your service → Variables, add:

```
TELEGRAM_BOT_TOKEN = <your bot token>
TELEGRAM_CHAT_ID   = <your chat id>
CHECK_INTERVAL_MIN = 20
PROB_THRESHOLD     = 0.70
USE_PRICES         = 1
STATE_DB           = /data/monitor_state.db
POLYMARKET_WALLET  = 0xYourAddress   (for position updates)
POSITION_UPDATE_MIN = 15
```

With `POLYMARKET_WALLET` set, the monitor also sends a 💼 position update
(all open weather positions + total P&L) every `POSITION_UPDATE_MIN` minutes,
and once immediately on boot.

4. Railway redeploys automatically. Within a minute you should get the
   "PolyWeather monitor online" message in Telegram.

---

## Step 4 — Test Before Trusting It

Locally first:
```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
python monitor.py --test     # sends one test message
python monitor.py --once     # runs ONE full scan, alerts, then exits
```

If `--test` delivers a Telegram message and `--once` runs a scan, you're good
to deploy the always-on version.

---

## What The Alerts Look Like

New signal:
```
🟢 NEW SIGNAL — MILAN
📅 Market date: 2026-06-14
🎯 Predicted: 31°C at 78%
🕐 Timing: GOLDEN (local 09:00)
📊 Models: strong agreement (DEB 31.0°C)
💰 BUY YES 31°C @ 40¢ → EDGE +38%
🔗 https://polymarket.com/event/...
```

Collapse:
```
🔴 SIGNAL COLLAPSED — HONG KONG
📅 Market date: 2026-06-14
📉 30°C dropped: 81% → 45%
⚠️ Reason: live obs 1.5° above model consensus
👉 If you hold this position, reconsider.
```

---

## Tuning

- **Too many alerts?** Raise `PROB_THRESHOLD` to 0.75 or 0.80
- **Want fewer cities?** Set `ALERT_CITIES=milan,madrid,tokyo,london`
- **Scan more often?** Lower `CHECK_INTERVAL_MIN` to 10 (watch API rate limits)
- **Costs:** Railway Hobby ~$5/mo; this worker uses minimal resources

---

## Important Notes

- The monitor only ALERTS. It never places trades. You decide and click buy.
- Alerts fire only on clean TRADE-verdict signals (conflict/stale/boundary
  signals are filtered out by the engine).
- A signal alerts once when it crosses 70%, and once more if/when it collapses.
- It will not spam you with the same signal every scan.
