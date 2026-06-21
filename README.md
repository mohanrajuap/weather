# PolyWeather Bot

A monitor for **Polymarket daily-temperature markets**. For 51 cities it blends
many weather forecasts into a single prediction, compares that to live Polymarket
prices, and alerts you (Telegram + ntfy) when there's a real edge — then **learns
from every outcome** so you can see whether to trust it before risking money.

> **It never trades for you.** It uses your wallet as a *read-only public address*
> and has **no private key and no order-placement code**. It can only read and
> alert — you place every trade yourself.

---

## How it works

```
51 cities ──► fetch many forecasts ──► validate & blend ──► probability per °bucket
                                                                   │
                                          live Polymarket prices ──┤
                                                                   ▼
                                  edge = model prob − market price ──► verdict
                                                                   │
                              TRADE / WAIT / SKIP  ──► alert (LIVE mode)
                                                                   │
                                        record prediction ──► settle outcome ──► learn
```

### 1. Multi-source forecasts
Each city's daily-max temperature is fetched from several independent sources in
parallel. Free, no-key sources are always on; keyed sources turn on when you add
their API key:

| Source | Key needed? | Notes |
|---|---|---|
| Open-Meteo (ECMWF, GFS, ICON, GEM + its own) | no | 5 models in 1 call |
| MET Norway (Yr) | no | independent ECMWF-based |
| US NWS (weather.gov) | no | US cities only, very accurate |
| 7Timer | no | global, coarse |
| OpenWeatherMap | `OPENWEATHER_API_KEY` | free 1M/mo |
| WeatherAPI.com | `WEATHERAPI_KEY` | free 1M/mo |
| Visual Crossing | `VISUALCROSSING_KEY` | free 1k/day |
| Tomorrow.io | `TOMORROW_API_KEY` | free ~500/day |

Every value is **auto-converted to the city's settlement unit** (°C/°F) and
**validated**: garbage is dropped, and a wrong-unit reading (e.g. °F where °C was
expected) is **auto-corrected against the consensus** of the other sources. Each
source can be turned off with `ENABLE_<SOURCE>=0`.

### 2. The blend (DEB — Dynamic Error Blending)
Sources are combined into one number, weighted by each source's recent accuracy.
A small upward **peak bias** is added because models smooth the afternoon peak
(starts at `+0.3°`, then switches to a *learned per-city bias* once history exists).
Every prediction shows **both** the raw blend and the bias-adjusted value.

### 3. Verdict (TRADE / WAIT / SKIP)
The blend produces a probability for each temperature bucket. The verdict starts at
`TRADE` and is downgraded if any guard trips: top bucket < 55%, uncertainty σ too
high, sources disagree, μ on a rounding edge, live obs conflict, pre-peak timing,
or the market has already decided. A trade only fires if it survives all of them
**and** clears `PROB_THRESHOLD` **and** has a real edge.

### 4. Edge & sizing
For each bucket, `edge = model_prob − market_price`. The bar scales with
confidence (a shaky read must clear a bigger edge). Each edge also reports
**EV** (expected return per $1) and a **Kelly** stake suggestion (use a fraction).
Set `BANKROLL` and every tradeable alert prints a **concrete stake** ("bet $X →
~N shares; wins $Y") sized at quarter-Kelly of your bankroll.

### 4a. Position management
Beyond entry alerts, the bot actively manages what you hold:
- **Stop-loss exits** — alerts to CUT when the model's probability for your bucket
  drops below your break-even (your entry price), i.e. holding turns EV-negative.
- **Live-vs-forecast tracker** — `/positions` shows today's observed max so far vs
  the predicted high, so you can see if a held bet is on track or falling behind.
- **Peak countdown** — every alert says whether the day's high is still forming
  ("peak in ~3h, may move") or locked in ("peak passed — most reliable").
- **Source-outlier flag** — if one API disagrees sharply with the rest, the alert
  names it so a bad/stale source is visible at a glance.
- **Morning digest** — once a day (`DIGEST_HOUR_UTC`) a single message lists the
  best edges across all cities. Mute noisy cities with `/mute <city>`.

### 5. Learning
Every scan records the prediction; once a market settles, the bot fetches the
actual high, scores the call, and feeds the result back into the bias learner.
This is what tells you whether the bot is trustworthy — see **Commands** below.

---

## Setup (Railway)

1. Deploy this repo to Railway with a **volume mounted at `/data`** (state + learning
   files live there and survive restarts).
2. Set the **required** variables, then redeploy.

### Required
| Variable | What |
|---|---|
| `TELEGRAM_BOT_TOKEN` | from @BotFather |
| `TELEGRAM_CHAT_ID` | your chat id (comma-separate for multiple) |

### Recommended
| Variable | What |
|---|---|
| `POLYMARKET_WALLET` | your **Polymarket profile/proxy** address (read-only) — *not* your MetaMask wallet |
| `NTFY_TOPIC` | ntfy.sh topic for push notifications |
| `TRADE_MODE` | `LIVE` (alerts) or `OBSERVE` (learn only, no buy alerts) |
| `CMD_SECRET` | secret prefix required for ntfy commands (see Security) |

A full list is in [`.env.example`](.env.example).

---

## Commands

Send these on Telegram, or publish them to your ntfy topic (prefixed with
`CMD_SECRET` if set, e.g. `mysecret /scan london`):

| Command | Does |
|---|---|
| `/scan` | scan all markets now |
| `/scan london` / `/scan europe` | scan one city / region |
| `/positions` | your live Polymarket positions + P&L, **observed-max-vs-forecast** tracker, and payout if each wins |
| `/pnl` | realized P&L ledger from settled alerts ($1 stake each), held vs missed, 7-day + lifetime |
| `/learn` | yesterday's prediction-vs-outcome scoreboard |
| `/learn all` | lifetime hit rate |
| `/learn calib` | is a predicted "70%" really 70%? (calibration) |
| `/learn sources` | which APIs are most reliable (MAE + bucket hit-rate) |
| `/learn cities` | rank cities by the bot's settled hit-rate (where it's proven vs not) |
| `/learn nobias` | hit rate **with vs without** the peak bias |
| `/missed` | $1 what-if P&L on alerts you were sent but didn't hold a position on |
| `/history <city>` | that city's full prediction-vs-outcome history + the exact `CITY_BIAS` to set |
| `/alerts [date]` | all alerts for a day grouped as one thread (defaults to today) |
| `/mute <city>` · `/unmute <city>` · `/muted` | silence a city's alerts (it keeps learning) |
| `/backup` | push learning data to GitHub now |
| `/help` | command list |

---

## Trade modes

- **`TRADE_MODE=LIVE`** — sends buy/sell signal alerts **and** learns.
- **`TRADE_MODE=OBSERVE`** — learns and tracks your held positions, but **suppresses
  buy alerts** so it can gather accuracy data hands-off while you decide whether to
  trust it. *Learning runs in both modes.*

**Suggested path to trust:** run in `OBSERVE` for 1–2 weeks → check `/learn calib`
and `/learn all` → only switch to `LIVE` (and only bet a fraction of Kelly) once the
numbers hold up.

---

## Security

- **No funds at risk from the bot.** Read-only wallet address; no private key, no
  signing, no order placement anywhere in the code.
- **Command authorization:**
  - *Telegram* commands are accepted only from `TELEGRAM_CHAT_ID`.
  - *ntfy* has no sender identity, so set **`CMD_SECRET`** — ntfy commands must then
    be prefixed with it. Without it, anyone who knows the topic name can send
    (read-only) commands.
- **Secrets** (tokens, API keys) live only in Railway env vars, never in code.
- **GitHub backup token** (optional) should be a **fine-grained** PAT scoped to this
  one repo with **Contents: Read+Write** only, ideally with an expiry.
- The repo is **public** — if you want your learning history private, use a private
  repo for the backup branch.

---

## Nightly backup

If `GITHUB_TOKEN` + `GITHUB_REPO` are set, the learning files
(`learn_history.json`, `deb_history.json`, `alerts_log.json`) are pushed nightly
(`BACKUP_HOUR_UTC`, default 02:00) to a **separate `learning-data` branch** — never
`main`, so it can't trigger a redeploy. Trigger manually any time with `/backup`.

The backup fires **at or after** the configured hour on any day it hasn't yet run,
and the last-success date is persisted to the `/data` volume — so a redeploy or
downtime that misses the exact hour still **catches up** the same day instead of
skipping it. Failures retry every 30 minutes.

**Auto-restore:** on startup, if the `/data` learning files are missing or empty
(e.g. the volume was wiped on a fresh deploy), the bot pulls them back from the
backup branch automatically. It **never overwrites** a file that already has data,
so your accumulated history is safe — backup nightly, restore on boot.

---

## Files

| File | Role |
|---|---|
| `polyweather_predict.py` | forecasts, blend, probabilities, edge, Polymarket lookup |
| `monitor.py` | scan loop, alerts, commands, positions, backup |
| `learn.py` | prediction-vs-outcome tracker & reports |
| `.env.example` | every configuration variable |

---

*Educational tool. Prediction markets are risky; the bot can be wrong. Use the
learning reports to judge it, and never stake more than you can afford to lose.*
