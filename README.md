# PolyWeather Bot

A monitor for **Polymarket daily-temperature markets**. For 51 cities it blends
many weather forecasts into a single prediction, compares that to live Polymarket
prices, and alerts you on **Telegram** when there's a real edge — then **learns
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

Every alert also shows the **live Polymarket price next to each probability bar**
and names the **market's favourite bucket**. When the market favours a *different*
bucket than the model (and prices it ≥50¢), the alert warns loudly — a big gap is
either a real edge or a sign the model is wrong, so you can judge before betting.

### 4a. Position management
Beyond entry alerts, the bot actively manages what you hold:
- **Stop-loss exits** — alerts to CUT when the model's probability for your bucket
  drops below your break-even (your entry price), i.e. holding turns EV-negative.
- **Live-vs-forecast tracker** — `/positions` shows today's observed max so far vs
  the predicted high, so you can see if a held bet is on track or falling behind.
- **Combined net P&L** — when you hold several buckets in the *same* market (only
  one can win), `/positions` adds a netted view: total invested vs each outcome's
  real payout, and the most-likely net profit/loss.
- **Peak countdown** — every alert says whether the day's high is still forming
  ("peak in ~3h, may move") or locked in ("peak passed — most reliable").
- **Source-outlier flag** — if one API disagrees sharply with the rest, the alert
  names it so a bad/stale source is visible at a glance.
- **Morning digest** — once a day (`DIGEST_HOUR_UTC`) a single message lists the
  best edges across all cities. Mute noisy cities with `/mute <city>`.

### 4b-i. Per-city settlement station (non-airport)
Most cities settle on (or near) their airport, but a few use a **specific
station**. Hong Kong settles on the **Hong Kong Observatory (HKO HQ, urban)** —
~35 km from the airport, which can read **1–3° warmer at the airport** during the
afternoon peak. This caused a real miss: the bot read the airport at **33°C** and
predicted **33°C @ 96%**, but the HKO observatory peaked at **32°C** — and the
market settled **32°C**.

The bot now pulls live obs + settlement for these cities from the **correct
station** (`"obs"` provider in the city config). HKO's open-data feed publishes
only the *current* temperature, so the bot **tracks the running daily max across
scans** (persisted to `/data/obs_max.json` and backed up) to recover the day's
peak. Alerts show the divergence — e.g. `HKO says 32°C but the airport reports
33°C — settlement follows HKO`.

It's **pluggable**: to fix another market the same way, write `fetch_<x>_obs` /
`fetch_<x>_actual`, register them in `_OBS_PROVIDERS`, and set `"obs": "<x>"` on
that city's config. (HKO is wired in via its free open-data API.)

### 4b. Live & settlement source (airport METAR)
Polymarket settles each temperature market on a specific **airport weather
station**. The bot reads that station's live observations every scan — both via
Wunderground (the settlement feed) and the **raw airport METAR** from
`aviationweather.gov` (the same data sites like *metar-taf.com* display), using
each city's configured ICAO (Tokyo `RJTT`, NYC `KJFK`, London `EGLC`, …). Alerts
name the station and show its current/max reading; if Wunderground and the raw
airport METAR disagree by ≥1°, the alert flags it — **settlement follows the
airport**. This works for every city automatically, no extra config.

### 4c. Outgoing webhook (integrate with another bot)
Set `WEBHOOK_URL` and the bot **POSTs every signal as JSON** to your other bot's
API — including the **bias-adjusted blend, the no-bias blend, both probability
distributions, the edge/trade, live market prices and the airport reading**.
Optional `WEBHOOK_TOKEN` is sent as `Authorization: Bearer <token>`. Choose which
events to forward with `WEBHOOK_EVENTS` (`new_signal`, `bucket_shift`, `collapse`).

- **Automatic** — every signal the scan finds is forwarded on its own.
- **Manual** — after a `/scan` you get a one-tap **📡 Send** button per signal, or
  use `/send <city>` to forward that city's current signal on demand. Manual
  sends carry `"event": "manual_signal"` so your bot can tell them apart.

Test the connection any time with `/webhook`. Example payload:

```json
{
  "event": "new_signal",
  "city": "istanbul", "target_date": "2026-06-22", "unit": "°C",
  "verdict": "TRADE", "timing": "FIRMING",
  "blend": { "with_bias": 31.3, "no_bias": 31.1, "peak_bias": 0.2, "sigma": 0.33 },
  "top_bucket": 31, "top_prob": 0.73,
  "distribution":         [{ "value": 31, "probability": 0.73 }],
  "distribution_no_bias": [{ "value": 31, "probability": 0.89 }],
  "best_trade": { "action": "BUY YES", "bucket": 31, "yes_price": 0.12,
                  "edge": 0.61, "suggested_stake_usd": 17.0 },
  "market": { "favorite_bucket": 33, "favorite_price": 0.64, "url": "https://polymarket.com/..." }
}
```

### 4d. Range-bucket markets
Some cities (e.g. **San Francisco**) use **2-degree range buckets** — `68-69°F`,
`70-71°F`, … — instead of single degrees. The bot detects these automatically and
re-bins its per-degree model distribution onto the market's buckets (so the
`68-69°F` bucket gets P(68)+P(69)). Probabilities, the verdict, edges, alerts and
learning all use the correct range, and alerts show range labels (`68-69°F`,
`80+°F`, `≤61°F`). Normal single-degree markets are unaffected.

### 4e. Endgame / closing-market scanner
A **separate** alert stream for markets that are *nearly decided* — only a few
buckets still "alive" (priced above `ENDGAME_ALIVE_CENTS`) with one clear
front-runner. It fires a `🔚 ENDING MARKET` alert that simply **shows the bot's
model pick** for that market alongside the market's front-runner — no edge
comparison or filtering, just an informational heads-up so you can decide:

```
🔚 ENDING MARKET — Chengdu · 2026-06-25
Nearly decided — only 3 bucket(s) still alive (>2¢).
🏛️ Market front-runner: 25°C @ 81¢
🎯 Bot's pick: 25°C (48% model)
✅ Bot agrees with the market's front-runner.
```

It fires once per ending market, and only when the main signal *didn't* (so it
complements, never duplicates). Tune with `ENDGAME_*` vars; scan on demand with
`/endgame`. (Highest-temperature markets only.)

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
| `TRADE_MODE` | `LIVE` (alerts) or `OBSERVE` (learn only, no buy alerts) |
| `BANKROLL` | your total USD pot — turns the Kelly fraction into a concrete suggested stake per alert |

A full list is in [`.env.example`](.env.example).

---

## Commands

Send these on Telegram. There's also a **button menu**: tap the blue **Menu** button (or send
`/menu`) for one-tap access to every command, and `/` autocompletes the full list.
Typing commands manually always works too.

| Command | Does |
|---|---|
| `/menu` | Telegram button menu of every action |
| `/scan` | scan all markets now |
| `/scan london` / `/scan europe` | scan one city / region |
| `/endgame` | ending markets (nearly decided) where the bot still sees a small edge |
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
| `/watch <city> <bucket> below\|above <price>` | **custom price alert** — e.g. `/watch london 14 below 50` pings you when London 14°C YES drops to ≤50¢ (checks live CLOB prices every `PRICE_WATCH_MIN`). `/watches` lists them, `/unwatch <n>` removes one |
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
- **Command authorization:** Telegram commands are accepted only from
  `TELEGRAM_CHAT_ID`.
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
