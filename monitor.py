"""
monitor.py — Continuous PolyWeather monitor with Telegram alerts
================================================================
Runs forever on Railway. Every CHECK_INTERVAL minutes it scans all cities and:
  • Sends a 🟢 ALERT when a city's top bucket crosses ABOVE 70% (new signal)
  • Sends a 🔴 COLLAPSE alert when a previously-alerted city drops BELOW 70%
  • Tracks state in SQLite so it doesn't re-alert the same signal repeatedly

Environment variables (set in Railway):
  TELEGRAM_BOT_TOKEN   (required) — from @BotFather
  TELEGRAM_CHAT_ID     (required) — your chat id (use @userinfobot to find it)
  POLYMARKET_WALLET    (optional) — to include your position P&L in alerts
  CHECK_INTERVAL_MIN   (optional) — minutes between scans (default 20)
  PROB_THRESHOLD       (optional) — alert threshold 0-1 (default 0.70)
  ALERT_CITIES         (optional) — comma list to limit cities (default all)
  STATE_DB             (optional) — path to state db (default /data/monitor_state.db)
  USE_PRICES           (optional) — "1" to fetch Polymarket edge in alerts (default 1)

Run locally to test:
  python monitor.py --once          # single scan, then exit
  python monitor.py                 # loop forever
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import threading
from datetime import datetime, timezone, timedelta

# Force line-buffered stdout so Railway shows the app's logs immediately instead
# of holding them in a buffer (which makes a running bot look "frozen").
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import httpx

# import the prediction engine (same folder)
import polyweather_predict as pw
# daily prediction-vs-outcome learning tracker (same folder)
import learn

import html as _html

def city_display(name: str) -> str:
    """Clean, properly-cased city name (Hong Kong, Tel Aviv, São Paulo, etc.)."""
    if not name:
        return "?"
    fixes = {"Nyc": "New York", "Sao Paulo": "São Paulo"}
    t = name.strip().title()
    return fixes.get(t, t)

def esc(s) -> str:
    """HTML-escape so names with & < > don't break Telegram's HTML parser."""
    return _html.escape(str(s)) if s is not None else ""

# ── Time helpers (IST for you + the city's local time) ────────────────────────
# Your reference timezone for every alert. Default IST (UTC+5:30); override with
# USER_TZ_OFFSET_MIN (minutes) and USER_TZ_LABEL.
_USER_TZ_MIN   = int(os.environ.get("USER_TZ_OFFSET_MIN", "330"))   # 5:30 = 330
_USER_TZ_LABEL = os.environ.get("USER_TZ_LABEL", "IST").strip()

def _fmt_clock(dt) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")     # cross-platform "8:05 AM"

def _now_user() -> str:
    return _fmt_clock(datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=_USER_TZ_MIN))

def _now_city(city_key) -> str:
    tz = (pw.CITIES.get(city_key) or {}).get("tz", 0)
    return (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=tz)).strftime("%H:%M")

def _time_footer(city_key) -> str:
    return f"🕐 Now: {city_display(city_key)} {_now_city(city_key)} · {_now_user()} {_USER_TZ_LABEL}"

def _position_advice(pos: dict, p: dict, prev_bucket) -> str:
    """Guidance when the model moved OFF the bucket you hold. Temperature buckets
    are mutually exclusive — you can't 'average' a wrong one, so spell out the real
    options (cut / hedge / hold) with your actual entry and the new price."""
    sym = p.get("temp_unit", "°")
    held = pos.get("bucket")
    entry = pos.get("avg_price"); cur = pos.get("cur_price")
    shares = pos.get("size") or 0.0; val = pos.get("current_value") or 0.0
    new_bucket = p.get("top_bucket")
    bt = p.get("best_trade")
    np_ = bt.get("yes_price") if bt else None
    e_s = f"{entry*100:.0f}¢" if entry is not None else "—"
    c_s = f"{cur*100:.0f}¢"   if cur   is not None else "—"
    np_s = f" @ {np_*100:.0f}¢" if np_ is not None else ""
    L = ["", _DIV, f"💼 <b>You hold {held}{sym}</b>: {shares:.1f} shares @ {e_s} (now {c_s}, ${val:.2f})"]
    if held is not None and new_bucket is not None and held != new_bucket:
        L.append(f"⚖️ Model moved to <b>{new_bucket}{sym}</b> — your {held}{sym} is now the underdog.")
        L.append("You can't average across buckets (only one settles). Your options:")
        L.append(f"   • <b>Cut and switch</b>: sell {held}{sym} (~${val:.2f} back) → buy {new_bucket}{sym}{np_s}")
        L.append(f"   • <b>Hedge</b>: keep {held}{sym}, also buy {new_bucket}{sym}{np_s} — covers both, but you pay twice and only one wins")
        L.append(f"   • <b>Hold</b>: only if you think the model is wrong and {held}{sym} still hits")
    return "\n".join(L)

# ── Config from environment ───────────────────────────────────────────────────
TG_TOKEN      = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT       = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
# Support multiple recipients: comma-separated in TELEGRAM_CHAT_ID
TG_CHAT_IDS   = [c.strip() for c in TG_CHAT.split(",") if c.strip()]
WALLET        = os.environ.get("POLYMARKET_WALLET", "").strip()
INTERVAL_MIN  = int(os.environ.get("CHECK_INTERVAL_MIN", "20"))
POS_UPDATE_MIN = int(os.environ.get("POSITION_UPDATE_MIN", "15"))
# Tighter watch on held positions — checks model vs your bucket and alerts on flips
POS_WATCH_MIN  = int(os.environ.get("POSITION_WATCH_MIN", "5"))
# Alert to book profit when a position is up this % or more (default 10%)
PROFIT_TAKE_PCT = float(os.environ.get("PROFIT_TAKE_PCT", "10"))
# How often to fast-recheck ACTIVE signals (ones we already alerted on) for drops
SIGNAL_WATCH_MIN = int(os.environ.get("SIGNAL_WATCH_MIN", "5"))
# If "1", only alert on RELIABLE (post-peak) signals; speculative pre-peak suppressed
RELIABLE_ONLY  = os.environ.get("RELIABLE_ONLY", "0") == "1"
# Print per-city status during each scan (why each city qualifies or not)
VERBOSE_LOG    = os.environ.get("VERBOSE_LOG", "0") == "1"
# TRADE_MODE — "LIVE" sends buy/sell signal alerts (default); "OBSERVE" keeps the
# bot scanning, learning and tracking your held positions but SUPPRESSES new-trade
# signal alerts, so it gathers calibration data hands-off while you decide whether
# to trust it. Position-management alerts (held positions, profit-taking) still fire.
TRADE_MODE     = os.environ.get("TRADE_MODE", "LIVE").strip().upper()
OBSERVE_ONLY   = TRADE_MODE in ("OBSERVE", "OBSERVER", "OBSERVE_ONLY", "PAPER")
# Nightly backup of the learning files to GitHub (off-Railway safety copy). Needs
# a GitHub token with Contents:write. Pushes to a SEPARATE branch so it never
# triggers a Railway redeploy of main. Leave GITHUB_TOKEN unset to disable.
GITHUB_TOKEN         = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO          = os.environ.get("GITHUB_REPO", "").strip()           # "owner/repo"
GITHUB_BACKUP_BRANCH = os.environ.get("GITHUB_BACKUP_BRANCH", "learning-data").strip()
BACKUP_HOUR_UTC      = int(os.environ.get("BACKUP_HOUR_UTC", "2"))         # nightly hour (UTC)
THRESHOLD     = float(os.environ.get("PROB_THRESHOLD", "0.70"))
USE_PRICES    = os.environ.get("USE_PRICES", "1") == "1"
# ── ENDGAME / closing-market scanner (separate from the main signal) ──────────
# Finds markets that are nearly decided — only a few buckets still "alive" (priced
# above ENDGAME_ALIVE_CENTS) with one clear front-runner — and just SHOWS the bot's
# model pick for that market. No edge comparison/filtering.
# ENABLE_ENDGAME controls the AUTOMATIC alerts during scans (default OFF). The
# /endgame command for an on-demand scan works regardless.
ENABLE_ENDGAME      = os.environ.get("ENABLE_ENDGAME", "0") == "1"
ENDGAME_MAX_ALIVE   = int(os.environ.get("ENDGAME_MAX_ALIVE", "3"))          # ≤ this many buckets alive
ENDGAME_ALIVE_CENTS = float(os.environ.get("ENDGAME_ALIVE_CENTS", "2")) / 100.0  # >2¢ = alive
ENDGAME_DOMINANT    = float(os.environ.get("ENDGAME_DOMINANT", "0.70"))      # front-runner ≥70¢ = ending
# Outgoing webhook — POST every signal (with bias + no-bias values) to your other
# bot's API. Set WEBHOOK_URL to enable. WEBHOOK_TOKEN (optional) is sent as a
# Bearer auth header. WEBHOOK_EVENTS picks which events to forward.
WEBHOOK_URL     = os.environ.get("WEBHOOK_URL", "").strip()
WEBHOOK_TOKEN   = os.environ.get("WEBHOOK_TOKEN", "").strip()
WEBHOOK_TIMEOUT = float(os.environ.get("WEBHOOK_TIMEOUT", "10"))
WEBHOOK_EVENTS  = {e for e in os.environ.get(
    "WEBHOOK_EVENTS", "new_signal,bucket_shift,collapse,endgame").replace(" ", "").split(",") if e}
STATE_DB      = os.environ.get("STATE_DB", "/data/monitor_state.db")
_alert_cities = os.environ.get("ALERT_CITIES", "").strip()
ALERT_CITIES  = [c.strip() for c in _alert_cities.split(",") if c.strip()] or list(pw.CITIES.keys())

# fall back to local path if /data doesn't exist (local testing)
if not os.path.isdir(os.path.dirname(STATE_DB) or "."):
    STATE_DB = "monitor_state.db"

# Your bankroll in USD — used to turn a Kelly fraction into a concrete suggested
# stake in the alert ("bet $X"). Quarter-Kelly is applied for safety.
BANKROLL       = float(os.environ.get("BANKROLL", "100"))
# Hour (UTC) to send the once-a-day morning digest of the day's best edges.
DIGEST_HOUR_UTC = int(os.environ.get("DIGEST_HOUR_UTC", "6"))
# Send the morning digest at all? (needs LIVE mode to be useful)
ENABLE_DIGEST   = os.environ.get("ENABLE_DIGEST", "1") == "1"

# ── Muted cities (runtime /mute, persisted so it survives restarts) ───────────
def _resolve_mute_file() -> str:
    env = os.environ.get("MUTE_FILE")
    if env:
        return env
    return "/data/muted.json" if os.path.isdir("/data") else "muted.json"

MUTE_FILE = _resolve_mute_file()

def load_muted() -> set:
    try:
        if os.path.exists(MUTE_FILE):
            with open(MUTE_FILE) as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()

def save_muted(muted: set):
    try:
        with open(MUTE_FILE, "w") as f:
            json.dump(sorted(muted), f)
    except Exception as e:
        print(f"[mute] save error: {e}")

MUTED = load_muted()

# ── Last successful nightly-backup date (persisted) ───────────────────────────
# Stored on the /data volume so a Railway redeploy doesn't forget it. Without this
# the in-memory timer resets on every deploy: if the redeploy lands AFTER the
# backup hour, that day's backup is silently skipped.
def _resolve_backup_stamp() -> str:
    return "/data/last_backup.txt" if os.path.isdir("/data") else "last_backup.txt"

BACKUP_STAMP = _resolve_backup_stamp()

def load_backup_day():
    try:
        with open(BACKUP_STAMP) as f:
            return datetime.strptime(f.read().strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def save_backup_day(d):
    try:
        with open(BACKUP_STAMP, "w") as f:
            f.write(d.isoformat())
    except Exception as e:
        print(f"[backup] stamp save error: {e}")

# ── Custom price watches (/watch) ─────────────────────────────────────────────
# "Alert me when London 14°C YES drops below 50¢." Persisted so they survive a
# restart; checked every PRICE_WATCH_MIN minutes against live CLOB prices.
PRICE_WATCH_MIN = int(os.environ.get("PRICE_WATCH_MIN", "3"))

def _resolve_watch_file() -> str:
    env = os.environ.get("WATCH_FILE")
    if env:
        return env
    return "/data/price_watches.json" if os.path.isdir("/data") else "price_watches.json"

WATCH_FILE = _resolve_watch_file()

def load_watches() -> list:
    try:
        if os.path.exists(WATCH_FILE):
            with open(WATCH_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_watches(watches: list):
    try:
        with open(WATCH_FILE, "w") as f:
            json.dump(watches, f, indent=2)
    except Exception as e:
        print(f"[watch] save error: {e}")

PRICE_WATCHES = load_watches()


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT_IDS:
        print(f"[telegram] not configured — would send:\n{text}\n")
        return False
    ok_any = False
    for chat_id in TG_CHAT_IDS:
        try:
            r = httpx.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                      "disable_web_page_preview": True},
                timeout=15.0,
            )
            if r.status_code != 200:
                print(f"[telegram] error to {chat_id}: {r.status_code} {r.text[:150]}")
            else:
                ok_any = True
        except Exception as e:
            print(f"[telegram] exception to {chat_id}: {e}")
    return ok_any


def alert_signal(text: str) -> bool:
    """Send a NEW-TRADE signal alert (buy / bucket-shift / collapse).

    In TRADE_MODE=OBSERVE these are suppressed — the bot still scans, learns and
    tracks held positions, it just doesn't prompt you to trade. Position-management
    alerts go through send_telegram directly and are NOT affected.
    """
    if OBSERVE_ONLY:
        return False
    return send_telegram(text)


# ── Outgoing webhook to your other bot ────────────────────────────────────────
def build_signal_payload(p, event="new_signal") -> dict:
    """Structured JSON for the external bot — includes the bias-adjusted blend,
    the no-bias blend, both probability distributions, the edge/trade, the live
    market prices and the airport reading."""
    bt  = p.get("best_trade") or {}
    mkt = _market_prices(p)
    fav = max(mkt, key=mkt.get) if mkt else None
    stake = None
    if bt.get("kelly_quarter") and bt.get("yes_price"):
        s = BANKROLL * bt["kelly_quarter"]
        stake = round(s, 2) if s >= 1 else None
    live = p.get("live") or {}
    return {
        "event":        event,                 # new_signal | bucket_shift | collapse
        "ts":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":       "polyweather",
        "mode":         TRADE_MODE,
        "city":         p.get("city"),
        "city_display": city_display(p.get("city")),
        "target_date":  p.get("target_date"),
        "predicting":   p.get("predicting"),
        "unit":         p.get("temp_unit"),
        "verdict":      p.get("verdict"),
        "timing":       (p.get("timing") or {}).get("quality"),
        "reliable":     p.get("reliable"),
        "blend": {
            "with_bias":  p.get("deb"),        # the value the bot trades on
            "no_bias":    p.get("deb_raw"),    # raw model blend, bias removed
            "peak_bias":  p.get("peak_bias"),
            "sigma":      p.get("sigma"),
        },
        "top_bucket":  p.get("top_bucket"),
        "top_prob":    p.get("top_prob"),
        "confidence":  p.get("confidence"),
        "agreement":   p.get("agreement"),
        "distribution":          p.get("distribution"),      # with bias
        "distribution_no_bias":  p.get("distribution_raw"),  # without bias
        "nobias_note": p.get("nobias_note"),
        "forecasts":   p.get("forecasts"),
        "best_trade": ({
            "action":              bt.get("action"),
            "bucket":              bt.get("temp"),
            "yes_price":           bt.get("yes_price"),
            "edge":                bt.get("best_edge"),
            "model_prob":          bt.get("model_prob"),
            "ev":                  bt.get("ev"),
            "kelly":               bt.get("kelly"),
            "kelly_quarter":       bt.get("kelly_quarter"),
            "suggested_stake_usd": stake,
        } if bt else None),
        "market": {
            "favorite_bucket": fav,
            "favorite_price":  (mkt.get(fav) if fav is not None else None),
            "prices":          {str(k): v for k, v in mkt.items()},
            "url":             (p.get("polymarket") or {}).get("url"),
        },
        "live": {
            "max_so_far":   live.get("max_so_far"),
            "current_temp": live.get("current_temp"),
            "source":       live.get("source"),
            "airport_icao": live.get("airport_icao"),
            "airport_max":  live.get("airport_max"),
        },
    }


def send_webhook(payload: dict) -> bool:
    """POST one JSON payload to WEBHOOK_URL. Never raises."""
    if not WEBHOOK_URL:
        return False
    headers = {"Content-Type": "application/json"}
    if WEBHOOK_TOKEN:
        headers["Authorization"] = f"Bearer {WEBHOOK_TOKEN}"
    try:
        r = httpx.post(WEBHOOK_URL, json=payload, headers=headers, timeout=WEBHOOK_TIMEOUT)
        if not (200 <= r.status_code < 300):
            print(f"[webhook] {r.status_code} from target: {r.text[:120]}")
            return False
        return True
    except Exception as e:
        print(f"[webhook] error: {e}")
        return False


def fire_webhook(p, event="new_signal"):
    """Build + POST a signal to the external bot in a background thread (so a slow
    target never delays the scan). No-op unless WEBHOOK_URL is set and the event
    is enabled in WEBHOOK_EVENTS."""
    if not WEBHOOK_URL or event not in WEBHOOK_EVENTS:
        return
    try:
        payload = build_signal_payload(p, event)
    except Exception as e:
        print(f"[webhook] build error: {e}")
        return
    threading.Thread(target=send_webhook, args=(payload,), daemon=True).start()


def reply_telegram(chat_id, text: str, keyboard=None) -> bool:
    """Reply to a single chat (used by the command listener)."""
    if not TG_TOKEN:
        return False
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    if keyboard:
        payload["reply_markup"] = {"inline_keyboard": keyboard}
    try:
        r = httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                       json=payload, timeout=15.0)
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] reply error: {e}")
        return False


# ── State store (SQLite) ──────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(STATE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            key          TEXT PRIMARY KEY,   -- city|target_date
            city         TEXT,
            target_date  TEXT,
            bucket       INTEGER,
            prob         REAL,
            alerted_high INTEGER DEFAULT 0,  -- 1 if we sent the >=70% alert
            last_prob    REAL,
            updated_at   TEXT
        )
    """)
    conn.commit()
    return conn


def get_state(conn, key):
    cur = conn.execute("SELECT bucket, prob, alerted_high, last_prob FROM signals WHERE key=?", (key,))
    row = cur.fetchone()
    if row:
        return {"bucket": row[0], "prob": row[1], "alerted_high": row[2], "last_prob": row[3]}
    return None


def upsert_state(conn, key, city, target_date, bucket, prob, alerted_high):
    conn.execute("""
        INSERT INTO signals (key, city, target_date, bucket, prob, alerted_high, last_prob, updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(key) DO UPDATE SET
            bucket=excluded.bucket, prob=excluded.prob,
            alerted_high=excluded.alerted_high, last_prob=excluded.last_prob,
            updated_at=excluded.updated_at
    """, (key, city, target_date, bucket, prob, alerted_high, prob,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()


# ── Alert formatting ──────────────────────────────────────────────────────────
_DIV = "━━━━━━━━━━━━━━━━━━━━"

def _peak_countdown(p) -> str:
    """One-line, plain-English read on how settled today's high is — the single
    biggest driver of whether a forecast can still move."""
    tim = p.get("timing") or {}
    q   = tim.get("quality")
    hrs = tim.get("hours_until_golden")
    pw_ = tim.get("peak_window", "?")
    if q == "RELIABLE":
        return "⏱️ <b>Peak passed</b> — today's high is essentially locked (most reliable)."
    if q == "FIRMING":
        return "⏱️ <b>Peak forming now</b> — high firming up; watch live obs."
    if q == "FORECAST":
        return "⏱️ Tomorrow's market — peak ~a day out, expect big shifts."
    if q in ("SPECULATIVE", "OVERNIGHT") and hrs is not None:
        return f"⏱️ Peak in ~{hrs:.0f}h (window {esc(pw_)}) — forecast may still move; size down."
    return ""

def _outlier_line(p) -> str:
    """Flag the single source that disagrees most with the others, so a bad/stale
    API is visible. Over time /learn sources tells you which to disable."""
    fc   = p.get("forecasts") or {}
    vals = [(m, float(v)) for m, v in fc.items() if isinstance(v, (int, float))]
    if len(vals) < 3:
        return ""
    nums = sorted(v for _, v in vals)
    med  = nums[len(nums) // 2]
    m, v = max(vals, key=lambda kv: abs(kv[1] - med))
    dev  = abs(v - med)
    sym  = p.get("temp_unit", "°")
    is_f = "F" in (p.get("temp_unit") or "")
    thresh = 5.0 if is_f else 3.0
    if dev >= thresh:
        return (f"⚠️ Source outlier: <b>{esc(m)}</b> {v:.0f}{sym} is {dev:.0f}° "
                f"off the rest (median {med:.0f}{sym}) — discounted in the blend.")
    return ""


def _range_label(lo, hi, sym):
    """Pretty label for a bucket span: '68-69°F', '80+°F', '≤61°F', or None for a
    single-degree bucket (caller falls back to the plain value)."""
    if lo is None or hi is None or lo == hi:
        return None
    if hi >= 9000:
        return f"{lo}+{sym}"
    if lo <= -9000:
        return f"≤{hi}{sym}"
    return f"{lo}-{hi}{sym}"

def _bucket_label(p, value, sym):
    """Label for a bucket value, using the market's range if it's a range market."""
    pmb = (p.get("polymarket") or {}).get("buckets") or {}
    b = pmb.get(value) or pmb.get(str(value)) or {}
    return _range_label(b.get("lo"), b.get("hi"), sym) or f"{value}{sym}"


def _market_prices(p) -> dict:
    """{bucket_value:int -> yes_price:float} from the live Polymarket book."""
    pm  = p.get("polymarket") or {}
    out = {}
    for k, v in (pm.get("buckets") or {}).items():
        try:
            key = int(k)
        except (TypeError, ValueError):
            continue
        y = (v or {}).get("yes")
        if y is not None:
            out[key] = float(y)
    return out


def fmt_new_signal(p) -> str:
    """Polished signal card with an HONEST header reflecting verdict + timing."""
    sym  = p["temp_unit"]
    tim  = p.get("timing") or {}
    ens  = p.get("ensemble") or {}
    live = p.get("live") or {}
    bt   = p.get("best_trade")
    edges = p.get("edges") or []
    city = city_display(p.get("city"))
    verdict = p.get("verdict")
    quality = tim.get("quality")

    # ── Honest header + badge based on the REAL state ──
    # A signal is only "BUY NOW" when verdict is TRADE *and* timing is at least
    # firming (peak forming/done). Pre-peak speculative = WAIT, not buy.
    if quality == "FORECAST":
        head  = "🔮 <b>FORECAST (tomorrow)</b>"
        badge = "🔮 <i>Tomorrow's peak ~a day away — forecast only, may shift a lot</i>"
        action_ok = False
    elif quality in ("SPECULATIVE", "OVERNIGHT"):
        head  = "⏳ <b>TOO EARLY — WAIT</b>"
        ps = tim.get("peak_window", "peak")
        badge = f"⏳ <i>Pre-peak (local {tim.get('city_local_now','?')}). Day's high not formed yet — wait until ~{ps}. A {p['top_prob']*100:.0f}% now can still flip.</i>"
        action_ok = False
    elif verdict == "TRADE" and bt:
        head  = "🟢 <b>SIGNAL — tradeable now</b>"
        badge = ("✅ <i>Reliable — peak observed, high essentially locked</i>"
                 if tim.get("reliable") else
                 "🟡 <i>Firming — peak forming now, watch live obs</i>")
        action_ok = True
    elif verdict == "TRADE" and not bt:
        head  = "⚪ <b>NO EDGE — skip</b>"
        badge = "⚪ <i>Model agrees with market — nothing to trade here</i>"
        action_ok = False
    else:
        head  = "🟠 <b>WAIT — not clean</b>"
        reasons = p.get("verdict_reasons") or []
        why = next((r for r in reasons if "clear signal" not in r.lower()), "signal not clean")
        badge = f"🟠 <i>{esc(why)}</i>"
        action_ok = False

    L = []
    L.append(head)
    L.append(f"📍 <b>{esc(city)}</b>  ·  {esc(p.get('target_date'))} ({esc(p.get('predicting',''))})")
    L.append(_DIV)
    L.append(f"🎯 <b>{_bucket_label(p, p['top_bucket'], sym)}</b>  at  <b>{p['top_prob']*100:.0f}%</b>")
    L.append(f"🕐 {esc(tim.get('quality','?'))} · local {esc(tim.get('city_local_now','?'))} · peak {esc(tim.get('peak_window','?'))}")
    L.append(badge)
    L.append("")

    # model section
    # Show BOTH numbers: the raw model blend and the bias-adjusted one the bot
    # actually trades on, so the peak-bias contribution is always transparent.
    _bias = p.get("peak_bias") or 0.0
    if p.get("deb_raw") is not None and abs(_bias) >= 0.05:
        L.append(f"🧬 <b>Model blend:</b> {p.get('deb')}{sym}  "
                 f"(raw {p.get('deb_raw')}{sym} {_bias:+.1f}° bias · σ {p.get('sigma')})")
    else:
        L.append(f"🧬 <b>Model blend:</b> {p.get('deb')}{sym}  (σ {p.get('sigma')})")
    if p.get("forecasts"):
        fc = " · ".join(f"{esc(m)} {v}" for m, v in p["forecasts"].items())
        L.append(f"📊 {fc}")
    if ens.get("p10") is not None:
        L.append(f"📈 Ensemble: {ens['p10']} / {ens['median']} / {ens['p90']}  (P10/Med/P90)")
    agr = p.get("agreement")
    if agr and agr != "unknown":
        ae = {"strong":"✅","moderate":"⚠️","weak":"❌"}.get(agr,"•")
        L.append(f"{ae} Agreement: {esc(agr)}")
    if live.get("current_temp") is not None:
        _srcname = live.get("source")
        if _srcname == "hko":
            src = "🏛️ HKO Observatory (settlement)"
        elif _srcname == "wunderground":
            src = "🎯 Wunderground"
        else:
            src = f"🛩️ airport {esc(live.get('airport_icao','METAR'))} METAR"
        L.append(f"🌡️ Live: {live['current_temp']}{sym} (max {live.get('max_so_far')}{sym}, {esc(live.get('trend'))}) · {src}")
    # raw airport METAR alongside — for settlement-station cities (HKO) it's NOT
    # the settlement source, just a cross-check.
    if live.get("airport_max") is not None and live.get("source") in ("wunderground", "hko"):
        icao = live.get("airport_icao", "?")
        L.append(f"🛩️ Airport {esc(icao)} METAR: now {live.get('airport_temp')}{sym} · "
                 f"max today {live.get('airport_max')}{sym}")
        if p.get("live_source_disagree"):
            primary = "HKO" if live.get("source") == "hko" else "Wunderground"
            settles = "HKO" if live.get("source") == "hko" else "the airport"
            L.append(f"   ⚠️ {primary} says {live.get('max_so_far')}{sym} but the airport "
                     f"reports {live.get('airport_max')}{sym} — settlement follows {settles}.")

    # peak countdown + source-disagreement context (data-quality at a glance)
    cd = _peak_countdown(p)
    if cd:
        L.append(cd)
    ol = _outlier_line(p)
    if ol:
        L.append(ol)

    # this city's recent track record (pred→actual) so you can judge the call
    try:
        rl = learn.recent_city_line(p.get("city"))
        if rl:
            L.append(esc(rl))
    except Exception:
        pass

    # probabilities (with bias — what the bot trades on)
    dist = p.get("distribution") or []
    # market prices per bucket — so model probabilities sit next to what the
    # market actually charges (model 73% vs market 12¢ tells the whole story).
    mkt = _market_prices(p)
    def _mp(v):
        return f"  · mkt {mkt[v]*100:.0f}¢" if v in mkt else ""

    if dist:
        L.append("")
        _plabel = "no-bias model" if p.get("no_bias_mode") else "with bias"
        L.append(f"🎲 <b>Probabilities</b> ({_plabel} · vs market price)")
        for b in dist[:4]:
            bar = "▰" * max(1, round(b['probability'] * 10))
            lbl = _range_label(b.get('lo'), b.get('hi'), sym) or f"{b['value']}{sym}"
            L.append(f"   {lbl}  {bar} {b['probability']*100:.0f}%{_mp(b['value'])}")

    # probabilities WITHOUT the per-city bias — same bars, raw model centre
    dist_raw = p.get("distribution_raw") or []
    if dist_raw and p.get("peak_bias"):
        L.append("")
        L.append(f"🎲 <b>Probabilities (no bias · raw {p.get('deb_raw')}{sym})</b>")
        for b in dist_raw[:4]:
            bar = "▱" * max(1, round(b['probability'] * 10))
            lbl = _range_label(b.get('lo'), b.get('hi'), sym) or f"{b['value']}{sym}"
            L.append(f"   {lbl}  {bar} {b['probability']*100:.0f}%{_mp(b['value'])}")
    elif p.get("nobias_note"):
        L.append("")
        L.append(f"🎲 <i>No-bias view n/a — {esc(p['nobias_note'])}</i>")

    # ── The MARKET's own favourite (Polymarket prices). If it disagrees with the
    # model's pick, say so loudly — a big gap is EITHER a real edge OR the model is
    # simply wrong, and the user needs to see the market's number to judge. ──
    if mkt:
        fav   = max(mkt, key=mkt.get)
        fav_p = mkt[fav]
        model_b = p.get("top_bucket")
        L.append("")
        fav_l = _bucket_label(p, fav, sym)
        if model_b is None:
            # model has no firm pick — just report the market's favourite
            L.append(f"🏛️ Market favours {fav_l} @ {fav_p*100:.0f}¢ ({fav_p*100:.0f}% implied)")
        elif fav != model_b:
            in_model = any(b.get("value") == fav for b in (dist or []))
            L.append(f"🏛️ <b>Market favours {fav_l} @ {fav_p*100:.0f}¢</b> "
                     f"({fav_p*100:.0f}% implied) — model picks {_bucket_label(p, model_b, sym)}.")
            if fav_p >= 0.50:
                L.append(f"   ⚠️ Market strongly disagrees with the model. The edge is "
                         f"only real if the model is right and the market is wrong — "
                         f"if unsure, trust the market's {fav_l}.")
            if not in_model:
                L.append(f"   ℹ️ The model gives {fav_l} ~0% — that's the gap to weigh.")
        else:
            L.append(f"🏛️ Market agrees with model: {fav_l} @ {fav_p*100:.0f}¢")

    # best trade — only call it a BUY when action_ok; else show as "if it holds"
    if bt:
        L.append("")
        L.append(_DIV)
        if action_ok:
            L.append(f"🏆 <b>{esc(bt['action'])} {_bucket_label(p, bt['temp'], sym)} @ {bt['yes_price']*100:.0f}¢</b>")
            L.append(f"    edge <b>{bt['best_edge']*100:+.0f}%</b> · model {bt['model_prob']*100:.0f}%")
            # concrete stake: quarter-Kelly of your bankroll → shares + payout
            kq = bt.get("kelly_quarter") or 0.0
            yp = bt.get("yes_price") or 0.0
            stake = BANKROLL * kq
            if stake >= 1 and yp > 0:
                shares = stake / yp
                payout = shares * 1.0
                L.append(f"    💵 Suggested stake <b>${stake:.0f}</b> (¼-Kelly of ${BANKROLL:.0f}) "
                         f"→ ~{shares:.0f} shares; wins ${payout:.0f} (+${payout-stake:.0f})")
            elif kq <= 0:
                L.append("    💵 Kelly says ~$0 here — edge too thin to size up.")
            if bt.get("ev"):
                L.append(f"    📊 EV {bt['ev']*100:+.0f}% per $1 staked")
            if bt.get("thin"):
                L.append(f"    ⚠️ thin volume (${bt.get('vol',0):,.0f}) — size small")
        else:
            # NOT tradeable yet — show the potential trade but tell them to WAIT
            L.append(f"⏳ <b>Potential (don't buy yet):</b> {esc(bt['action'])} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢")
            L.append(f"    would be {bt['best_edge']*100:+.0f}% edge — but wait for the peak to confirm.")
            if quality in ("SPECULATIVE", "OVERNIGHT"):
                L.append(f"    👉 Re-check during the peak window ({esc(tim.get('peak_window','?'))}).")
    elif edges and action_ok:
        actionable = [e for e in edges if e["action"] in ("BUY YES","BUY NO")][:3]
        if actionable:
            L.append("")
            L.append("💰 <b>Edges</b>")
            for e in actionable:
                ys = f"{e['yes_price']*100:.0f}¢" if e.get("yes_price") is not None else "—"
                L.append(f"   {esc(e['action'])} {e['temp']}{sym} @ {ys} → {e['best_edge']*100:+.0f}%")

    pm = p.get("polymarket")
    if pm and pm.get("url"):
        L.append("")
        L.append(f'🔗 <a href="{esc(pm["url"])}">Open on Polymarket</a>')
    L.append(_time_footer(p.get("city")))
    return "\n".join(L)


def fmt_collapse(p, prev_prob) -> str:
    sym       = p["temp_unit"]
    new_prob  = p.get("top_prob", 0)
    verdict   = p.get("verdict")
    reasons   = p.get("verdict_reasons") or []
    city      = city_display(p.get("city"))

    if verdict != "TRADE":
        bad = [r for r in reasons if "clear signal" not in r.lower()]
        reason = bad[0] if bad else "signal no longer clean"
        headline = "model no longer confident"
    else:
        reason = (f"slipped below your {THRESHOLD*100:.0f}% alert bar "
                  f"(still a {new_prob*100:.0f}% lean)")
        headline = "confidence easing"

    L = [
        f"🔴 <b>SIGNAL WEAKENED</b>",
        f"📍 <b>{esc(city)}</b>  ·  {esc(p.get('target_date'))}",
        _DIV,
        f"📉 {p['top_bucket']}{sym}:  {prev_prob*100:.0f}% → <b>{new_prob*100:.0f}%</b>",
        f"<i>{esc(headline)}</i>",
        f"⚠️ {esc(reason)}",
    ]
    for r in reasons:
        rl = r.lower()
        if ("conflict" in rl or "stale" in rl or "boundary" in rl or "disagree" in rl) and r not in reason:
            L.append(f"🚨 {esc(r)}")
            break
    L.append("")
    L.append(f"👉 If you hold {p['top_bucket']}{sym}, reconsider — edge shrinking.")
    L.append(_time_footer(p.get("city")))
    return "\n".join(L)


def fmt_bucket_shift(p, prev_bucket, prev_prob, position=None) -> str:
    """The model's top bucket CHANGED while still high-confidence.
    e.g. morning said 32°C@70%, now says 28°C@70%. This is a different prediction."""
    sym = p["temp_unit"]
    new_bucket = p["top_bucket"]
    new_prob   = p["top_prob"]
    direction = "📈 higher" if new_bucket > prev_bucket else "📉 lower"
    city = city_display(p.get("city"))
    lines = [
        f"🔄 <b>PREDICTION CHANGED</b>",
        f"📍 <b>{esc(city)}</b>  ·  {esc(p.get('target_date'))}",
        _DIV,
        f"Moved {direction}:",
        f"   before  <b>{prev_bucket}{sym}</b> @ {prev_prob*100:.0f}%",
        f"   now     <b>{new_bucket}{sym}</b> @ {new_prob*100:.0f}%",
    ]
    # show where the old bucket sits now
    dist = {b["value"]: b["probability"] for b in p.get("distribution", [])}
    old_now = dist.get(prev_bucket)
    if old_now is not None:
        lines.append(f"   your old {prev_bucket}{sym} is now {old_now*100:.0f}%")
    lines.append("")
    lines.append(f"👉 Different bucket than before — if you hold {prev_bucket}{sym}, reconsider.")
    # edge on the new bucket
    bt = p.get("best_trade")
    if bt:
        lines.append("")
        lines.append(f"💰 New best: {esc(bt['action'])} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢ → {bt['best_edge']*100:+.0f}%")
    # if you hold a position on this city, advise how to manage it
    if position and position.get("bucket") is not None:
        lines.append(_position_advice(position, p, prev_bucket))
    pm = p.get("polymarket")
    if pm and pm.get("url"):
        lines.append(f'🔗 {pm["url"]}')
    lines.append(_time_footer(p.get("city")))
    return "\n".join(lines)


# ── Endgame / closing-market scanner ──────────────────────────────────────────
def check_endgame(p):
    """Spot a nearly-decided market (only a few buckets still alive, one clear
    front-runner) and report the bot's model pick for it. NO edge comparison —
    this just flags ending markets and shows what the model thinks; you decide."""
    pm = p.get("polymarket") or {}
    buckets = pm.get("buckets") or {}
    if len(buckets) < 3:
        return None
    alive = [(k, v.get("yes")) for k, v in buckets.items()
             if v.get("yes") is not None and v.get("yes") >= ENDGAME_ALIVE_CENTS]
    # "ending" = only a few buckets still alive AND one clear front-runner
    if not (1 <= len(alive) <= ENDGAME_MAX_ALIVE):
        return None
    dom_k, dom_y = max(alive, key=lambda x: x[1])
    if dom_y < ENDGAME_DOMINANT:
        return None
    if dom_y >= 0.99:                 # already fully resolved — nothing to say
        return None
    top_b = p.get("top_bucket")
    model = {b["value"]: b["probability"] for b in (p.get("distribution") or [])}
    pick  = buckets.get(top_b) or {}
    domb  = buckets.get(dom_k) or {}
    return {
        "city": p["city"], "sym": p.get("temp_unit", "°"), "target_date": p.get("target_date"),
        "pick": top_b, "pick_prob": model.get(top_b, 0.0) if top_b is not None else 0.0,
        "pick_price": pick.get("yes"), "pick_lo": pick.get("lo"), "pick_hi": pick.get("hi"),
        "dom": dom_k, "dom_price": dom_y, "dom_lo": domb.get("lo"), "dom_hi": domb.get("hi"),
        "agrees": top_b == dom_k, "alive": len(alive), "url": pm.get("url"),
    }


def fmt_endgame(o) -> str:
    sym  = o["sym"]
    city = city_display(o["city"])
    dom_lbl  = _range_label(o.get("dom_lo"), o.get("dom_hi"), sym) or f"{o['dom']}{sym}"
    pick_lbl = ((_range_label(o.get("pick_lo"), o.get("pick_hi"), sym) or f"{o['pick']}{sym}")
                if o.get("pick") is not None else "—")
    L = [
        "🔚 <b>ENDING MARKET</b>",
        f"📍 <b>{esc(city)}</b>  ·  {esc(o['target_date'])}",
        _DIV,
        f"Nearly decided — only <b>{o['alive']}</b> bucket(s) still alive "
        f"(&gt;{ENDGAME_ALIVE_CENTS*100:.0f}¢).",
        f"🏛️ Market front-runner: <b>{dom_lbl} @ {o['dom_price']*100:.0f}¢</b>",
        f"🎯 <b>Bot's pick: {pick_lbl}</b> ({o['pick_prob']*100:.0f}% model)",
    ]
    if o["agrees"]:
        L.append("✅ Bot agrees with the market's front-runner.")
    else:
        pp = f" — trades at {o['pick_price']*100:.0f}¢" if o.get("pick_price") is not None else ""
        L.append(f"⚠️ Bot disagrees — it favours a different bucket{pp}.")
    if o.get("url"):
        L.append(f'🔗 <a href="{esc(o["url"])}">Open on Polymarket</a>')
    L.append(_time_footer(o["city"]))
    return "\n".join(L)


# ── Position update ───────────────────────────────────────────────────────────
def _match_city_from_title(title: str):
    """Find which registered city a Polymarket position title refers to."""
    t = (title or "").lower()
    # try each city name and alias
    for ck in pw.CITIES:
        if ck in t:
            return ck
    for alias, ck in pw.ALIASES.items():
        if alias in t.replace(" ", ""):
            return ck
    return None

def _extract_pos_bucket(title: str):
    """Pull the temperature bucket from a position title (not the date)."""
    import re
    t = title or ""
    # Polymarket position titles often look like:
    #   "Will the highest temperature in Istanbul on June 14 be 24°C?"
    # Prefer a number directly attached to a degree symbol or C/F.
    m = re.search(r"(\d{1,3})\s*°\s*[CF]", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,3})\s*°", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,3})\s*[CF]\b", t)
    if m:
        return int(m.group(1))
    # "be 24" style (number after 'be')
    m = re.search(r"\bbe\s+(\d{1,3})\b", t, re.I)
    if m:
        return int(m.group(1))
    # "24 or below / or higher / or above"
    m = re.search(r"(\d{1,3})\s*(?:or (?:below|higher|above|lower))", t, re.I)
    if m:
        return int(m.group(1))
    return None

def _market_key(title: str):
    """Group positions that belong to the SAME market (same city + same date) so
    multiple buckets you hold can be netted — only one of them can ever settle."""
    import re
    t = title or ""
    mcity = _match_city_from_title(t) or t[:20].lower()
    md = re.search(r"([A-Za-z]{3,9}\s+\d{1,2})", t)     # e.g. "June 21"
    return f"{mcity}|{md.group(1).lower() if md else ''}"


def fmt_positions_update(wallet: str, positions) -> str:
    if positions is None:
        return "⚠️ Could not fetch your positions (check wallet / network)."
    if not positions:
        return "💼 <b>Positions update</b>\nNo open weather positions right now."

    import re
    lines = ["💼 <b>YOUR POSITIONS</b> (with live model read)\n"]
    total_now  = 0.0
    total_paid = 0.0
    total_pnl  = 0.0
    claimable  = []
    groups     = {}   # market_key -> [bucket legs] for netting multi-bucket holdings

    for pos in positions:
        title = (pos.get("title") or "")
        side  = (pos.get("outcome") or "?")
        entry = pos.get("avg_price")
        now   = pos.get("cur_price")
        val   = pos.get("current_value") or 0
        pnl   = pos.get("cash_pnl") or 0
        total_now  += val
        total_paid += pos.get("initial_value") or 0
        total_pnl  += pnl
        if pos.get("redeemable"):
            claimable.append(pos)

        e_s   = f"{entry*100:.0f}¢" if entry is not None else "—"
        n_s   = f"{now*100:.0f}¢"   if now   is not None else "—"
        emoji = "🟢" if pnl >= 0 else "🔴"

        short = title[:34]
        # readable header: pull "City — bucket" out of the long weather title
        # (they all start "Will the highest temperature in…"), else truncate.
        _m = re.search(r"\bin\s+(.+?)\s+on\b.*?\bbe\s+(.+?)\s*\??$", title, re.I)
        hdr = f"{_m.group(1)} — {_m.group(2)}" if _m else title[:40]
        lines.append(f"{emoji} <b>{esc(hdr)}</b>")
        lines.append(f"   {side} | {e_s}→{n_s} | ${val:.2f} (P&L {pnl:+.2f})")
        # ── winning amount: each share pays $1 if it settles in your favour ──
        shares = pos.get("size") or 0.0
        cost   = pos.get("initial_value") or 0.0
        if shares > 0:
            payout  = shares * 1.0
            wprofit = payout - cost
            roi     = (wprofit / cost * 100) if cost > 0 else 0
            lines.append(f"   🏆 If it WINS: <b>${payout:.2f}</b> back  "
                         f"(paid ${cost:.2f} → profit {wprofit:+.2f}, {roi:+.0f}%)")
        if pos.get("redeemable"):
            lines.append(f"   ✅ SETTLED — claimable")

        # ── live model read for THIS position's bucket ──
        city   = _match_city_from_title(title)
        bucket = _extract_pos_bucket(title)
        cur_price = pos.get("cur_price")   # market's current price for your side
        g_model_prob = None                # this leg's model prob (for the net view)
        if city and bucket is not None:
            try:
                pred = pw.predict(city, fetch_prices=False)
                sym  = pred.get("temp_unit", "°")
                dist = {b["value"]: b["probability"] for b in pred.get("distribution", [])}
                your_prob = dist.get(bucket, 0.0)
                g_model_prob = your_prob
                top_b   = pred.get("top_bucket")
                top_p   = pred.get("top_prob", 0)
                tim     = pred.get("timing") or {}
                peak_passed = tim.get("quality") == "RELIABLE"  # post-peak, high locked

                # ── live-vs-forecast tracker: is today actually heading to your bucket? ──
                live = pred.get("live") or {}
                msf  = live.get("max_so_far")
                pdeb = pred.get("deb")
                if msf is not None:
                    if msf >= bucket:
                        track = "✅ already at/above your bucket — looking good"
                    elif pdeb is not None and msf >= pdeb - 0.5:
                        track = "🟢 on track to forecast"
                    elif pdeb is not None and msf >= pdeb - 2.0:
                        track = "🟡 running a bit behind forecast"
                    else:
                        track = "🔴 behind forecast — watch it"
                    pstr = f" vs predicted {pdeb:.0f}{sym}" if pdeb is not None else ""
                    lines.append(f"   🌡️ Today's max so far {msf:.0f}{sym}{pstr} — {track}")

                # ── The MARKET is the source of truth once the peak is in. ──
                # If your position is priced high (market thinks you'll win) OR the
                # peak has passed, the model's "forecast" of a higher bucket is stale
                # (it's still predicting a peak that already happened). Trust the market.
                market_winning = cur_price is not None and cur_price >= 0.80

                if market_winning:
                    lines.append(f"   🟢 Market: {bucket}{sym} winning at {cur_price*100:.0f}¢ — likely settles in your favor")
                elif peak_passed and cur_price is not None and cur_price >= 0.50:
                    lines.append(f"   🟢 Peak passed · {bucket}{sym} holding at {cur_price*100:.0f}¢")
                else:
                    # model read is meaningful only PRE/DURING peak
                    icon = "🟢" if your_prob >= 0.55 else "🟡" if your_prob >= 0.30 else "🔴"
                    lines.append(f"   {icon} Model: your {bucket}{sym}={your_prob*100:.0f}% "
                                 f"| top {top_b}{sym}={top_p*100:.0f}%")
                    # only warn if NOT already winning on the market AND peak not passed
                    if your_prob < 0.30 and top_b != bucket and not peak_passed:
                        lines.append(f"   ⚠️ Model favors {top_b}{sym} — watch closely")
                    if pred.get("live_model_conflict"):
                        lines.append(f"   🚨 live/model conflict — uncertain")
            except Exception:
                pass

        # record this leg for the per-market net summary (only one bucket can win)
        if bucket is not None and shares > 0:
            gsym = "°F" if (pw.CITIES.get(city or "") or {}).get("f") else "°C"
            md   = re.search(r"([A-Za-z]{3,9}\s+\d{1,2})", title)
            groups.setdefault(_market_key(title), []).append({
                "bucket": bucket, "cost": cost, "payout": shares * 1.0,
                "prob": g_model_prob, "cur": now, "sym": gsym, "side": side,
                "cityd": city_display(city) if city else short,
                "date": md.group(1) if md else "",
            })
        lines.append("")

    # ── Combined net view for markets where you hold MULTIPLE buckets ──
    # Only one bucket settles, so you pay for every leg but collect on one. This
    # nets the cost of the losing legs against each winning scenario's payout.
    for legs in groups.values():
        if len(legs) < 2:
            continue
        # "Only ONE can win" holds only for YES legs across distinct buckets. A NO
        # leg can settle in your favour alongside a YES leg, so the netting would be
        # wrong — skip the combined view for mixed/NO groups.
        if not all((g.get("side") or "").lower() == "yes" for g in legs):
            continue
        legs.sort(key=lambda g: (g["prob"] if g["prob"] is not None
                                 else (g["cur"] or 0)), reverse=True)
        tcost = sum(g["cost"] for g in legs)
        sym   = legs[0]["sym"]
        hdr   = f"{legs[0]['cityd']} {legs[0]['date']}".strip()
        lines.append("━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📊 <b>Combined: {esc(hdr)}</b> — {len(legs)} buckets, only ONE can win")
        lines.append(f"   Total invested across all legs: <b>${tcost:.2f}</b>")
        for g in legs:
            net = g["payout"] - tcost           # collect this leg, lose all others
            ne  = "🟢" if net >= 0 else "🔴"
            if g["prob"] is not None:
                pr = f"model {g['prob']*100:.0f}%"
            elif g["cur"] is not None:
                pr = f"mkt {g['cur']*100:.0f}¢"
            else:
                pr = "—"
            lines.append(f"   {ne} If {g['bucket']}{sym} wins ({pr}): "
                         f"collect ${g['payout']:.2f} → net <b>{net:+.2f}</b>")
        best  = legs[0]
        bnet  = best["payout"] - tcost
        word  = "PROFIT" if bnet >= 0 else "LOSS"
        emoji = "🟢" if bnet >= 0 else "🔴"
        lines.append(f"   👉 Most likely ({best['bucket']}{sym}): {emoji} <b>{word} "
                     f"${abs(bnet):.2f}</b> on ${tcost:.2f} invested")

    tot_e = "🟢" if total_pnl >= 0 else "🔴"
    roi = (total_pnl / total_paid * 100) if total_paid > 0 else 0
    lines.append("━━━━━━━━━━━━━━━━━━━")
    lines.append(f"{tot_e} <b>Total: ${total_now:.2f}  P&L {total_pnl:+.2f} ({roi:+.1f}%)</b>")
    if claimable:
        lines.append(f"✅ {len(claimable)} settled & claimable — go redeem!")
    return "\n".join(lines)


def send_position_update():
    if not WALLET:
        print("  ⚠️ position update skipped — POLYMARKET_WALLET not set")
        return False
    try:
        # Fetch EVERYTHING first so we can tell apart the three failure modes:
        #   • None  → API/address problem (often: wrong wallet TYPE)
        #   • []    → address valid but holds no positions
        #   • >0 but 0 weather → positions exist, weather filter hid them
        all_pos = pw.fetch_positions(WALLET, weather_only=False)
        masked  = f"{WALLET[:6]}…{WALLET[-4:]}"
        if all_pos is None:
            print(f"  ⚠️ positions API returned nothing for {masked} — check the address. "
                  f"It must be your Polymarket PROFILE/deposit address (polymarket.com/profile), "
                  f"NOT your MetaMask/signing wallet.")
            positions = None
        else:
            weather = [p for p in all_pos
                       if ("temperature" in (p.get("title") or "").lower()
                           or "temp" in (p.get("event_slug") or "").lower()
                           or "weather" in (p.get("event_slug") or "").lower())]
            print(f"  💼 positions for {masked}: {len(all_pos)} total, "
                  f"{len(weather)} weather")
            # Show EVERYTHING the wallet holds — don't hide non-weather bets. The
            # live model read still applies only to the weather ones.
            positions = all_pos
        msg = fmt_positions_update(WALLET, positions)
        send_telegram(msg)
        return True
    except Exception as e:
        print(f"  position update error: {e}")
        return False


def watch_positions(conn):
    """
    Tight watch on HELD positions. For each open weather position, check the
    live model + market for that exact bucket. Alert immediately if:
      • model has flipped AWAY from your bucket (your bucket no longer top pick)
      • your bucket's probability dropped sharply since last watch
      • market price moved hard against you
    Only alerts on CHANGE, so it won't spam.
    """
    if not WALLET:
        return
    try:
        positions = pw.fetch_positions(WALLET, weather_only=True)
    except Exception as e:
        print(f"  watch error: {e}")
        return
    if not positions:
        return

    for pos in positions:
        title  = pos.get("title") or ""
        if pos.get("redeemable"):
            continue  # already settled

        # ── PROFIT-TAKE ALERT ──
        # If this position is up >= PROFIT_TAKE_PCT, suggest booking the gain.
        pct_pnl = pos.get("percent_pnl")
        cash    = pos.get("cash_pnl") or 0
        if pct_pnl is not None and pct_pnl >= PROFIT_TAKE_PCT:
            pkey = f"profit|{title[:40]}"
            prev = get_state(conn, pkey)
            already = prev["alerted_high"] if prev else 0
            if not already:
                cur = pos.get("cur_price")
                ent = pos.get("avg_price")
                val = pos.get("current_value") or 0
                lines = [
                    f"💰 <b>PROFIT ALERT — book the gain?</b>",
                    f"📈 {title[:46]}",
                    f"   Up <b>{pct_pnl:+.0f}%</b> (+${cash:.2f}) — now worth ${val:.2f}",
                ]
                if ent is not None and cur is not None:
                    lines.append(f"   Entry {ent*100:.0f}¢ → now {cur*100:.0f}¢")
                lines.append(f"👉 You're past your {PROFIT_TAKE_PCT:.0f}% target. Consider selling to lock it in.")
                send_telegram("\n".join(lines))
                upsert_state(conn, pkey, "profit", "", 0, pct_pnl/100.0, alerted_high=1)
                print(f"  💰 PROFIT alert: {title[:30]} +{pct_pnl:.0f}%")
        elif pct_pnl is not None and pct_pnl < (PROFIT_TAKE_PCT * 0.5):
            # dropped well back below target → reset so we can alert again later
            pkey = f"profit|{title[:40]}"
            prev = get_state(conn, pkey)
            if prev and prev["alerted_high"]:
                upsert_state(conn, pkey, "profit", "", 0, max(pct_pnl, 0)/100.0, alerted_high=0)

        city   = _match_city_from_title(title)
        bucket = _extract_pos_bucket(title)
        if not city or bucket is None:
            continue

        try:
            pred = pw.predict(city, fetch_prices=True)
        except Exception:
            continue

        sym  = pred.get("temp_unit", "°")
        dist = {b["value"]: b["probability"] for b in pred.get("distribution", [])}
        your_prob = dist.get(bucket, 0.0)
        top_b     = pred.get("top_bucket")
        top_p     = pred.get("top_prob", 0)
        tim       = pred.get("timing") or {}
        peak_passed = tim.get("quality") == "RELIABLE"

        # Market truth: your position's current price. If the market already
        # prices your bucket as a likely winner, the model's stale forecast of a
        # higher bucket is NOISE — don't fire a scary "flipped away" alert.
        cur_price = pos.get("cur_price")
        market_winning = cur_price is not None and cur_price >= 0.70

        # state key for the watch (per position bucket)
        wkey = f"watch|{city}|{pred.get('target_date')}|{bucket}"
        prev = get_state(conn, wkey)
        prev_prob = prev["prob"] if prev else your_prob

        # conditions to alert
        flipped_away = (top_b is not None and top_b != bucket and your_prob < 0.40)
        big_drop     = (prev_prob - your_prob) >= 0.15   # dropped 15+ pts since last watch
        conflict     = pred.get("live_model_conflict")
        # ── STOP-LOSS: holding is EV-negative once the model's probability for your
        # bucket falls below what you paid for it (break-even = your entry price). ──
        entry        = pos.get("avg_price")
        breakeven    = entry if entry is not None else cur_price
        below_breakeven = (breakeven is not None and your_prob < (breakeven - 0.05)
                           and not peak_passed and not market_winning)

        # Suppress the alert entirely if the market says you're winning, or the
        # peak already passed (the model can't "un-happen" a locked-in high).
        if market_winning or peak_passed:
            upsert_state(conn, wkey, city, pred.get("target_date"), bucket, your_prob, alerted_high=0)
            continue

        # only alert once per worsening state — track with alerted_high flag
        already = prev["alerted_high"] if prev else 0

        if (flipped_away or big_drop or below_breakeven) and not already:
            head_icon = "🛑" if below_breakeven else "⚡"
            lines = [
                f"{head_icon} <b>POSITION WATCH — {esc(city_display(city))}</b>",
                f"📅 {pred.get('target_date')}  |  you hold <b>{bucket}{sym}</b>",
                "",
            ]
            if below_breakeven:
                lines.append(f"🛑 <b>STOP-LOSS</b>: model now <b>{your_prob*100:.0f}%</b> for {bucket}{sym}, "
                             f"below your break-even <b>{breakeven*100:.0f}¢</b>.")
                lines.append(f"   Holding is EV-negative — cutting now caps the loss.")
            if flipped_away:
                lines.append(f"🔴 Model FLIPPED away from your bucket:")
                lines.append(f"   your {bucket}{sym} now only <b>{your_prob*100:.0f}%</b>")
                lines.append(f"   model's top pick is now <b>{top_b}{sym} ({top_p*100:.0f}%)</b>")
            elif big_drop:
                lines.append(f"📉 Your {bucket}{sym} dropped fast: "
                             f"{prev_prob*100:.0f}% → <b>{your_prob*100:.0f}%</b>")
            if conflict:
                lines.append(f"🚨 live/model conflict — extra uncertainty")
            # market price for your bucket
            pm = pred.get("polymarket") or {}
            pmb = (pm.get("buckets") or {}).get(bucket) or {}
            if pmb.get("yes") is not None:
                lines.append(f"💰 Market price for {bucket}{sym}: {pmb['yes']*100:.0f}¢")
            lines.append("")
            lines.append(f"👉 Consider cutting {bucket}{sym} — model no longer backs it.")
            if pm.get("url"):
                lines.append(f"🔗 {pm['url']}")
            send_telegram("\n".join(lines))
            upsert_state(conn, wkey, city, pred.get("target_date"), bucket, your_prob, alerted_high=1)
            print(f"  ⚡ POSITION WATCH alert {city} {bucket}° now {your_prob*100:.0f}%")
        elif your_prob >= 0.55 and already:
            # recovered — reset so we can warn again if it drops later
            upsert_state(conn, wkey, city, pred.get("target_date"), bucket, your_prob, alerted_high=0)
        else:
            upsert_state(conn, wkey, city, pred.get("target_date"), bucket, your_prob,
                         alerted_high=already)


def watch_active_signals(conn):
    """
    Fast re-check of signals we've ALREADY alerted on (alerted_high=1).
    Runs every SIGNAL_WATCH_MIN minutes — far quicker than the 20-min full scan —
    so if an active 70% signal starts dropping, you hear about it immediately,
    not up to 20 minutes later.

    Only re-checks already-alerted, non-watch keys (skips the 'watch|' position keys).
    """
    cur = conn.execute(
        "SELECT key, city, target_date, bucket, prob FROM signals "
        "WHERE alerted_high=1 AND key NOT LIKE 'watch|%'"
    )
    rows = cur.fetchall()
    if not rows:
        return

    for key, city, tdate, prev_bucket, prev_prob in rows:
        try:
            p = pw.predict(city, fetch_prices=USE_PRICES)
        except Exception:
            continue
        if "error" in p:
            continue

        # only relevant if still the same target market
        if p.get("target_date") != tdate:
            continue

        prob    = p.get("top_prob", 0.0)
        bucket  = p.get("top_bucket")
        verdict = p.get("verdict")
        clean   = verdict == "TRADE"

        collapsed      = (prob < THRESHOLD or not clean)
        bucket_shifted = (clean and prob >= THRESHOLD
                          and prev_bucket is not None and bucket != prev_bucket)

        if bucket_shifted:
            alert_signal(fmt_bucket_shift(p, prev_bucket, prev_prob))
            fire_webhook(p, "bucket_shift")
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=1)
            print(f"  ⚡🔄 FAST SHIFT {city} {prev_bucket}°→{bucket}°")
        elif collapsed:
            alert_signal(fmt_collapse(p, prev_prob if prev_prob is not None else prob))
            fire_webhook(p, "collapse")
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=0)
            print(f"  ⚡🔴 FAST COLLAPSE {city} {prob*100:.0f}%")
        else:
            # still healthy — just refresh stored prob
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=1)


# ── Learning: record yesterday's actual highs ─────────────────────────────────
def backfill_actuals():
    """
    Close the learning loop. For every city, look at the last couple of COMPLETED
    local days that we saved forecasts for but haven't recorded an actual high
    for yet, fetch the settled daily max, and store it. Once a city has >=2 days
    of actuals, deb_blend stops using the flat PEAK_BIAS and switches to the
    learned per-city signed bias automatically.

    Cheap to run repeatedly: it only hits the network for day/city pairs that
    have forecasts but no actual yet, and skips everything already recorded.
    """
    try:
        history = pw._load_history()
    except Exception as e:
        print(f"  [learn] could not load history: {e}")
        return 0

    now_utc  = pw._now_utc()
    recorded = 0
    for city_key, meta in pw.CITIES.items():
        city_hist = history.get(city_key, {})
        if not city_hist:
            continue
        local_now = now_utc + timedelta(seconds=meta.get("tz", 0))
        # yesterday and the day before, in this city's local time (both complete)
        for back in (1, 2):
            d = (local_now - timedelta(days=back)).strftime("%Y-%m-%d")
            rec = city_hist.get(d)
            if not rec or not rec.get("forecasts"):
                continue                       # no forecast saved → nothing to learn
            if rec.get("actual_high") is not None:
                continue                       # already recorded
            try:
                actual = pw.fetch_actual_high(city_key, d)
            except Exception:
                actual = None
            if actual is not None:
                pw.record_actual(city_key, d, actual)
                recorded += 1

    if recorded:
        print(f"  [learn] recorded {recorded} actual high(s) — DEB will adapt")
    return recorded


# ── One monitoring pass ───────────────────────────────────────────────────────
def run_scan(conn):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[{ts}] scanning {len(ALERT_CITIES)} cities (threshold {THRESHOLD*100:.0f}%)...")

    # Pull every city's airport METAR in ONE batched call (not 51) so we don't
    # rate-limit aviationweather.gov.
    try:
        pw.prefetch_metars([(pw.CITIES.get(c) or {}).get("icao") for c in ALERT_CITIES])
    except Exception as e:
        print(f"  [metar] prefetch failed: {e}")

    new_alerts = 0
    collapses  = 0
    shifts     = 0
    # near-miss tally: cities that were close but filtered
    nm = {"no_edge": 0, "pre_peak": 0, "below_thresh": 0, "not_trade": 0, "high_ok": 0}
    # data-health tally: how many cities we could actually evaluate vs lost to
    # errors / missing forecast data. Without this, a total upstream outage looks
    # identical to a calm "0 new" market in the logs.
    health = {"total": len(ALERT_CITIES), "errored": 0, "no_data": 0,
              "evaluated": 0, "no_market": 0, "muted": 0}
    scan_preds = []   # collected for the learning tracker (one disk write at end)

    # Your current positions per city — used for the missed-trade tracker AND for
    # position-management advice when a prediction changes off a bucket you hold.
    held_positions = {}
    if WALLET:
        try:
            for pos in (pw.fetch_positions(WALLET, weather_only=True) or []):
                ck = _match_city_from_title(pos.get("title", ""))
                if ck:
                    held_positions[ck] = {**pos, "bucket": _extract_pos_bucket(pos.get("title", ""))}
        except Exception as e:
            print(f"  [missed] position check failed: {e}")
    held_cities = set(held_positions.keys())

    for city in ALERT_CITIES:
        try:
            p = pw.predict(city, fetch_prices=USE_PRICES)
        except Exception as e:
            print(f"  {city}: error {e}")
            health["errored"] += 1
            continue
        if "error" in p:
            health["errored"] += 1
            continue
        # A city with no probability distribution had no usable forecast data
        # (all upstream fetches failed) — it was scanned but never truly evaluated.
        if not p.get("distribution"):
            health["no_data"] += 1
            continue
        health["evaluated"] += 1
        scan_preds.append(p)   # snapshot this city's current prediction for learning

        # Muted city (/mute): keep learning + tracking, just don't alert on it.
        if city in MUTED or (pw.resolve_city(city) or city) in MUTED:
            health["muted"] += 1
            continue

        # No live Polymarket market today (weekend/holiday/not listed) — nothing to
        # trade. We still recorded the forecast for learning above; skip alert logic.
        # Only when prices are ON: with USE_PRICES off we never fetch buckets, so an
        # empty market is expected and the bot alerts on probability alone.
        _pm = p.get("polymarket") or {}
        if USE_PRICES and not _pm.get("buckets"):
            health["no_market"] += 1
            continue

        prob   = p.get("top_prob", 0.0)
        bucket = p.get("top_bucket")
        tdate  = p.get("target_date")
        key    = f"{p['city']}|{tdate}"
        verdict = p.get("verdict")

        # only treat as a real signal if it's a clean TRADE verdict
        clean = (verdict == "TRADE")

        # require an actual tradeable edge — no point alerting when the market
        # already agrees with the model (no >10% edge = nothing to do)
        has_edge = p.get("best_trade") is not None

        prev = get_state(conn, key)
        prev_alerted = prev["alerted_high"] if prev else 0
        prev_prob    = prev["prob"] if prev else None
        prev_bucket  = prev["bucket"] if prev else None

        # if USE_PRICES is on, require edge; if off, fall back to prob only
        reliable_ok = (not RELIABLE_ONLY) or p.get("reliable", False)
        signal_ok = (clean and prob >= THRESHOLD
                     and (has_edge or not USE_PRICES)
                     and reliable_ok)

        crossed_up = signal_ok and not prev_alerted
        collapsed  = prev_alerted and (prob < THRESHOLD or not clean)
        # bucket shift: we previously alerted, it's STILL high+clean, but the
        # winning bucket CHANGED (e.g. 32°C@70% → 28°C@70%)
        bucket_shifted = (prev_alerted and clean and prob >= THRESHOLD
                          and prev_bucket is not None and bucket != prev_bucket)

        # ── verbose per-city logging — shows WHY a city does/doesn't alert ──
        if prob >= 0.50:
            # tally reason
            if not clean:
                nm["not_trade"] += 1
            elif prob < THRESHOLD:
                nm["below_thresh"] += 1
            elif USE_PRICES and not has_edge:
                nm["no_edge"] += 1
            elif not reliable_ok:
                nm["pre_peak"] += 1
            else:
                nm["high_ok"] += 1

        if VERBOSE_LOG and prob >= 0.50:
            if not clean:
                why = f"verdict={verdict}"
            elif prob < THRESHOLD:
                why = f"prob {prob*100:.0f}%<{THRESHOLD*100:.0f}%"
            elif USE_PRICES and not has_edge:
                why = "no edge (market efficient)"
            elif not reliable_ok:
                why = "not reliable (pre-peak)"
            elif prev_alerted:
                why = "already alerted"
            else:
                why = "WOULD ALERT"
            tq = (p.get("timing") or {}).get("quality", "?")
            edge = ""
            bt = p.get("best_trade")
            if bt:
                edge = f" edge{bt['best_edge']*100:+.0f}%"
            print(f"    {city:<14} {bucket}° {prob*100:>3.0f}% {tq:<11} {why}{edge}")

        mode_tag = " [observe]" if OBSERVE_ONLY else ""
        psym = p.get("temp_unit", "°")
        cdisp = city_display(p["city"])
        def _log(line):
            if not OBSERVE_ONLY:           # only log alerts actually sent
                try: learn.log_alert_line(line, p["city"])
                except Exception: pass
        if crossed_up:
            alert_signal(fmt_new_signal(p))   # suppressed in OBSERVE mode
            fire_webhook(p, "new_signal")     # forward to your other bot (if set)
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            new_alerts += 1
            bt = p.get("best_trade")
            buy = (f" · {bt['action']} {bt['temp']}{psym}@{bt['yes_price']*100:.0f}¢ "
                   f"{bt['best_edge']*100:+.0f}%") if bt else ""
            _log(f"🟢 {cdisp} {bucket}{psym}@{prob*100:.0f}%{buy}")
            # Record the alert for the missed-trade tracker (LIVE only — in OBSERVE
            # no alert was actually sent, so "you missed it" wouldn't be true).
            if not OBSERVE_ONLY and p.get("best_trade"):
                held = p["city"] in held_cities
                try:
                    learn.note_alert(p, held)
                except Exception:
                    pass
                if not held:
                    print(f"     ↳ recorded as potential missed trade (no position held)")
            print(f"  🟢 ALERT {city} {bucket}° {prob*100:.0f}%{mode_tag}")
        elif bucket_shifted:
            alert_signal(fmt_bucket_shift(p, prev_bucket, prev_prob if prev_prob is not None else prob,
                                          position=held_positions.get(p["city"])))
            fire_webhook(p, "bucket_shift")
            # keep alerted_high=1 since it's still a high-conf signal, just a new bucket
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            shifts += 1
            _log(f"🔄 {cdisp} {prev_bucket}{psym}→{bucket}{psym} @{prob*100:.0f}%")
            print(f"  🔄 SHIFT {city} {prev_bucket}°→{bucket}° {prob*100:.0f}%{mode_tag}")
        elif collapsed:
            alert_signal(fmt_collapse(p, prev_prob if prev_prob is not None else prob))
            fire_webhook(p, "collapse")
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=0)
            collapses += 1
            _log(f"🔴 {cdisp} {bucket}{psym} weakened to {prob*100:.0f}%")
            print(f"  🔴 COLLAPSE {city} {prob*100:.0f}%{mode_tag}")
        else:
            # update stored prob without alerting
            if prev:
                upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=prev_alerted)

        # ── ENDGAME scanner (separate feature) — only when the main signal did NOT
        # fire for this city, so it catches near-closed markets the main logic
        # skips (market-decided / sub-threshold small edges). Deduped per pick. ──
        if ENABLE_ENDGAME and not crossed_up and not bucket_shifted:
            try:
                opp = check_endgame(p)
            except Exception:
                opp = None
            if opp:
                ekey  = f"endgame|{p['city']}|{tdate}"     # once per ending market
                eprev = get_state(conn, ekey)
                if not (eprev and eprev["alerted_high"]):
                    alert_signal(fmt_endgame(opp))      # suppressed in OBSERVE
                    fire_webhook(p, "endgame")
                    upsert_state(conn, ekey, p["city"], tdate, opp["dom"], opp["dom_price"], alerted_high=1)
                    _log(f"🔚 {cdisp} ending · front {opp['dom']}{psym}@{opp['dom_price']*100:.0f}¢ · bot {opp['pick']}{psym}")
                    print(f"  🔚 ENDGAME {city} front {opp['dom']}°@{opp['dom_price']*100:.0f}¢ "
                          f"bot {opp['pick']}°{mode_tag}")

    # Learning: snapshot every prediction from this scan (1 disk write). The last
    # snapshot taken before each city's local day ends is what gets scored later.
    try:
        learn.note_many(scan_preds)
    except Exception as e:
        print(f"  [learn] note error: {e}")

    print(f"[{ts}] done — {new_alerts} new, {shifts} shifted, {collapses} collapsed")
    # data health: how many cities we could actually evaluate this pass.
    print(f"           data: {health['evaluated']}/{health['total']} evaluated, "
          f"{health['no_market']} no-market, {health['muted']} muted, "
          f"{health['no_data']} no-data, {health['errored']} errored")
    # if most cities had no usable data, the run is unreliable — flag it loudly so
    # a "0 new" line isn't mistaken for a calm market when it's really an outage.
    lost = health["no_data"] + health["errored"]
    if lost >= max(1, health["total"] // 2):
        print(f"           ⚠️  WARNING: {lost}/{health['total']} cities had no usable "
              f"data — upstream weather API likely degraded/rate-limited; signals unreliable")
    # always show WHY nothing alerted (even without full verbose)
    if new_alerts == 0 and (nm["no_edge"] or nm["pre_peak"] or nm["below_thresh"] or nm["not_trade"]):
        print(f"           ({nm['no_edge']} no-edge, {nm['pre_peak']} pre-peak, "
              f"{nm['below_thresh']} below-{THRESHOLD*100:.0f}%, {nm['not_trade']} not-clean, "
              f"{nm['high_ok']} ready)")
    return new_alerts, collapses


def send_morning_digest():
    """Once-a-day 'here's what's worth a look today' — the best edges across all
    cities in one message, so you don't have to wait for live alerts to trickle in.
    Suppressed in OBSERVE mode (it's a trade prompt)."""
    if OBSERVE_ONLY:
        return
    print("[digest] building morning digest…")
    try:
        pw.prefetch_metars([(pw.CITIES.get(c) or {}).get("icao") for c in ALERT_CITIES])
    except Exception as e:
        print(f"[digest] metar prefetch failed: {e}")
    hits = []
    for city in ALERT_CITIES:
        if city in MUTED or (pw.resolve_city(city) or city) in MUTED:
            continue
        try:
            p = pw.predict(city, fetch_prices=USE_PRICES)
        except Exception:
            continue
        if "error" in p:
            continue
        if p.get("best_trade") and p.get("verdict") == "TRADE":
            hits.append(p)
    if not hits:
        print("[digest] nothing tradeable — skipping")
        return
    hits.sort(key=lambda x: (x.get("best_trade") or {}).get("best_edge", 0), reverse=True)
    L = [f"🌅 <b>MORNING DIGEST — {datetime.now(timezone.utc).strftime('%b %d')}</b>",
         f"Top edges across {len(ALERT_CITIES)} cities right now:", ""]
    for p in hits[:6]:
        bt = p.get("best_trade"); sym = p.get("temp_unit", "°")
        tq = (p.get("timing") or {}).get("quality", "?")
        L.append(f"• <b>{esc(city_display(p['city']))}</b> {esc(bt['action'])} {bt['temp']}{sym} "
                 f"@ {bt['yes_price']*100:.0f}¢ → edge {bt['best_edge']*100:+.0f}% "
                 f"(model {bt['model_prob']*100:.0f}%, {esc(tq)})")
    L.append("")
    L.append("👉 /scan &lt;city&gt; for the full card with stake sizing.")
    send_telegram("\n".join(L))
    print(f"[digest] sent — {len(hits)} tradeable")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND LISTENER — /scan on demand
# Polls getUpdates in a background thread so you can trigger scans from the app.
# ══════════════════════════════════════════════════════════════════════════════
REGION_CITIES = {
    "asia":   ["tokyo", "hong kong", "singapore", "manila", "seoul", "taipei",
               "shanghai", "jakarta", "bangkok", "mumbai", "lucknow", "busan"],
    "europe": ["london", "paris", "milan", "madrid", "munich", "warsaw",
               "amsterdam", "helsinki", "istanbul"],
    "americas": ["new york", "toronto", "sao paulo", "buenos aires", "mexico city"],
}

def scan_for_command(scope="all", reply_to=None):
    """Run a scan on demand. Logs full per-city detail to Railway, and for a
    single named city ALWAYS sends full detail (even if not a clean TRADE)."""
    single_city = False
    if scope == "all":
        cities = ALERT_CITIES
        header = f"🔍 <b>Scan: ALL {len(cities)} markets</b>"
    elif scope in REGION_CITIES:
        cities = [c for c in REGION_CITIES[scope] if c in pw.CITIES]
        header = f"🔍 <b>Scan: {scope.upper()} ({len(cities)} cities)</b>"
    else:
        resolved = pw.resolve_city(scope) if hasattr(pw, "resolve_city") else scope
        cities = [resolved] if resolved in pw.CITIES else []
        header = f"🔍 <b>Scan: {city_display(resolved)}</b>"
        single_city = True

    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"\n[{ts}] /scan {scope} → {len(cities)} city(ies)")

    if not cities:
        print(f"  ❓ unknown market '{scope}'")
        reply_telegram(reply_to, f"❓ Unknown market '{scope}'. Try /scan or /scan europe.")
        return

    reply_telegram(reply_to, f"{header}\n⏳ Scanning…")

    # batch-prefetch airport METARs for these cities (one call, not one-per-city)
    try:
        pw.prefetch_metars([(pw.CITIES.get(c) or {}).get("icao") for c in cities])
    except Exception as e:
        print(f"  [metar] prefetch failed: {e}")

    hits = []
    single_pred = None
    for city in cities:
        try:
            p = pw.predict(city, fetch_prices=USE_PRICES)
        except Exception as e:
            print(f"  {city}: error {e}")
            continue
        if "error" in p:
            print(f"  {city}: {p['error']}")
            continue

        if single_city:
            single_pred = p

        prob    = p.get("top_prob", 0)
        bucket  = p.get("top_bucket")
        verdict = p.get("verdict")
        tq      = (p.get("timing") or {}).get("quality", "?")
        bt      = p.get("best_trade")
        edge_s  = f" edge{bt['best_edge']*100:+.0f}%" if bt else " no-edge"

        # FULL per-city log line (this is what shows in Railway)
        print(f"  {city:<14} {bucket}° {prob*100:>3.0f}% {verdict:<5} {tq:<11}{edge_s}")

        # for a single-city scan, log the FULL detail card to Railway too
        if single_city:
            detail = fmt_new_signal(p)
            import re
            clean = re.sub(r"<[^>]+>", "", detail)
            print("  ┌─ full detail ─────────────────")
            for ln in clean.splitlines():
                print(f"  │ {ln}")
            print("  └───────────────────────────────")

        if verdict == "TRADE" and prob >= THRESHOLD and bt:
            hits.append(p)

    # ── Telegram replies ──
    if single_city:
        if single_pred:
            reply_telegram(reply_to, fmt_new_signal(single_pred))
            # offer a one-tap forward when there's an actual trade to send
            if single_pred.get("best_trade"):
                _forward_prompt(reply_to, [single_pred["city"]])
        else:
            reply_telegram(reply_to, f"{header}\n\n❌ Could not fetch data.")
        print(f"[{ts}] /scan {scope} done")
        return

    if not hits:
        reply_telegram(reply_to, f"{header}\n\n😴 No clean tradeable signals right now "
                                 f"(TRADE + ≥{THRESHOLD*100:.0f}% + real edge).")
        print(f"[{ts}] /scan {scope} done — 0 tradeable")
        return

    hits.sort(key=lambda x: x.get("top_prob", 0), reverse=True)
    reply_telegram(reply_to, f"{header}\n\n✅ {len(hits)} tradeable signal(s):")
    for p in hits[:10]:
        reply_telegram(reply_to, fmt_new_signal(p))
    # offer to forward the found signals to the external bot (manual option)
    _forward_prompt(reply_to, [p["city"] for p in hits[:8]])
    print(f"[{ts}] /scan {scope} done — {len(hits)} tradeable")


def _tg_authorized(chat_id) -> bool:
    """Only act on Telegram commands from configured recipients. If no chat IDs
    are set (Telegram unconfigured), fall through so local testing still works."""
    if not TG_CHAT_IDS:
        return True
    return str(chat_id) in TG_CHAT_IDS


# ── Telegram button menu ──────────────────────────────────────────────────────
def _main_menu_keyboard():
    """Inline keyboard mirroring every command. Each button sends callback_data
    'cmd:<command>' which is routed straight back through handle_command, so the
    buttons and typed commands share one code path."""
    return [
        [{"text": "🔍 Scan all",      "callback_data": "cmd:/scan"},
         {"text": "🌍 Pick region",   "callback_data": "menu:regions"}],
        [{"text": "💼 Positions",     "callback_data": "cmd:/positions"},
         {"text": "💰 P&L ledger",    "callback_data": "cmd:/pnl"}],
        [{"text": "📊 Learn",         "callback_data": "cmd:/learn"},
         {"text": "🎯 Calibration",   "callback_data": "cmd:/learn calib"}],
        [{"text": "🏙️ Best cities",   "callback_data": "cmd:/learn cities"},
         {"text": "📡 Sources",       "callback_data": "cmd:/learn sources"}],
        [{"text": "🧪 No-bias check", "callback_data": "cmd:/learn nobias"},
         {"text": "💸 Missed",        "callback_data": "cmd:/missed"}],
        [{"text": "🔚 Ending markets","callback_data": "cmd:/endgame"},
         {"text": "🧵 Today's alerts","callback_data": "cmd:/alerts"}],
        [{"text": "🔭 Price watches", "callback_data": "cmd:/watches"},
         {"text": "🔕 Muted",         "callback_data": "cmd:/muted"}],
        [{"text": "💾 Backup",        "callback_data": "cmd:/backup"},
         {"text": "❓ Help",          "callback_data": "cmd:/help"}],
    ]


def _regions_keyboard():
    return [
        [{"text": "🌏 Asia",     "callback_data": "cmd:/scan asia"},
         {"text": "🌍 Europe",   "callback_data": "cmd:/scan europe"}],
        [{"text": "🌎 Americas", "callback_data": "cmd:/scan americas"}],
        [{"text": "⬅️ Back to menu", "callback_data": "menu:main"}],
    ]


def _forward_prompt(reply_to, cities):
    """After a manual scan, offer one-tap buttons to forward each signal to the
    external bot. No-op unless a webhook is configured."""
    cities = [c for c in (cities or []) if c]
    if not WEBHOOK_URL or not cities or not reply_to:
        return
    rows = [[{"text": f"📡 Send {city_display(c)}", "callback_data": f"send:{c}"}]
            for c in cities[:8]]
    reply_telegram(reply_to, "📡 <b>Forward to your other bot?</b>", keyboard=rows)


def _manual_send(ck, reply_to):
    """Forward one city's CURRENT signal to the webhook synchronously, reporting
    success/failure back to the user. Used by the /send command and Send buttons."""
    if not WEBHOOK_URL:
        reply_telegram(reply_to, "📡 Webhook not configured (set WEBHOOK_URL).")
        return
    ck = pw.resolve_city(ck) or ck
    if ck not in pw.CITIES:
        reply_telegram(reply_to, f"❓ Unknown city '{esc(ck)}'.")
        return
    reply_telegram(reply_to, f"📡 Sending {city_display(ck)} to your bot…")
    ok = False
    try:
        pp = pw.predict(ck, fetch_prices=True)
        ok = send_webhook(build_signal_payload(pp, "manual_signal"))
    except Exception as e:
        print(f"[send] {ck}: {e}")
    reply_telegram(reply_to, f"✅ Sent {city_display(ck)} (event=manual_signal)." if ok
                   else "❌ Send failed — run /webhook to test the connection.")


def set_bot_commands():
    """Register the command list so Telegram shows the blue 'Menu' button next to
    the input box and '/' autocomplete. Typed commands keep working regardless."""
    if not TG_TOKEN:
        return
    cmds = [
        {"command": "menu",      "description": "📋 Button menu of all actions"},
        {"command": "scan",      "description": "🔍 Scan all markets (or /scan <city>)"},
        {"command": "endgame",   "description": "🔚 Ending markets with a small edge"},
        {"command": "positions", "description": "💼 Your positions + P&L"},
        {"command": "pnl",       "description": "💰 Realized P&L ledger"},
        {"command": "learn",     "description": "📊 Scoreboard (calib/sources/cities/nobias)"},
        {"command": "missed",    "description": "💸 What-if P&L on alerts you skipped"},
        {"command": "history",   "description": "📜 A city's history (/history <city>)"},
        {"command": "alerts",    "description": "🧵 Alert thread (today or /alerts <city>)"},
        {"command": "watch",     "description": "🔔 Price alert (/watch london 14 below 50)"},
        {"command": "watches",   "description": "🔭 List your price watches"},
        {"command": "unwatch",   "description": "🗑️ Remove a price watch (/unwatch N)"},
        {"command": "mute",      "description": "🔕 Silence a city (/mute <city>)"},
        {"command": "unmute",    "description": "🔔 Unmute a city (/unmute <city>)"},
        {"command": "muted",     "description": "📋 List muted cities"},
        {"command": "send",      "description": "📡 Forward a city's signal to your bot (/send london)"},
        {"command": "webhook",   "description": "📡 Test the outgoing webhook"},
        {"command": "backup",    "description": "💾 Back up learning data"},
        {"command": "help",      "description": "❓ Command list"},
    ]
    try:
        httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/setMyCommands",
                   json={"commands": cmds}, timeout=15.0)
        httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/setChatMenuButton",
                   json={"menu_button": {"type": "commands"}}, timeout=15.0)
        print("[listener] registered Telegram command menu (setMyCommands)")
    except Exception as e:
        print(f"[listener] setMyCommands failed: {e}")


# ── Price-watch helpers ───────────────────────────────────────────────────────
def _city_sym(ck) -> str:
    return "°F" if (pw.CITIES.get(ck or "") or {}).get("f") else "°C"

def _parse_price01(s):
    """'50' / '50¢' / '0.5' → 0.50 (a 0–1 probability/price). None if invalid."""
    try:
        v = float(str(s).lower().replace("¢", "").replace("c", "").strip())
    except Exception:
        return None
    if v > 1:
        v = v / 100.0
    return round(v, 3) if 0 < v < 1 else None

def _parse_watch(text):
    """Parse '/watch <city> <bucket> [yes|no] <below|above> <price>'.
    Returns ((city, bucket, side, dir, price01), None) or (None, error_msg)."""
    toks = text.split()[1:]
    usage = ("Usage: <code>/watch &lt;city&gt; &lt;bucket&gt; [yes|no] "
             "&lt;below|above&gt; &lt;price&gt;</code>\ne.g. <code>/watch london 14 below 50</code>")
    if not toks:
        return None, usage
    # greedily match the city (handles 'new york', 'hong kong')
    city, rest = None, toks
    for n in (3, 2, 1):
        if len(toks) >= n:
            ck = pw.resolve_city(" ".join(toks[:n]))
            if ck:
                city, rest = ck, toks[n:]
                break
    if not city:
        return None, "Unknown city. Try e.g. london, tokyo, new york."
    # bucket = first integer token
    bucket = None
    for i, t in enumerate(rest):
        if t.lstrip("-").isdigit():
            bucket = int(t); rest = rest[:i] + rest[i+1:]; break
    if bucket is None:
        return None, "Give a bucket temperature (a number), e.g. 14.\n" + usage
    side, direction, price = "yes", None, None
    for t in rest:
        tl = t.lower()
        if tl in ("yes", "no"):
            side = tl
        elif tl in ("below", "under", "<", "<=", "≤"):
            direction = "below"
        elif tl in ("above", "over", ">", ">=", "≥"):
            direction = "above"
        else:
            v = _parse_price01(t)
            if v is not None:
                price = v
    if direction is None:
        return None, "Say <b>below</b> or <b>above</b>, e.g. /watch london 14 below 50"
    if price is None:
        return None, "Give a target price in cents, e.g. <b>50</b> (= 50¢)."
    return (city, bucket, side, direction, price), None

def _watch_price(w):
    """Freshest available price (0–1) for a watch's side."""
    side, tok = w.get("side", "yes"), w.get("token_yes")
    if side == "yes" and tok:
        p = pw.fetch_clob_price(tok, "buy", cache_ttl=60)   # live order book
        if p is not None:
            return p
    pm = pw.fetch_polymarket_market(w["city"], w.get("target_date"), cache_ttl=60)
    b  = ((pm or {}).get("buckets") or {}).get(w["bucket"]) or {}
    if side == "no":
        if b.get("no") is not None:
            return b["no"]
        return round(1.0 - b["yes"], 3) if b.get("yes") is not None else None
    return b.get("yes")

def _market_day_past(city_key, date_str) -> bool:
    try:
        tz = (pw.CITIES.get(city_key) or {}).get("tz", 0)
        local_now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=tz)
        return bool(date_str) and date_str < local_now.strftime("%Y-%m-%d")
    except Exception:
        return False

def check_price_watches():
    """Fire any price watch whose target is met; expire ones whose market settled.
    One-shot: a fired/expired watch is removed."""
    if not PRICE_WATCHES:
        return
    keep, changed = [], False
    for w in PRICE_WATCHES:
        ck, sym = w.get("city"), _city_sym(w.get("city"))
        price = None
        try:
            price = _watch_price(w)
        except Exception as e:
            print(f"  [watch] price fetch error {ck} {w.get('bucket')}: {e}")
        if price is None:
            if _market_day_past(ck, w.get("target_date")):
                try:
                    reply_telegram(w.get("chat_id"),
                        f"⌛ Price watch on {city_display(ck)} {w['bucket']}{sym} expired — "
                        f"that market has settled.")
                except Exception:
                    pass
                changed = True
            else:
                keep.append(w)
            continue
        hit = (price <= w["price"]) if w["dir"] == "below" else (price >= w["price"])
        if hit:
            arrow = "≤" if w["dir"] == "below" else "≥"
            try:
                reply_telegram(w.get("chat_id"),
                    f"🔔 <b>PRICE ALERT</b>\n"
                    f"{city_display(ck)} <b>{w['bucket']}{sym} {w['side'].upper()}</b> is now "
                    f"<b>{price*100:.0f}¢</b> ({arrow} your {w['price']*100:.0f}¢ target).\n"
                    f"👉 Buy now if you still want it — this watch is now cleared.")
            except Exception:
                pass
            print(f"  🔔 PRICE WATCH hit {ck} {w['bucket']}{sym} {w['side']} {price*100:.0f}¢")
            changed = True
        else:
            keep.append(w)
    if changed:
        PRICE_WATCHES[:] = keep
        save_watches(PRICE_WATCHES)


def _answer_callback(cq_id):
    """Acknowledge a button tap so Telegram stops the little loading spinner."""
    if not (TG_TOKEN and cq_id):
        return
    try:
        httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/answerCallbackQuery",
                   json={"callback_query_id": cq_id}, timeout=10.0)
    except Exception:
        pass


def handle_command(text, chat_id):
    """Parse a /command and act."""
    text = (text or "").strip()
    low = text.lower()

    if low in ("/menu", "/start"):
        reply_telegram(chat_id,
            "📋 <b>PolyWeather menu</b>\n"
            "Tap a button below, or type any command.\n"
            "For one city, type: <code>/scan tokyo</code> · <code>/history tokyo</code> · "
            "<code>/mute tokyo</code>",
            keyboard=_main_menu_keyboard())
        return

    if low == "/help":
        reply_telegram(chat_id,
            "🤖 <b>PolyWeather commands</b>\n\n"
            "/menu — button menu of everything\n"
            "/scan — scan ALL markets now\n"
            "/scan europe — scan a region (asia/europe/americas)\n"
            "/scan munich — scan one city\n"
            "/endgame — ending markets (nearly decided) with a small edge left\n"
            "/positions — show your positions now\n"
            "/pnl — realized P&L ledger from settled alerts\n"
            "/learn — prediction-vs-outcome scoreboard (also: all / calib / sources / cities / nobias)\n"
            "/missed — $1 what-if P&L on alerts you didn't take\n"
            "/history <city> — that city's prediction history + suggested bias\n"
            "/alerts [city|date] — alert thread: a city's alerts, or a day's\n"
            "/watch <city> <bucket> below|above <price> — price alert "
            "(e.g. /watch london 14 below 50); /watches, /unwatch &lt;n&gt;\n"
            "/mute <city> · /unmute <city> · /muted — silence a city (keeps learning)\n"
            "/send <city> — forward that city's signal to your other bot now\n"
            "/webhook — test the outgoing API webhook to your other bot\n"
            "/backup — back up learning data to GitHub\n"
            "/help — this message")
        return

    if low.startswith("/learn"):
        parts = text.split()
        arg = parts[1].lower() if len(parts) > 1 else ""
        if arg == "all":
            reply_telegram(chat_id, learn.report(all_time=True))
        elif arg in ("calib", "calibration"):
            reply_telegram(chat_id, learn.report_calibration())
        elif arg in ("sources", "source", "apis", "api"):
            # optional city after it, e.g. "/learn sources london"
            city = next((pw.resolve_city(x) for x in parts[2:] if pw.resolve_city(x)), None)
            reply_telegram(chat_id, learn.report_sources(city))
        elif arg in ("nobias", "no-bias", "biascheck"):
            reply_telegram(chat_id, learn.report_bias_free())
        elif arg in ("cities", "city", "best"):
            reply_telegram(chat_id, learn.report_cities())
        else:
            # optional explicit date as second arg, else most recent settled day
            date = next((x for x in parts[1:] if x.count("-") == 2), None)
            reply_telegram(chat_id, learn.report(date))
        return

    if low == "/positions":
        if WALLET:
            # show ALL positions (weather + everything else you hold)
            positions = pw.fetch_positions(WALLET, weather_only=False)
            reply_telegram(chat_id, fmt_positions_update(WALLET, positions))
        else:
            reply_telegram(chat_id, "No wallet set (POLYMARKET_WALLET).")
        return

    if low.startswith("/missed"):
        reply_telegram(chat_id, learn.report_missed())
        return

    if low.startswith("/pnl") or low.startswith("/ledger"):
        reply_telegram(chat_id, learn.report_pnl())
        return

    if low.startswith("/endgame") or low.startswith("/ending"):
        reply_telegram(chat_id, "🔚 Scanning for ending markets (small-edge closing plays)…")
        try:
            pw.prefetch_metars([(pw.CITIES.get(c) or {}).get("icao") for c in ALERT_CITIES])
        except Exception:
            pass
        found = []
        for city in ALERT_CITIES:
            try:
                pp = pw.predict(city, fetch_prices=USE_PRICES)
            except Exception:
                continue
            if "error" in pp:
                continue
            opp = check_endgame(pp)
            if opp:
                found.append(opp)
        if not found:
            reply_telegram(chat_id, "🔚 No ending markets with an edge right now.")
            return
        found.sort(key=lambda o: o["edge"], reverse=True)
        reply_telegram(chat_id, f"🔚 <b>{len(found)} ending market(s) with an edge:</b>")
        for o in found[:10]:
            reply_telegram(chat_id, fmt_endgame(o))
        return

    if low.startswith("/webhook"):
        if not WEBHOOK_URL:
            reply_telegram(chat_id, "📡 Webhook not configured. Set <b>WEBHOOK_URL</b> "
                           "(and optional <b>WEBHOOK_TOKEN</b>) in Railway.")
            return
        reply_telegram(chat_id, f"📡 Sending a test payload to {esc(WEBHOOK_URL[:50])}…")
        ok = send_webhook({"event": "test", "source": "polyweather",
                           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "message": "PolyWeather webhook test — your endpoint is reachable."})
        events = ", ".join(sorted(WEBHOOK_EVENTS)) or "none"
        reply_telegram(chat_id, (f"✅ Webhook OK (2xx). Forwarding events: {events}."
                                 if ok else
                                 "❌ Webhook failed — check the URL / token / target logs."))
        return

    if low.startswith("/send"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            reply_telegram(chat_id, "Usage: <code>/send &lt;city&gt;</code> — forward that "
                           "city's current signal to your other bot.\ne.g. /send london")
            return
        _manual_send(parts[1].strip(), chat_id)
        return

    if low == "/muted" or low.startswith("/mute") or low.startswith("/unmute"):
        parts = text.split(maxsplit=1)
        if low == "/muted":
            reply_telegram(chat_id, "🔕 Muted cities: "
                           + (", ".join(city_display(c) for c in sorted(MUTED)) if MUTED else "none"))
            return
        if len(parts) < 2:
            reply_telegram(chat_id, "Usage: /mute &lt;city&gt;  ·  /unmute &lt;city&gt;  ·  /muted")
            return
        ck = pw.resolve_city(parts[1].strip()) or parts[1].strip().lower()
        if low.startswith("/unmute"):
            MUTED.discard(ck); save_muted(MUTED)
            reply_telegram(chat_id, f"🔔 Unmuted {city_display(ck)} — alerts back on.")
        else:
            MUTED.add(ck); save_muted(MUTED)
            reply_telegram(chat_id, f"🔕 Muted {city_display(ck)} — still learning, but no alerts.")
        return

    # ── custom price watches ── (/watches and /unwatch BEFORE /watch: prefix order)
    if low.startswith("/watches") or low == "/watchlist":
        if not PRICE_WATCHES:
            reply_telegram(chat_id, "🔭 No price watches set.\nAdd one: "
                           "<code>/watch london 14 below 50</code>")
            return
        L = ["🔭 <b>Price watches</b>"]
        for i, w in enumerate(PRICE_WATCHES, 1):
            arrow = "≤" if w["dir"] == "below" else "≥"
            L.append(f"{i}. {city_display(w['city'])} {w['bucket']}{_city_sym(w['city'])} "
                     f"{w['side'].upper()} {arrow} {w['price']*100:.0f}¢  ({w.get('target_date','')})")
        L.append("\nRemove with /unwatch &lt;number&gt;")
        reply_telegram(chat_id, "\n".join(L))
        return

    if low.startswith("/unwatch"):
        parts = text.split()
        if len(parts) < 2 or not parts[1].isdigit():
            reply_telegram(chat_id, "Usage: /unwatch &lt;number&gt;  (see /watches)")
            return
        idx = int(parts[1]) - 1
        if 0 <= idx < len(PRICE_WATCHES):
            w = PRICE_WATCHES.pop(idx); save_watches(PRICE_WATCHES)
            reply_telegram(chat_id, f"🗑️ Removed watch on {city_display(w['city'])} "
                           f"{w['bucket']}{_city_sym(w['city'])} {w['side'].upper()}.")
        else:
            reply_telegram(chat_id, "No watch with that number. See /watches.")
        return

    if low.startswith("/watch"):
        parsed, err = _parse_watch(text)
        if err:
            reply_telegram(chat_id, "❓ " + err)
            return
        city, bucket, side, direction, price = parsed
        # resolve the live market once: confirm the bucket, capture date + YES token
        try:
            p = pw.predict(city, fetch_prices=True)
        except Exception:
            p = {}
        buckets = ((p.get("polymarket") or {}).get("buckets")) or {}
        sym = p.get("temp_unit", _city_sym(city))
        b = buckets.get(bucket)
        if not b:
            avail = ", ".join(str(k) for k in sorted(buckets)) or "none"
            reply_telegram(chat_id, f"❓ No {bucket}{sym} bucket in {city_display(city)}'s "
                           f"market right now. Available: {avail}")
            return
        if side == "yes":
            cur = b.get("yes")
        else:
            cur = b.get("no") if b.get("no") is not None else (
                round(1 - b["yes"], 3) if b.get("yes") is not None else None)
        w = {"city": city, "bucket": bucket, "side": side, "dir": direction,
             "price": price, "target_date": p.get("target_date"),
             "token_yes": b.get("token_yes"), "chat_id": chat_id,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        PRICE_WATCHES.append(w); save_watches(PRICE_WATCHES)
        arrow = "≤" if direction == "below" else "≥"
        curs = f"{cur*100:.0f}¢" if cur is not None else "—"
        reply_telegram(chat_id,
            f"✅ Watching <b>{city_display(city)} {bucket}{sym} {side.upper()}</b>\n"
            f"Alert when {arrow} <b>{price*100:.0f}¢</b> (now {curs}). "
            f"Checking every {PRICE_WATCH_MIN} min · /watches to view.")
        return

    if low.startswith("/history"):
        parts = text.split(maxsplit=1)
        if len(parts) > 1:
            reply_telegram(chat_id, learn.report_city(parts[1].strip()))
        else:
            reply_telegram(chat_id, "Usage: /history <city>   e.g. /history manila")
        return

    if low.startswith("/alerts") or low.startswith("/today"):
        parts = text.split()
        date = next((x for x in parts[1:] if x.count("-") == 2), None)
        city = next((x for x in parts[1:] if x.count("-") != 2 and pw.resolve_city(x)), None)
        reply_telegram(chat_id, learn.report_alerts(date, city))
        return

    if low == "/backup":
        if GITHUB_TOKEN and GITHUB_REPO:
            reply_telegram(chat_id, "💾 Backing up learning data to GitHub…")
            n = github_backup()
            if n > 0:
                save_backup_day(datetime.now(timezone.utc).date())
                reply_telegram(chat_id, f"💾 Done — pushed {n} file(s) → {GITHUB_REPO}@{GITHUB_BACKUP_BRANCH}")
            else:
                reply_telegram(chat_id, "⚠️ Backup pushed 0 files — check the logs "
                                        "(token scope / repo / branch).")
        else:
            reply_telegram(chat_id, "Backup not configured (set GITHUB_TOKEN + GITHUB_REPO).")
        return

    if low == "/pick":
        kb = [
            [{"text": "🌍 All markets", "callback_data": "scan:all"}],
            [{"text": "🌏 Asia", "callback_data": "scan:asia"},
             {"text": "🌍 Europe", "callback_data": "scan:europe"}],
            [{"text": "🌎 Americas", "callback_data": "scan:americas"}],
        ]
        reply_telegram(chat_id, "Pick what to scan:", keyboard=kb)
        return

    if low.startswith("/scan"):
        parts = text.split(maxsplit=1)
        scope = parts[1].strip().lower() if len(parts) > 1 else "all"
        # "/scan positions" / "/scan pnl" etc. — the user meant that command, not a
        # market named "positions". Route it to the real handler instead of erroring.
        _CMD_WORDS = {"positions", "pnl", "ledger", "learn", "missed", "history",
                      "alerts", "today", "backup", "pick", "help", "start", "menu",
                      "mute", "muted", "unmute", "watch", "watches", "unwatch",
                      "watchlist", "webhook", "send", "endgame", "ending"}
        first = scope.split()[0] if scope else ""
        if first in _CMD_WORDS:
            handle_command("/" + scope, chat_id)
            return
        scan_for_command(scope, reply_to=chat_id)
        return

    # unknown
    reply_telegram(chat_id, "❓ Unknown command. Send /help for options.")


def command_listener():
    """Background thread: long-poll Telegram for /commands."""
    if not TG_TOKEN:
        return
    # getUpdates is blocked if a webhook is set OR another instance is polling.
    # Clearing any stale webhook fixes the most common "/scan does nothing" case.
    try:
        wi = httpx.get(f"https://api.telegram.org/bot{TG_TOKEN}/getWebhookInfo", timeout=15.0)
        url = (wi.json().get("result") or {}).get("url") if wi.status_code == 200 else None
        if url:
            httpx.get(f"https://api.telegram.org/bot{TG_TOKEN}/deleteWebhook", timeout=15.0)
            print(f"[listener] cleared a webhook ({url}) that was blocking commands")
    except Exception as e:
        print(f"[listener] webhook check failed: {e}")
    set_bot_commands()          # populate the blue Menu button + / autocomplete
    offset = None
    consec_errors = 0
    print("[listener] Telegram command listener started — /menu for buttons, or type commands")
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            r = httpx.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                          params=params, timeout=40.0)
            if r.status_code != 200:
                # 409 = a webhook is set or ANOTHER instance is polling. Surface it
                # so the cause is visible instead of commands silently failing.
                body = r.text[:160].replace("\n", " ")
                if r.status_code == 429:
                    # Honor Telegram's own retry_after — retrying sooner just earns
                    # another 429 (and a tight retry loop can cause them).
                    try:
                        wait = max(5, int((r.json().get("parameters") or {}).get("retry_after", 5)))
                    except Exception:
                        wait = 5
                else:
                    consec_errors += 1
                    wait = min(5 + consec_errors * 3, 30)   # back off on 5xx/Bad Gateway
                print(f"[listener] getUpdates {r.status_code} (retry in {wait}s): {body}")
                if r.status_code == 409:
                    print("[listener] ⚠️ CONFLICT — another bot instance is running "
                          "(old Railway deploy or a local run). Stop the duplicate so "
                          "/scan, /learn, /positions work.")
                time.sleep(wait); continue
            consec_errors = 0          # healthy poll — reset the backoff
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                # text message commands
                msg = upd.get("message") or {}
                text = msg.get("text", "")
                chat_id = (msg.get("chat") or {}).get("id")
                if text and chat_id and text.startswith("/"):
                    if _tg_authorized(chat_id):
                        handle_command(text, chat_id)
                    else:
                        print(f"[listener] ignored command from unauthorized chat {chat_id}")
                # inline button taps
                cq = upd.get("callback_query")
                if cq:
                    data    = cq.get("data", "")
                    cq_chat = (cq.get("message") or {}).get("chat", {}).get("id")
                    _answer_callback(cq.get("id"))     # stop the button spinner
                    if cq_chat and _tg_authorized(cq_chat):
                        if data.startswith("cmd:"):
                            handle_command(data[4:], cq_chat)        # any menu button
                        elif data == "menu:main":
                            reply_telegram(cq_chat, "📋 <b>Menu</b> — tap or type a command:",
                                           keyboard=_main_menu_keyboard())
                        elif data == "menu:regions":
                            reply_telegram(cq_chat, "🌍 Pick a region to scan:",
                                           keyboard=_regions_keyboard())
                        elif data.startswith("send:"):                # forward to bot
                            _manual_send(data.split(":", 1)[1], cq_chat)
                        elif data.startswith("scan:"):                # legacy /pick
                            scan_for_command(data.split(":", 1)[1], reply_to=cq_chat)
        except Exception as e:
            # network read timeouts / connection blips — back off so a Telegram
            # outage doesn't turn into a tight retry loop (which itself draws 429s).
            consec_errors += 1
            wait = min(5 + consec_errors * 3, 30)
            print(f"[listener] error: {e} (retry in {wait}s)")
            time.sleep(wait)


# ── Nightly GitHub backup of the learning data ────────────────────────────────
def _gh_headers():
    return {"Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json"}

def _gh_ensure_branch() -> bool:
    """Make sure the backup branch exists; create it from the default branch."""
    api = f"https://api.github.com/repos/{GITHUB_REPO}"
    r = httpx.get(f"{api}/branches/{GITHUB_BACKUP_BRANCH}", headers=_gh_headers(), timeout=20.0)
    if r.status_code == 200:
        return True
    repo = httpx.get(api, headers=_gh_headers(), timeout=20.0).json()
    default = repo.get("default_branch", "main")
    ref = httpx.get(f"{api}/git/ref/heads/{default}", headers=_gh_headers(), timeout=20.0).json()
    sha = (ref.get("object") or {}).get("sha")
    if not sha:
        return False
    cr = httpx.post(f"{api}/git/refs", headers=_gh_headers(),
                    json={"ref": f"refs/heads/{GITHUB_BACKUP_BRANCH}", "sha": sha}, timeout=20.0)
    return cr.status_code in (200, 201)

def _gh_put_file(repo_path: str, local_path: str, message: str) -> bool:
    """Create/update one file on the backup branch via the Contents API."""
    if not os.path.exists(local_path):
        return False
    import base64
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}"
    g = httpx.get(api, headers=_gh_headers(), params={"ref": GITHUB_BACKUP_BRANCH}, timeout=20.0)
    sha = g.json().get("sha") if g.status_code == 200 else None
    payload = {"message": message, "content": b64, "branch": GITHUB_BACKUP_BRANCH}
    if sha:
        payload["sha"] = sha
    r = httpx.put(api, headers=_gh_headers(), json=payload, timeout=30.0)
    if r.status_code not in (200, 201):
        print(f"[backup] {repo_path}: {r.status_code} {r.text[:120]}")
        return False
    return True

def github_backup() -> int:
    """Push the learning files to GitHub (separate branch). Never raises.
    Returns the number of files successfully pushed (0 = nothing/failed)."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return 0
    try:
        if not _gh_ensure_branch():
            print("[backup] could not ensure backup branch — check token/repo")
            return 0
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        files = [(learn._LEARN_FILE, "data/learn_history.json"),
                 (pw._HISTORY_FILE,  "data/deb_history.json"),
                 (learn.ALERTS_FILE, "data/alerts_log.json"),
                 (pw._OBSMAX_FILE,   "data/obs_max.json")]
        n = sum(1 for local, repo_path in files
                if _gh_put_file(repo_path, local, f"learning backup {ts}"))
        print(f"[backup] pushed {n} file(s) to {GITHUB_REPO}@{GITHUB_BACKUP_BRANCH}")
        return n
    except Exception as e:
        print(f"[backup] error: {e}")
        return 0


def _gh_get_file(repo_path: str):
    """Fetch a file's raw bytes from the backup branch, or None."""
    import base64
    api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}"
    r = httpx.get(api, headers=_gh_headers(), params={"ref": GITHUB_BACKUP_BRANCH}, timeout=20.0)
    if r.status_code != 200:
        return None
    content = r.json().get("content")
    if not content:
        return None
    try:
        return base64.b64decode(content)
    except Exception:
        return None

def _local_is_empty(local: str) -> bool:
    """True if a learning file is missing or holds no real data (so it's safe to
    restore over it without losing anything)."""
    if not os.path.exists(local):
        return True
    try:
        with open(local) as f:
            return not json.load(f)        # {} / [] / empty → empty
    except Exception:
        return True

def github_restore():
    """On startup, if a local learning file is missing/empty, restore it from the
    GitHub backup branch. NEVER overwrites a file that already has data."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
    try:
        files = [(learn._LEARN_FILE, "data/learn_history.json"),
                 (pw._HISTORY_FILE,  "data/deb_history.json"),
                 (learn.ALERTS_FILE, "data/alerts_log.json"),
                 (pw._OBSMAX_FILE,   "data/obs_max.json")]
        restored = 0
        for local, repo_path in files:
            if not _local_is_empty(local):
                continue                   # local already has data — keep it
            data = _gh_get_file(repo_path)
            if data:
                os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
                with open(local, "wb") as f:
                    f.write(data)
                restored += 1
                print(f"[restore] recovered {repo_path} ← {GITHUB_BACKUP_BRANCH}")
        if restored:
            print(f"[restore] restored {restored} learning file(s) from GitHub backup")
    except Exception as e:
        print(f"[restore] error: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PolyWeather Telegram monitor")
    parser.add_argument("--once", action="store_true", help="Run a single scan then exit")
    parser.add_argument("--test", action="store_true", help="Send a test Telegram message and exit")
    parser.add_argument("--positions-now", action="store_true", help="Send one position update and exit")
    args = parser.parse_args()

    if args.test:
        ok = send_telegram("✅ <b>PolyWeather monitor</b> test message — Telegram is working!")
        print("Test sent." if ok else "Test FAILED — check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return

    if args.positions_now:
        if not WALLET:
            print("No POLYMARKET_WALLET set.")
            return
        send_position_update()
        return

    print("="*60)
    print("  PolyWeather Monitor starting")
    print(f"  Trade mode:        {TRADE_MODE}"
          + ("  (OBSERVE — learning only, NO buy alerts)" if OBSERVE_ONLY else "  (LIVE — sending alerts)"))
    print(f"  Signal threshold:  {THRESHOLD*100:.0f}%")
    print(f"  Scan interval:     {INTERVAL_MIN} min")
    print(f"  Position updates:  every {POS_UPDATE_MIN} min" if WALLET else "  Position updates:  OFF (no wallet)")
    if WALLET:
        print(f"  Position WATCH:    every {POS_WATCH_MIN} min (flips + stop-loss)")
        print(f"  Profit alert:      at +{PROFIT_TAKE_PCT:.0f}% per position")
    print(f"  Bankroll (sizing): ${BANKROLL:.0f}  (¼-Kelly stakes in alerts)")
    if getattr(pw, "USE_NOBIAS", False):
        print(f"  Bias mode:         NO-BIAS (trading on the raw model blend)")
    print(f"  Endgame alerts:    {'AUTO on' if ENABLE_ENDGAME else 'off'} (/endgame works on demand)")
    if WEBHOOK_URL:
        print(f"  Webhook → {WEBHOOK_URL[:48]}  events={','.join(sorted(WEBHOOK_EVENTS))}")
    if ENABLE_DIGEST and not OBSERVE_ONLY:
        print(f"  Morning digest:    daily at {DIGEST_HOUR_UTC:02d}:00 UTC")
    if MUTED:
        print(f"  Muted cities:      {', '.join(sorted(MUTED))}")
    print(f"  Cities:            {len(ALERT_CITIES)}")
    print(f"  Prices:            {'on' if USE_PRICES else 'off'}")
    print(f"  State DB:          {STATE_DB}")
    print(f"  Telegram:          {len(TG_CHAT_IDS)} recipient(s)" if (TG_TOKEN and TG_CHAT_IDS) else "  Telegram:          NOT configured")
    print("="*60)

    conn = init_db()

    # If the /data volume was wiped (fresh deploy), pull the learning history back
    # from the GitHub backup before anything reads or overwrites it.
    github_restore()

    # startup ping
    if OBSERVE_ONLY:
        startup = (
            f"🔭 <b>PolyWeather monitor online — OBSERVE mode</b>\n"
            f"Watching {len(ALERT_CITIES)} cities every {INTERVAL_MIN} min, "
            f"<b>learning only — no buy alerts</b>.\n"
            f"Gathering accuracy data; check /learn and /learn calib."
        )
    else:
        startup = (
            f"🌡️ <b>PolyWeather monitor online — LIVE mode</b>\n"
            f"Watching {len(ALERT_CITIES)} cities every {INTERVAL_MIN} min.\n"
            f"Alerts when a city crosses {THRESHOLD*100:.0f}% (and when it collapses)."
        )
    if WALLET:
        startup += f"\n💼 Position updates every {POS_UPDATE_MIN} min."
    startup += "\n\n💬 Commands: /scan  /positions  /pnl  /learn  /learn cities  /mute  /help"
    send_telegram(startup)

    # record any outstanding actual highs so the model can learn its bias
    try:
        backfill_actuals()
    except Exception as e:
        print(f"[learn] backfill error: {e}")

    if args.once:
        run_scan(conn)
        if WALLET:
            send_position_update()
        return

    # start the Telegram command listener in a background thread
    threading.Thread(target=command_listener, daemon=True).start()

    # independent timers
    last_scan = 0.0
    last_pos  = 0.0
    # send an immediate position update on boot
    if WALLET:
        send_position_update()
        last_pos = time.time()

    last_watch = 0.0
    last_sigwatch = 0.0
    last_pricewatch = 0.0
    last_backfill = time.time()   # already ran once above, just before the loop
    BACKFILL_SEC  = int(os.environ.get("BACKFILL_HOURS", "6")) * 3600
    # learning digest: settle completed days + send a Telegram scoreboard once/day
    last_learn_report = time.time()
    LEARN_REPORT_SEC  = int(os.environ.get("LEARN_REPORT_HOURS", "24")) * 3600
    last_backup_day   = load_backup_day()   # persisted across redeploys
    last_backup_try   = 0.0                  # throttle retries on failure
    last_digest_day   = None   # morning digest runs once per UTC day
    if GITHUB_TOKEN and GITHUB_REPO:
        print(f"  Backup last ran: {last_backup_day or 'never'} "
              f"(nightly at/after {BACKUP_HOUR_UTC:02d}:00 UTC)")
    while True:
        now = time.time()

        # nightly off-Railway backup of the learning files to GitHub.
        # Fires any time AT/AFTER the backup hour on a day not yet backed up — so a
        # restart that misses the exact hour still catches up. Retries every 30 min
        # on failure; stamps to disk only on success so it survives redeploys.
        nowdt = datetime.now(timezone.utc)
        if (GITHUB_TOKEN and GITHUB_REPO and nowdt.hour >= BACKUP_HOUR_UTC
                and last_backup_day != nowdt.date()
                and now - last_backup_try >= 1800):
            last_backup_try = now
            try:
                if github_backup() > 0:
                    last_backup_day = nowdt.date()
                    save_backup_day(last_backup_day)
            except Exception as e:
                print(f"[loop] backup error: {e}")

        # once-a-day morning digest of the best edges across all cities. Fires in a
        # 3-hour window from DIGEST_HOUR_UTC (not an exact-hour match) so a restart
        # that misses the precise hour still sends it that morning.
        if (ENABLE_DIGEST and not OBSERVE_ONLY
                and DIGEST_HOUR_UTC <= nowdt.hour < DIGEST_HOUR_UTC + 3
                and last_digest_day != nowdt.date()):
            last_digest_day = nowdt.date()
            try:
                send_morning_digest()
            except Exception as e:
                print(f"[loop] digest error: {e}")

        # signal scan timer
        if now - last_scan >= INTERVAL_MIN * 60:
            try:
                run_scan(conn)
            except Exception as e:
                print(f"[loop] scan error: {e}")
            last_scan = time.time()

        # learning: periodically record completed-day actual highs + settle the
        # prediction-vs-outcome tracker (scores which side won for each city)
        if now - last_backfill >= BACKFILL_SEC:
            try:
                backfill_actuals()
            except Exception as e:
                print(f"[loop] backfill error: {e}")
            try:
                learn.settle()
            except Exception as e:
                print(f"[loop] learn settle error: {e}")
            last_backfill = time.time()

        # daily learning scoreboard to Telegram: "we predicted X — it WON/LOST"
        if now - last_learn_report >= LEARN_REPORT_SEC:
            try:
                rep = learn.settle_and_report()
                if rep and "No settled" not in rep and "No learning" not in rep:
                    send_telegram(rep)
            except Exception as e:
                print(f"[loop] learn report error: {e}")
            # also push the bias-vs-no-bias scoreboard so you can watch which is
            # more accurate over time (the thing we're monitoring from today)
            try:
                nb = learn.report_bias_free()
                if nb and "No settled" not in nb:
                    send_telegram(nb)
            except Exception as e:
                print(f"[loop] nobias report error: {e}")
            last_learn_report = time.time()

        # tight position WATCH timer (alerts on model flipping against held bucket)
        if WALLET and (now - last_watch >= POS_WATCH_MIN * 60):
            try:
                watch_positions(conn)
            except Exception as e:
                print(f"[loop] watch error: {e}")
            last_watch = time.time()

        # FAST watch on active signals — catch drops without waiting 20 min
        if now - last_sigwatch >= SIGNAL_WATCH_MIN * 60:
            try:
                watch_active_signals(conn)
            except Exception as e:
                print(f"[loop] signal watch error: {e}")
            last_sigwatch = time.time()

        # custom price watches — fire when a bucket hits the user's target price
        if PRICE_WATCHES and (now - last_pricewatch >= PRICE_WATCH_MIN * 60):
            try:
                check_price_watches()
            except Exception as e:
                print(f"[loop] price watch error: {e}")
            last_pricewatch = time.time()

        # periodic full position P&L update
        if WALLET and (now - last_pos >= POS_UPDATE_MIN * 60):
            try:
                send_position_update()
            except Exception as e:
                print(f"[loop] position error: {e}")
            last_pos = time.time()

        # sleep until the next due event (check every 30s)
        time.sleep(30)


if __name__ == "__main__":
    main()
