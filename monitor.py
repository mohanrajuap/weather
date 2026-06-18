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
from datetime import datetime, timezone, timedelta

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

# ── Config from environment ───────────────────────────────────────────────────
TG_TOKEN      = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT       = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
# Support multiple recipients: comma-separated in TELEGRAM_CHAT_ID
TG_CHAT_IDS   = [c.strip() for c in TG_CHAT.split(",") if c.strip()]
# ntfy.sh push notifications — extra layer alongside Telegram (works in India).
# Set NTFY_TOPIC to a secret topic name; subscribe to it in the ntfy phone app.
NTFY_TOPIC    = os.environ.get("NTFY_TOPIC", "").strip()
NTFY_SERVER   = os.environ.get("NTFY_SERVER", "https://ntfy.sh").strip().rstrip("/")
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
THRESHOLD     = float(os.environ.get("PROB_THRESHOLD", "0.70"))
USE_PRICES    = os.environ.get("USE_PRICES", "1") == "1"
STATE_DB      = os.environ.get("STATE_DB", "/data/monitor_state.db")
_alert_cities = os.environ.get("ALERT_CITIES", "").strip()
ALERT_CITIES  = [c.strip() for c in _alert_cities.split(",") if c.strip()] or list(pw.CITIES.keys())

# fall back to local path if /data doesn't exist (local testing)
if not os.path.isdir(os.path.dirname(STATE_DB) or "."):
    STATE_DB = "monitor_state.db"


# ── Telegram ──────────────────────────────────────────────────────────────────
def _strip_html(text: str) -> str:
    """ntfy is plain-text — remove HTML tags Telegram uses."""
    import re
    t = re.sub(r"<a href=\"([^\"]*)\">([^<]*)</a>", r"\2: \1", text)  # keep links readable
    t = re.sub(r"<[^>]+>", "", t)
    return t


def send_ntfy(text: str) -> bool:
    """Push to ntfy.sh (extra layer alongside Telegram; works in India)."""
    if not NTFY_TOPIC:
        return False
    body = _strip_html(text)
    # title = first line, rest = body
    lines = body.splitlines()
    title = lines[0][:100] if lines else "PolyWeather"
    rest  = "\n".join(lines[1:]).strip() or title
    # ntfy sends the title as an HTTP header. Stripping emoji with ascii-ignore
    # can leave leading/trailing spaces (e.g. "🌡️ PolyWeather" → " PolyWeather"),
    # and header values may not start with whitespace — that raises "Illegal
    # header value". Sanitize: drop non-ascii, collapse whitespace, fall back.
    safe_title = " ".join(title.encode("ascii", "ignore").decode().split()) or "PolyWeather"
    try:
        r = httpx.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data=rest.encode("utf-8"),
            headers={
                "Title": safe_title,
                "Tags": "chart_with_upwards_trend",
                "Priority": "default",
            },
            timeout=15.0,
        )
        if r.status_code not in (200, 201):
            print(f"[ntfy] error: {r.status_code} {r.text[:120]}")
            return False
        return True
    except Exception as e:
        print(f"[ntfy] exception: {e}")
        return False


def send_telegram(text: str) -> bool:
    # extra layer: always also push to ntfy (no-op if NTFY_TOPIC unset)
    send_ntfy(text)

    if not TG_TOKEN or not TG_CHAT_IDS:
        if not NTFY_TOPIC:
            print(f"[telegram] not configured — would send:\n{text}\n")
        return bool(NTFY_TOPIC)
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


def reply_telegram(chat_id, text: str, keyboard=None) -> bool:
    """Reply to a single chat (used by the command listener)."""
    # extra layer: also push command results to ntfy (skip the button menus)
    if not keyboard:
        send_ntfy(text)
    if not TG_TOKEN:
        return bool(NTFY_TOPIC) and not keyboard
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
    L.append(f"🎯 <b>{p['top_bucket']}{sym}</b>  at  <b>{p['top_prob']*100:.0f}%</b>")
    L.append(f"🕐 {esc(tim.get('quality','?'))} · local {esc(tim.get('city_local_now','?'))} · peak {esc(tim.get('peak_window','?'))}")
    L.append(badge)
    L.append("")

    # model section
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
        src = "🎯 Wunderground" if live.get("source") == "wunderground" else "METAR"
        L.append(f"🌡️ Live: {live['current_temp']}{sym} (max {live.get('max_so_far')}{sym}, {esc(live.get('trend'))}) · {src}")

    # probabilities
    dist = p.get("distribution") or []
    if dist:
        L.append("")
        L.append("🎲 <b>Probabilities</b>")
        for b in dist[:4]:
            bar = "▰" * max(1, round(b['probability'] * 10))
            L.append(f"   {b['value']}{sym}  {bar} {b['probability']*100:.0f}%")

    # best trade — only call it a BUY when action_ok; else show as "if it holds"
    if bt:
        L.append("")
        L.append(_DIV)
        if action_ok:
            L.append(f"🏆 <b>{esc(bt['action'])} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢</b>")
            L.append(f"    edge <b>{bt['best_edge']*100:+.0f}%</b> · model {bt['model_prob']*100:.0f}%")
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
    return "\n".join(L)


def fmt_bucket_shift(p, prev_bucket, prev_prob) -> str:
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
    pm = p.get("polymarket")
    if pm and pm.get("url"):
        lines.append(f'🔗 {pm["url"]}')
    return "\n".join(lines)


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

def fmt_positions_update(wallet: str, positions) -> str:
    if positions is None:
        return "⚠️ Could not fetch your positions (check wallet / network)."
    if not positions:
        return "💼 <b>Positions update</b>\nNo open weather positions right now."

    lines = ["💼 <b>YOUR POSITIONS</b> (with live model read)\n"]
    total_now  = 0.0
    total_paid = 0.0
    total_pnl  = 0.0
    claimable  = []

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
        lines.append(f"{emoji} <b>{short}</b>")
        lines.append(f"   {side} | {e_s}→{n_s} | ${val:.2f} (P&L {pnl:+.2f})")
        if pos.get("redeemable"):
            lines.append(f"   ✅ SETTLED — claimable")

        # ── live model read for THIS position's bucket ──
        city   = _match_city_from_title(title)
        bucket = _extract_pos_bucket(title)
        cur_price = pos.get("cur_price")   # market's current price for your side
        if city and bucket is not None:
            try:
                pred = pw.predict(city, fetch_prices=False)
                sym  = pred.get("temp_unit", "°")
                dist = {b["value"]: b["probability"] for b in pred.get("distribution", [])}
                your_prob = dist.get(bucket, 0.0)
                top_b   = pred.get("top_bucket")
                top_p   = pred.get("top_prob", 0)
                tim     = pred.get("timing") or {}
                peak_passed = tim.get("quality") == "RELIABLE"  # post-peak, high locked

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
        lines.append("")

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
            positions = [p for p in all_pos
                         if ("temperature" in (p.get("title") or "").lower()
                             or "temp" in (p.get("event_slug") or "").lower()
                             or "weather" in (p.get("event_slug") or "").lower())]
            print(f"  💼 positions for {masked}: {len(all_pos)} total, "
                  f"{len(positions)} weather")
            if all_pos and not positions:
                # Show what IS there so the user can see why it was filtered out.
                sample = ", ".join((p.get("title") or "?")[:40] for p in all_pos[:3])
                print(f"     ↳ non-weather positions held: {sample}")
                # Don't hide the user's money — show all positions if none are weather.
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

        # Suppress the alert entirely if the market says you're winning, or the
        # peak already passed (the model can't "un-happen" a locked-in high).
        if market_winning or peak_passed:
            upsert_state(conn, wkey, city, pred.get("target_date"), bucket, your_prob, alerted_high=0)
            continue

        # only alert once per worsening state — track with alerted_high flag
        already = prev["alerted_high"] if prev else 0

        if (flipped_away or big_drop) and not already:
            lines = [
                f"⚡ <b>POSITION WATCH — {esc(city_display(city))}</b>",
                f"📅 {pred.get('target_date')}  |  you hold <b>{bucket}{sym}</b>",
                "",
            ]
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
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=1)
            print(f"  ⚡🔄 FAST SHIFT {city} {prev_bucket}°→{bucket}°")
        elif collapsed:
            alert_signal(fmt_collapse(p, prev_prob if prev_prob is not None else prob))
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

    new_alerts = 0
    collapses  = 0
    shifts     = 0
    # near-miss tally: cities that were close but filtered
    nm = {"no_edge": 0, "pre_peak": 0, "below_thresh": 0, "not_trade": 0, "high_ok": 0}
    # data-health tally: how many cities we could actually evaluate vs lost to
    # errors / missing forecast data. Without this, a total upstream outage looks
    # identical to a calm "0 new" market in the logs.
    health = {"total": len(ALERT_CITIES), "errored": 0, "no_data": 0, "evaluated": 0}
    scan_preds = []   # collected for the learning tracker (one disk write at end)

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
        if crossed_up:
            alert_signal(fmt_new_signal(p))   # suppressed in OBSERVE mode
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            new_alerts += 1
            print(f"  🟢 ALERT {city} {bucket}° {prob*100:.0f}%{mode_tag}")
        elif bucket_shifted:
            alert_signal(fmt_bucket_shift(p, prev_bucket, prev_prob if prev_prob is not None else prob))
            # keep alerted_high=1 since it's still a high-conf signal, just a new bucket
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            shifts += 1
            print(f"  🔄 SHIFT {city} {prev_bucket}°→{bucket}° {prob*100:.0f}%{mode_tag}")
        elif collapsed:
            alert_signal(fmt_collapse(p, prev_prob if prev_prob is not None else prob))
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=0)
            collapses += 1
            print(f"  🔴 COLLAPSE {city} {prob*100:.0f}%{mode_tag}")
        else:
            # update stored prob without alerting
            if prev:
                upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=prev_alerted)

    # Learning: snapshot every prediction from this scan (1 disk write). The last
    # snapshot taken before each city's local day ends is what gets scored later.
    try:
        learn.note_many(scan_preds)
    except Exception as e:
        print(f"  [learn] note error: {e}")

    print(f"[{ts}] done — {new_alerts} new, {shifts} shifted, {collapses} collapsed")
    # data health: how many cities we could actually evaluate this pass.
    print(f"           data: {health['evaluated']}/{health['total']} evaluated, "
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
    print(f"[{ts}] /scan {scope} done — {len(hits)} tradeable")


def handle_command(text, chat_id):
    """Parse a /command and act."""
    text = (text or "").strip()
    low = text.lower()

    if low in ("/start", "/help"):
        reply_telegram(chat_id,
            "🤖 <b>PolyWeather commands</b>\n\n"
            "/scan — scan ALL markets now\n"
            "/scan europe — scan a region (asia/europe/americas)\n"
            "/scan munich — scan one city\n"
            "/pick — choose a market with buttons\n"
            "/positions — show your positions now\n"
            "/learn — prediction-vs-outcome scoreboard (add 'all' for lifetime)\n"
            "/help — this message")
        return

    if low.startswith("/learn"):
        parts = text.split()
        arg = parts[1].lower() if len(parts) > 1 else ""
        if arg == "all":
            reply_telegram(chat_id, learn.report(all_time=True))
        elif arg in ("calib", "calibration"):
            reply_telegram(chat_id, learn.report_calibration())
        else:
            # optional explicit date as second arg, else most recent settled day
            date = next((x for x in parts[1:] if x.count("-") == 2), None)
            reply_telegram(chat_id, learn.report(date))
        return

    if low == "/positions":
        if WALLET:
            positions = pw.fetch_positions(WALLET, weather_only=True)
            reply_telegram(chat_id, fmt_positions_update(WALLET, positions))
        else:
            reply_telegram(chat_id, "No wallet set (POLYMARKET_WALLET).")
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
        scan_for_command(scope, reply_to=chat_id)
        return

    # unknown
    reply_telegram(chat_id, "❓ Unknown command. Send /help for options.")


def command_listener():
    """Background thread: long-poll Telegram for /commands."""
    if not TG_TOKEN:
        return
    offset = None
    print("[listener] Telegram command listener started (/scan, /positions, /pick, /help)")
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            r = httpx.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                          params=params, timeout=40.0)
            if r.status_code != 200:
                time.sleep(5); continue
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                # text message commands
                msg = upd.get("message") or {}
                text = msg.get("text", "")
                chat_id = (msg.get("chat") or {}).get("id")
                if text and chat_id and text.startswith("/"):
                    handle_command(text, chat_id)
                # inline button taps
                cq = upd.get("callback_query")
                if cq:
                    data = cq.get("data", "")
                    cq_chat = (cq.get("message") or {}).get("chat", {}).get("id")
                    if data.startswith("scan:") and cq_chat:
                        scan_for_command(data.split(":", 1)[1], reply_to=cq_chat)
        except Exception as e:
            print(f"[listener] error: {e}")
            time.sleep(5)


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
        print(f"  Position WATCH:    every {POS_WATCH_MIN} min (alerts on flips)")
        print(f"  Profit alert:      at +{PROFIT_TAKE_PCT:.0f}% per position")
    print(f"  Cities:            {len(ALERT_CITIES)}")
    print(f"  Prices:            {'on' if USE_PRICES else 'off'}")
    print(f"  State DB:          {STATE_DB}")
    print(f"  Telegram:          {len(TG_CHAT_IDS)} recipient(s)" if (TG_TOKEN and TG_CHAT_IDS) else "  Telegram:          NOT configured")
    print(f"  ntfy push:         {NTFY_SERVER}/{NTFY_TOPIC}" if NTFY_TOPIC else "  ntfy push:         NOT configured")
    print("="*60)

    conn = init_db()

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
    startup += "\n\n💬 Commands: /scan  /positions  /learn  /learn calib  /help"
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
    import threading
    listener = threading.Thread(target=command_listener, daemon=True)
    listener.start()

    # independent timers
    last_scan = 0.0
    last_pos  = 0.0
    # send an immediate position update on boot
    if WALLET:
        send_position_update()
        last_pos = time.time()

    last_watch = 0.0
    last_sigwatch = 0.0
    last_backfill = time.time()   # already ran once above, just before the loop
    BACKFILL_SEC  = int(os.environ.get("BACKFILL_HOURS", "6")) * 3600
    # learning digest: settle completed days + send a Telegram scoreboard once/day
    last_learn_report = time.time()
    LEARN_REPORT_SEC  = int(os.environ.get("LEARN_REPORT_HOURS", "24")) * 3600
    while True:
        now = time.time()

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
