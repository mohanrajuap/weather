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
THRESHOLD     = float(os.environ.get("PROB_THRESHOLD", "0.70"))
USE_PRICES    = os.environ.get("USE_PRICES", "1") == "1"
STATE_DB      = os.environ.get("STATE_DB", "/data/monitor_state.db")
_alert_cities = os.environ.get("ALERT_CITIES", "").strip()
ALERT_CITIES  = [c.strip() for c in _alert_cities.split(",") if c.strip()] or list(pw.CITIES.keys())

# fall back to local path if /data doesn't exist (local testing)
if not os.path.isdir(os.path.dirname(STATE_DB) or "."):
    STATE_DB = "monitor_state.db"


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
def fmt_new_signal(p) -> str:
    """Full city detail — mirrors what --detail --prices shows."""
    sym  = p["temp_unit"]
    tim  = p.get("timing") or {}
    ens  = p.get("ensemble") or {}
    live = p.get("live") or {}
    bt   = p.get("best_trade")
    edges = p.get("edges") or []

    lines = [
        f"🟢 <b>NEW SIGNAL — {p['city'].upper()}</b>",
        f"📅 Market date: <b>{p['target_date']}</b> ({p.get('predicting','')})",
        f"🕐 Timing: <b>{tim.get('quality','?')}</b> (local {tim.get('city_local_now','?')}, peak {tim.get('peak_window','?')})",
        ("✅ RELIABLE — peak observed, high essentially locked" if tim.get("reliable")
         else "🔮 FORECAST ONLY — tomorrow's peak ~a day away, can shift a lot" if tim.get("quality") == "FORECAST"
         else "⏳ NOT YET RELIABLE — peak still forming, signal can shift"),
        "",
        f"🎯 Predicted: <b>{p['top_bucket']}{sym}</b> at <b>{p['top_prob']*100:.0f}%</b>",
        f"🧬 DEB blend: {p.get('deb')}{sym}  (μ={p.get('mu')}, σ={p.get('sigma')})",
    ]

    # model forecasts
    if p.get("forecasts"):
        fc = "  ".join(f"{m}:{v}" for m, v in p["forecasts"].items())
        lines.append(f"📊 Models: {fc}")
    # ensemble
    if ens.get("p10") is not None:
        lines.append(f"📉 Ensemble: P10={ens['p10']} Med={ens['median']} P90={ens['p90']}")
    # agreement
    agr = p.get("agreement")
    if agr:
        ae = {"strong":"✅","moderate":"⚠️","weak":"❌"}.get(agr,"")
        lines.append(f"{ae} Agreement: {agr} (differ {p.get('disagreement')}°)")
    # live obs
    if live.get("current_temp") is not None:
        lines.append(f"🌡️ Live: {live['current_temp']}{sym} (max {live.get('max_so_far')}{sym}, {live.get('trend')})")

    # probability distribution (top 3)
    dist = p.get("distribution") or []
    if dist:
        lines.append("")
        lines.append("🎲 <b>Probabilities:</b>")
        for b in dist[:4]:
            lines.append(f"   {b['value']}{sym}: {b['probability']*100:.0f}%")

    # live polymarket edge table
    if edges:
        lines.append("")
        lines.append("💰 <b>Polymarket edge:</b>")
        shown = 0
        for e in edges:
            if e["action"] in ("BUY YES", "BUY NO") and shown < 4:
                ys = f"{e['yes_price']*100:.0f}¢" if e.get("yes_price") is not None else "—"
                lines.append(f"   {e['action']} {e['temp']}{sym} @ {ys} → {e['best_edge']*100:+.0f}% (model {e['model_prob']*100:.0f}%)")
                shown += 1
        if shown == 0:
            lines.append("   (no >10% edge — market efficient)")

    # best trade headline
    if bt:
        lines.append("")
        lines.append(f"🏆 <b>BEST: {bt['action']} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢ → {bt['best_edge']*100:+.0f}% edge</b>")
        if bt.get("thin"):
            lines.append(f"⚠️ Low volume (${bt.get('vol',0):,.0f}) — keep size tiny")

    pm = p.get("polymarket")
    if pm and pm.get("url"):
        lines.append(f'🔗 {pm["url"]}')
    return "\n".join(lines)


def fmt_collapse(p, prev_prob) -> str:
    sym       = p["temp_unit"]
    new_prob  = p.get("top_prob", 0)
    verdict   = p.get("verdict")
    reasons   = p.get("verdict_reasons") or []

    # Work out the REAL reason it collapsed (not the TRADE 'clear signal' text).
    if verdict != "TRADE":
        # it failed a quality check — use that reason (it's a genuine warning)
        bad = [r for r in reasons if "clear signal" not in r.lower()]
        reason = bad[0] if bad else "signal no longer clean"
        headline = "model no longer confident"
    else:
        # still a TRADE verdict, but probability fell below YOUR alert threshold
        reason = (f"probability fell below your {THRESHOLD*100:.0f}% alert bar "
                  f"(still a {new_prob*100:.0f}% lean, but less certain)")
        headline = "confidence easing"

    lines = [
        f"🔴 <b>SIGNAL WEAKENED — {p['city'].upper()}</b>",
        f"📅 Market date: {p['target_date']}",
        f"📉 {p['top_bucket']}{sym}: {prev_prob*100:.0f}% → <b>{new_prob*100:.0f}%</b>  ({headline})",
        f"⚠️ {reason}",
    ]
    # if a conflict/stale flag is present, surface it explicitly
    for r in reasons:
        rl = r.lower()
        if ("conflict" in rl or "stale" in rl or "boundary" in rl or "disagree" in rl) and r not in reason:
            lines.append(f"🚨 {r}")
            break
    lines.append(f"👉 If you hold {p['top_bucket']}{sym}, reconsider — edge is shrinking.")
    return "\n".join(lines)


def fmt_bucket_shift(p, prev_bucket, prev_prob) -> str:
    """The model's top bucket CHANGED while still high-confidence.
    e.g. morning said 32°C@70%, now says 28°C@70%. This is a different prediction."""
    sym = p["temp_unit"]
    new_bucket = p["top_bucket"]
    new_prob   = p["top_prob"]
    direction = "📈 higher" if new_bucket > prev_bucket else "📉 lower"
    lines = [
        f"🔄 <b>PREDICTION CHANGED — {p['city'].upper()}</b>",
        f"📅 Market date: {p['target_date']}",
        "",
        f"⚠️ The model's prediction MOVED ({direction}):",
        f"   BEFORE: <b>{prev_bucket}{sym}</b> at {prev_prob*100:.0f}%",
        f"   NOW:    <b>{new_bucket}{sym}</b> at {new_prob*100:.0f}%",
        "",
        f"👉 This is a DIFFERENT bucket than before.",
        f"   If you bought {prev_bucket}{sym}, the model no longer favors it!",
    ]
    # show where the old bucket sits now
    dist = {b["value"]: b["probability"] for b in p.get("distribution", [])}
    old_now = dist.get(prev_bucket)
    if old_now is not None:
        lines.append(f"   Your old {prev_bucket}{sym} is now only {old_now*100:.0f}%.")
    # edge on the new bucket
    bt = p.get("best_trade")
    if bt:
        lines.append("")
        lines.append(f"💰 New best: {bt['action']} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢ → {bt['best_edge']*100:+.0f}%")
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
        if city and bucket is not None:
            try:
                pred = pw.predict(city, fetch_prices=False)
                sym  = pred.get("temp_unit", "°")
                dist = {b["value"]: b["probability"] for b in pred.get("distribution", [])}
                your_prob = dist.get(bucket, 0.0)
                top_b   = pred.get("top_bucket")
                top_p   = pred.get("top_prob", 0)
                verdict = pred.get("verdict")
                # model's current view of YOUR bucket
                if your_prob >= 0.55:
                    icon = "🟢"
                elif your_prob >= 0.30:
                    icon = "🟡"
                else:
                    icon = "🔴"
                lines.append(f"   {icon} Model now: your {bucket}{sym}={your_prob*100:.0f}% "
                             f"| top pick {top_b}{sym}={top_p*100:.0f}%")
                # warn if model has moved away from your bucket
                if your_prob < 0.30 and top_b != bucket:
                    lines.append(f"   ⚠️ Model favors {top_b}{sym} now — your bucket weakening")
                # conflict/stale flags
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
        return False
    try:
        positions = pw.fetch_positions(WALLET, weather_only=True)
        msg = fmt_positions_update(WALLET, positions)
        send_telegram(msg)
        print(f"  💼 sent position update ({len(positions or [])} positions)")
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

        # state key for the watch (per position bucket)
        wkey = f"watch|{city}|{pred.get('target_date')}|{bucket}"
        prev = get_state(conn, wkey)
        prev_prob = prev["prob"] if prev else your_prob

        # conditions to alert
        flipped_away = (top_b is not None and top_b != bucket and your_prob < 0.40)
        big_drop     = (prev_prob - your_prob) >= 0.15   # dropped 15+ pts since last watch
        conflict     = pred.get("live_model_conflict")

        # only alert once per worsening state — track with alerted_high flag
        already = prev["alerted_high"] if prev else 0

        if (flipped_away or big_drop) and not already:
            lines = [
                f"⚡ <b>POSITION WATCH — {city.upper()}</b>",
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
            send_telegram(fmt_bucket_shift(p, prev_bucket, prev_prob))
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=1)
            print(f"  ⚡🔄 FAST SHIFT {city} {prev_bucket}°→{bucket}°")
        elif collapsed:
            send_telegram(fmt_collapse(p, prev_prob if prev_prob is not None else prob))
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=0)
            print(f"  ⚡🔴 FAST COLLAPSE {city} {prob*100:.0f}%")
        else:
            # still healthy — just refresh stored prob
            upsert_state(conn, key, city, tdate, bucket, prob, alerted_high=1)


# ── One monitoring pass ───────────────────────────────────────────────────────
def run_scan(conn):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[{ts}] scanning {len(ALERT_CITIES)} cities (threshold {THRESHOLD*100:.0f}%)...")

    new_alerts = 0
    collapses  = 0
    shifts     = 0

    for city in ALERT_CITIES:
        try:
            p = pw.predict(city, fetch_prices=USE_PRICES)
        except Exception as e:
            print(f"  {city}: error {e}")
            continue
        if "error" in p:
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

        if crossed_up:
            msg = fmt_new_signal(p)
            send_telegram(msg)
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            new_alerts += 1
            print(f"  🟢 ALERT {city} {bucket}° {prob*100:.0f}%")
        elif bucket_shifted:
            msg = fmt_bucket_shift(p, prev_bucket, prev_prob if prev_prob is not None else prob)
            send_telegram(msg)
            # keep alerted_high=1 since it's still a high-conf signal, just a new bucket
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            shifts += 1
            print(f"  🔄 SHIFT {city} {prev_bucket}°→{bucket}° {prob*100:.0f}%")
        elif collapsed:
            msg = fmt_collapse(p, prev_prob if prev_prob is not None else prob)
            send_telegram(msg)
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=0)
            collapses += 1
            print(f"  🔴 COLLAPSE {city} {prob*100:.0f}%")
        else:
            # update stored prob without alerting
            if prev:
                upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=prev_alerted)

    print(f"[{ts}] done — {new_alerts} new, {shifts} shifted, {collapses} collapsed")
    return new_alerts, collapses


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
    print("="*60)

    conn = init_db()

    # startup ping
    startup = (
        f"🌡️ <b>PolyWeather monitor online</b>\n"
        f"Watching {len(ALERT_CITIES)} cities every {INTERVAL_MIN} min.\n"
        f"Alerts when a city crosses {THRESHOLD*100:.0f}% (and when it collapses)."
    )
    if WALLET:
        startup += f"\n💼 Position updates every {POS_UPDATE_MIN} min."
    send_telegram(startup)

    if args.once:
        run_scan(conn)
        if WALLET:
            send_position_update()
        return

    # independent timers
    last_scan = 0.0
    last_pos  = 0.0
    # send an immediate position update on boot
    if WALLET:
        send_position_update()
        last_pos = time.time()

    last_watch = 0.0
    last_sigwatch = 0.0
    while True:
        now = time.time()

        # signal scan timer
        if now - last_scan >= INTERVAL_MIN * 60:
            try:
                run_scan(conn)
            except Exception as e:
                print(f"[loop] scan error: {e}")
            last_scan = time.time()

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
