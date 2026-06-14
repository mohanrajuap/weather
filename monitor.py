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
WALLET        = os.environ.get("POLYMARKET_WALLET", "").strip()
INTERVAL_MIN  = int(os.environ.get("CHECK_INTERVAL_MIN", "20"))
POS_UPDATE_MIN = int(os.environ.get("POSITION_UPDATE_MIN", "15"))
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
    if not TG_TOKEN or not TG_CHAT:
        print(f"[telegram] not configured — would send:\n{text}\n")
        return False
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=15.0,
        )
        if r.status_code != 200:
            print(f"[telegram] error {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[telegram] exception: {e}")
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
def fmt_new_signal(p) -> str:
    sym  = p["temp_unit"]
    tim  = p.get("timing") or {}
    bt   = p.get("best_trade")
    lines = [
        f"🟢 <b>NEW SIGNAL — {p['city'].upper()}</b>",
        f"📅 Market date: <b>{p['target_date']}</b>",
        f"🎯 Predicted: <b>{p['top_bucket']}{sym}</b> at <b>{p['top_prob']*100:.0f}%</b>",
        f"🕐 Timing: {tim.get('quality','?')} (local {tim.get('city_local_now','?')})",
    ]
    # model agreement context
    agr = p.get("agreement")
    if agr:
        lines.append(f"📊 Models: {agr} agreement (DEB {p.get('deb')}{sym})")
    # live polymarket edge if available
    if bt:
        lines.append(
            f"💰 <b>{bt['action']} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢ "
            f"→ EDGE {bt['best_edge']*100:+.0f}%</b>"
        )
        if bt.get("thin"):
            lines.append("⚠️ Low volume — keep size tiny")
    elif p.get("polymarket") is not None:
        lines.append("ℹ️ Market found but no >10% edge — verify before trading")
    pm = p.get("polymarket")
    if pm and pm.get("url"):
        lines.append(f'🔗 {pm["url"]}')
    return "\n".join(lines)


def fmt_collapse(p, prev_prob) -> str:
    sym = p["temp_unit"]
    reasons = p.get("verdict_reasons") or []
    reason  = reasons[0] if reasons else "probability dropped"
    lines = [
        f"🔴 <b>SIGNAL COLLAPSED — {p['city'].upper()}</b>",
        f"📅 Market date: {p['target_date']}",
        f"📉 {p['top_bucket']}{sym} dropped: {prev_prob*100:.0f}% → <b>{p['top_prob']*100:.0f}%</b>",
        f"⚠️ Reason: {reason}",
        f"👉 If you hold this position, reconsider — model no longer confident.",
    ]
    return "\n".join(lines)


# ── Position update ───────────────────────────────────────────────────────────
def fmt_positions_update(wallet: str, positions) -> str:
    if positions is None:
        return "⚠️ Could not fetch your positions (check wallet / network)."
    if not positions:
        return "💼 <b>Positions update</b>\nNo open weather positions right now."

    lines = ["💼 <b>YOUR POSITIONS</b>"]
    total_now  = 0.0
    total_paid = 0.0
    total_pnl  = 0.0
    claimable  = []

    for p in positions:
        title = (p.get("title") or "")[:32]
        side  = (p.get("outcome") or "?")
        entry = p.get("avg_price")
        now   = p.get("cur_price")
        val   = p.get("current_value") or 0
        pnl   = p.get("cash_pnl") or 0
        total_now  += val
        total_paid += p.get("initial_value") or 0
        total_pnl  += pnl
        if p.get("redeemable"):
            claimable.append(p)

        e_s   = f"{entry*100:.0f}¢" if entry is not None else "—"
        n_s   = f"{now*100:.0f}¢"   if now   is not None else "—"
        emoji = "🟢" if pnl >= 0 else "🔴"
        lines.append(f"{emoji} {title} ({side}) {e_s}→{n_s}  ${val:.2f} ({pnl:+.2f})")

    tot_e = "🟢" if total_pnl >= 0 else "🔴"
    roi = (total_pnl / total_paid * 100) if total_paid > 0 else 0
    lines.append(f"━━━━━━━━━━━━━━━━━━━")
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


# ── One monitoring pass ───────────────────────────────────────────────────────
def run_scan(conn):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[{ts}] scanning {len(ALERT_CITIES)} cities (threshold {THRESHOLD*100:.0f}%)...")

    new_alerts = 0
    collapses  = 0

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

        prev = get_state(conn, key)
        prev_alerted = prev["alerted_high"] if prev else 0
        prev_prob    = prev["prob"] if prev else None

        crossed_up = clean and prob >= THRESHOLD and not prev_alerted
        collapsed  = prev_alerted and (prob < THRESHOLD or not clean)

        if crossed_up:
            msg = fmt_new_signal(p)
            send_telegram(msg)
            upsert_state(conn, key, p["city"], tdate, bucket, prob, alerted_high=1)
            new_alerts += 1
            print(f"  🟢 ALERT {city} {bucket}° {prob*100:.0f}%")
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

    print(f"[{ts}] done — {new_alerts} new, {collapses} collapsed")
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
    print(f"  Cities:            {len(ALERT_CITIES)}")
    print(f"  Prices:            {'on' if USE_PRICES else 'off'}")
    print(f"  State DB:          {STATE_DB}")
    print(f"  Telegram:          {'configured' if (TG_TOKEN and TG_CHAT) else 'NOT configured'}")
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

    while True:
        now = time.time()

        # signal scan timer
        if now - last_scan >= INTERVAL_MIN * 60:
            try:
                run_scan(conn)
            except Exception as e:
                print(f"[loop] scan error: {e}")
            last_scan = time.time()

        # position update timer
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
