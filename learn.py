"""
learn.py — Daily prediction-vs-outcome learning tracker for PolyWeather
========================================================================
Closes the loop between what OUR BOT predicted and what ACTUALLY won on
Polymarket, for all 51 cities, every day.

For each city/day it:
  1. RECORDS the bot's current prediction (top bucket, prob, verdict, the full
     model blend, the live Polymarket bucket prices, and any trade the bot would
     make). This snapshot can be taken any time before the market settles — the
     latest snapshot before the city's local day ends is the one that's scored.
  2. SETTLES the outcome once that local day is complete: fetches the actual
     settled daily high (Wunderground — Polymarket's own source), rounds it to
     the winning bucket, and also reads which Polymarket bucket resolved YES.
  3. SCORES the bot: did our predicted bucket match the bucket that won? And on
     the cities where the bot actually made a call (verdict=TRADE), would that
     call have won?

It also feeds every settled actual back into the existing DEB learner
(deb_history.json) so the temperature model keeps improving too.

Storage: a single JSON file on the persistent volume (/data/learn_history.json
on Railway, or ./learn_history.json locally; override with LEARN_HISTORY_FILE).

CLI:
    python learn.py record              # snapshot all 51 cities now
    python learn.py record paris milan  # snapshot specific cities
    python learn.py settle              # score every completed-but-unscored day
    python learn.py report              # scoreboard for the most recent settled day
    python learn.py report 2026-06-17   # scoreboard for a specific date
    python learn.py report --all        # lifetime accuracy across all days
    python learn.py run                 # record + settle + print report (one shot)
"""

import os
import sys
import json
import argparse
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any

import polyweather_predict as pw

# A Polymarket bucket whose YES has settled at/above this is the winner.
_RESOLVED_YES = 0.95


# ── persistence ───────────────────────────────────────────────────────────────
def _resolve_learn_file() -> str:
    env = os.environ.get("LEARN_HISTORY_FILE")
    if env:
        return env
    if os.path.isdir("/data"):
        return "/data/learn_history.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn_history.json")


_LEARN_FILE = _resolve_learn_file()
_LOCK = threading.Lock()


def _load() -> dict:
    try:
        if os.path.exists(_LEARN_FILE):
            with open(_LEARN_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save(data: dict):
    try:
        tmp = _LEARN_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _LEARN_FILE)
    except Exception as e:
        print(f"  [learn] save error: {e}")


# ── Per-day alert log ('threads' grouped by day; pull up with /alerts) ─────────
def _resolve_alerts_file() -> str:
    env = os.environ.get("ALERTS_LOG_FILE")
    if env:
        return env
    if os.path.isdir("/data"):
        return "/data/alerts_log.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts_log.json")

ALERTS_FILE = _resolve_alerts_file()
_ALERTS_LOCK = threading.Lock()

def _load_alerts() -> dict:
    try:
        if os.path.exists(ALERTS_FILE):
            with open(ALERTS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def log_alert_line(line: str):
    """Append a one-line alert summary to TODAY's thread (grouped by your local
    date, so /alerts gives you a clean folder-per-day view). Never raises."""
    try:
        mins, _ = _user_tz()
        now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=mins)
        day = now.strftime("%Y-%m-%d")
        with _ALERTS_LOCK:
            d = _load_alerts()
            d.setdefault(day, []).append({"t": now.strftime("%I:%M %p").lstrip("0"), "line": line})
            d[day] = d[day][-300:]                 # cap per day
            # prune to the last 60 days
            for old in sorted(d.keys())[:-60]:
                d.pop(old, None)
            tmp = ALERTS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=2)
            os.replace(tmp, ALERTS_FILE)
    except Exception:
        pass

def report_alerts(date: Optional[str] = None) -> str:
    """All alerts for a day, as one grouped thread. Defaults to today (your tz)."""
    d = _load_alerts()
    if not d:
        return "🧵 No alerts logged yet — they thread here by day as they fire."
    mins, lbl = _user_tz()
    today = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=mins)).strftime("%Y-%m-%d")
    day = date or today
    items = d.get(day) or []
    if not items:
        have = ", ".join(sorted(d.keys())[-7:])
        return f"🧵 No alerts on {day}. Days with alerts: {have}"
    L = [f"🧵 <b>ALERTS — {day} ({lbl})</b>  ·  {len(items)} alert(s)"]
    for it in items:
        L.append(f"   {it.get('t','')}  {it.get('line','')}")
    return "\n".join(L)


# ── 1. RECORD ─────────────────────────────────────────────────────────────────
def _build_snap(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn a prediction dict into a compact snapshot, or None if unusable."""
    if not p or "error" in p:
        return None
    if not p.get("city") or not p.get("target_date"):
        return None
    pm = (p.get("polymarket") or {})
    pm_buckets = {}
    for k, v in (pm.get("buckets") or {}).items():
        y = v.get("yes")
        if y is not None:
            pm_buckets[str(k)] = round(float(y), 3)
    bt = p.get("best_trade")
    return {
        "ts":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predicting": p.get("predicting"),
        "unit":       p.get("temp_unit"),
        "top_bucket": p.get("top_bucket"),
        "top_prob":   round(float(p.get("top_prob") or 0.0), 3),
        "verdict":    p.get("verdict"),
        "made_call":  p.get("verdict") == "TRADE",
        "deb":        p.get("deb"),
        "deb_raw":    p.get("deb_raw"),       # blend WITHOUT peak bias
        "peak_bias":  p.get("peak_bias"),     # bias added to the raw blend
        "sigma":      p.get("sigma"),
        "forecasts":  p.get("forecasts") or {},
        "timing":     (p.get("timing") or {}).get("quality"),
        "pm_buckets": pm_buckets,
        "pm_title":   pm.get("title"),
        "best_trade": ({"action": bt.get("action"), "temp": bt.get("temp"),
                        "yes_price": bt.get("yes_price"),
                        "edge": bt.get("best_edge")} if bt else None),
    }


def _apply(data: dict, p: Dict[str, Any]) -> bool:
    """Write one prediction's snapshot into `data`. Returns True if changed."""
    snap = _build_snap(p)
    if snap is None:
        return False
    rec = data.setdefault(p["target_date"], {}).setdefault(p["city"], {})
    if "outcome" in rec:                 # already settled — never overwrite history
        return False
    rec["pred"] = snap
    return True


def note(p: Dict[str, Any]):
    """Snapshot a single prediction (loads + saves the file). For CLI/standalone.
    Never raises — learning must never break the trading loop."""
    try:
        with _LOCK:
            data = _load()
            if _apply(data, p):
                _save(data)
    except Exception:
        pass


def note_many(preds: List[Dict[str, Any]]):
    """Snapshot a whole scan's worth of predictions with ONE load + save.
    This is what the monitor calls each scan (51 cities → 1 disk write)."""
    try:
        with _LOCK:
            data = _load()
            changed = False
            for p in preds:
                changed = _apply(data, p) or changed
            if changed:
                _save(data)
    except Exception:
        pass


def note_alert(p: Dict[str, Any], had_position: bool):
    """Record a BUY alert as a (possibly missed) trade for what-if $1 P&L.

    Called from the monitor only when a live buy alert fires. Stores the exact
    buy details the bot showed (bucket, side, entry price) plus whether you
    already held a position on that city — so we can later tell you what you
    missed (or avoided) by not taking it.
    """
    try:
        bt = p.get("best_trade")
        city = p.get("city")
        date = p.get("target_date")
        if not bt or not city or not date:
            return
        snap = {
            "ts":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action":       bt.get("action"),
            "bucket":       bt.get("temp"),
            "yes_price":    bt.get("yes_price"),
            "edge":         bt.get("best_edge"),
            "prob":         round(float(p.get("top_prob") or 0.0), 3),
            "unit":         p.get("temp_unit"),
            "had_position": bool(had_position),
        }
        with _LOCK:
            data = _load()
            rec = data.setdefault(date, {}).setdefault(city, {})
            if "outcome" in rec or "alert" in rec:
                return            # already settled, or already recorded today
            rec["alert"] = snap
            _save(data)
    except Exception:
        pass


def _alert_pnl(al: dict, actual_bucket) -> Optional[dict]:
    """$1-stake P&L for a recorded alert, given the settled winning bucket.
    BUY YES at price p: win pays 1/p per $1 (profit 1/p-1); lose = -$1."""
    bk   = al.get("bucket")
    side = al.get("action")
    yp   = al.get("yes_price")
    if bk is None or yp is None:
        return None
    if side == "BUY NO":
        entry = round(1.0 - yp, 3)
        won   = (bk != actual_bucket)
    else:                                       # BUY YES (default)
        entry = round(yp, 3)
        won   = (bk == actual_bucket)
    entry = max(0.01, min(0.99, entry))
    pnl   = (1.0 / entry - 1.0) if won else -1.0
    return {"won": won, "entry": round(entry, 3), "pnl": round(pnl, 2)}


def record_all(cities: Optional[List[str]] = None) -> int:
    """Standalone snapshot pass — predicts each city and notes it. Returns count."""
    cities = cities or list(pw.CITIES.keys())
    n = 0
    for city in cities:
        ck = pw.resolve_city(city) or city
        try:
            p = pw.predict(ck, fetch_prices=True)
        except Exception as e:
            print(f"  [learn] {city}: predict error {e}")
            continue
        if "error" in p:
            continue
        note(p)
        n += 1
        b, pr, v = p.get("top_bucket"), p.get("top_prob", 0), p.get("verdict")
        print(f"  recorded {ck:<14} {b}{p.get('temp_unit','°')} {pr*100:>3.0f}% {v}")
    print(f"[learn] recorded {n} cities")
    return n


# ── 2. SETTLE ─────────────────────────────────────────────────────────────────
def _winning_bucket_from_market(city: str, date: str) -> Optional[int]:
    """Which Polymarket bucket resolved YES (~1.0), if the market is settled."""
    try:
        pm = pw.fetch_polymarket_market(city, date)
    except Exception:
        pm = None
    if not pm or not pm.get("buckets"):
        return None
    best_t, best_y = None, -1.0
    for t, b in pm["buckets"].items():
        y = b.get("yes")
        if y is not None and y > best_y:
            best_t, best_y = t, y
    if best_y >= _RESOLVED_YES:
        return best_t
    return None


def _day_complete(city: str, date: str) -> bool:
    """True once `date` is fully in the past in the city's local time."""
    meta = pw.CITIES.get(city)
    if not meta:
        return False
    local_now = pw._now_utc() + timedelta(seconds=meta.get("tz", 0))
    return date < local_now.strftime("%Y-%m-%d")


def settle(feed_deb: bool = True) -> int:
    """Score every recorded day/city that is complete but not yet settled.

    Returns the number of newly-settled city/day pairs.
    """
    with _LOCK:
        data = _load()

    settled = 0
    for date in sorted(data.keys()):
        for city, rec in data[date].items():
            if rec.get("outcome") or "pred" not in rec:
                continue
            if not _day_complete(city, date):
                continue            # market not done yet — leave it for later

            try:
                actual = pw.fetch_actual_high(city, date)
            except Exception:
                actual = None
            if actual is None:
                continue            # settlement data not available yet; retry next run

            actual_bucket = pw.settlement_round(city, actual)
            pm_win        = _winning_bucket_from_market(city, date)

            pred       = rec["pred"]
            bot_bucket = pred.get("top_bucket")
            hit        = (bot_bucket is not None and bot_bucket == actual_bucket)

            # Would the bot's actual trade have won?
            trade_win = None
            bt = pred.get("best_trade")
            if bt and bt.get("temp") is not None:
                if bt.get("action") == "BUY YES":
                    trade_win = (bt["temp"] == actual_bucket)
                elif bt.get("action") == "BUY NO":
                    trade_win = (bt["temp"] != actual_bucket)

            rec["outcome"] = {
                "actual_high":       actual,
                "actual_bucket":     actual_bucket,
                "pm_winning_bucket": pm_win,
                "settled_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            rec["score"] = {
                "bot_bucket":   bot_bucket,
                "hit":          hit,
                "made_call":    bool(pred.get("made_call")),
                "trade_win":    trade_win,
            }

            # Settle a recorded buy-alert into $1 what-if P&L (missed-trade tracker)
            al = rec.get("alert")
            if al and "result" not in al:
                res = _alert_pnl(al, actual_bucket)
                if res:
                    al["result"] = res
            settled += 1

            if feed_deb:
                # keep the temperature model learning too
                try:
                    pw.record_actual(city, date, actual)
                except Exception:
                    pass

    if settled:
        with _LOCK:
            _save(data)
    print(f"[learn] settled {settled} city/day outcome(s)")
    return settled


# ── 3. REPORT ─────────────────────────────────────────────────────────────────
def _latest_settled_date(data: dict) -> Optional[str]:
    dates = [d for d in data
             if any("outcome" in rec for rec in data[d].values())]
    return max(dates) if dates else None


def report(date: Optional[str] = None, all_time: bool = False) -> str:
    data = _load()
    if not data:
        return "📊 No learning data yet — run `learn.py record` first."

    if all_time:
        return _report_all_time(data)

    date = date or _latest_settled_date(data)
    if not date or date not in data:
        return "📊 No settled outcomes yet to report."

    day = data[date]
    scored = [(c, r) for c, r in day.items() if "score" in r and "outcome" in r]
    if not scored:
        return f"📊 {date}: no settled outcomes yet."

    # split into "bot made a call" vs everything
    calls = [(c, r) for c, r in scored if r["score"].get("made_call")]
    call_hits = sum(1 for _, r in calls if r["score"]["hit"])
    all_hits  = sum(1 for _, r in scored if r["score"]["hit"])

    L = [f"📊 <b>LEARNING REPORT — {date}</b>  ({len(scored)} cities settled)"]

    if calls:
        pct = call_hits / len(calls) * 100
        L.append("")
        L.append(f"🎯 <b>Bot made a call on {len(calls)} cities — "
                 f"{call_hits} ✅ / {len(calls)-call_hits} ❌ ({pct:.0f}% hit)</b>")
        for c, r in sorted(calls, key=lambda x: x[1]["score"]["hit"], reverse=True):
            L.append(_line(c, r))

    # cities where the bot did NOT make a firm call but still had a top pick
    others = [(c, r) for c, r in scored if not r["score"].get("made_call")]
    if others:
        L.append("")
        L.append(f"👀 <b>Watched (no firm call) — {len(others)} cities</b>")
        for c, r in sorted(others, key=lambda x: x[1]["score"]["hit"], reverse=True)[:12]:
            L.append(_line(c, r))
        if len(others) > 12:
            L.append(f"   …and {len(others)-12} more")

    L.append("")
    L.append(f"📈 <b>Overall top-bucket accuracy: {all_hits}/{len(scored)} "
             f"({all_hits/len(scored)*100:.0f}%)</b>")
    return "\n".join(L)


def _line(city: str, rec: dict) -> str:
    s = rec["score"]; o = rec["outcome"]; p = rec.get("pred", {})
    unit = p.get("unit", "°")
    emoji = "✅" if s["hit"] else "❌"
    word  = "WON" if s["hit"] else "LOST"
    bot_b = s.get("bot_bucket")
    prob  = p.get("top_prob", 0) * 100
    verd  = p.get("verdict", "?")
    act_b = o.get("actual_bucket")
    act_h = o.get("actual_high")
    extra = ""
    if s.get("trade_win") is not None:
        extra = "  💰trade WON" if s["trade_win"] else "  💸trade lost"
    # Show the raw blend + bias so the peak-bias contribution is visible.
    blend = ""
    raw, bias = p.get("deb_raw"), p.get("peak_bias")
    if raw is not None and bias is not None and abs(bias) >= 0.05:
        blend = f" [raw {raw}{unit}{bias:+.1f}→{p.get('deb')}{unit}]"
    return (f"   {emoji} {city_disp(city):<13} bot {bot_b}{unit} @{prob:.0f}% {verd:<5} "
            f"→ actual {act_h}{unit} ({act_b}{unit}) {word}{extra}{blend}")


def city_disp(c: str) -> str:
    return {"new york": "NYC"}.get(c, c.title())


def _user_tz():
    return int(os.environ.get("USER_TZ_OFFSET_MIN", "330")), os.environ.get("USER_TZ_LABEL", "IST")

def _now_user_str() -> str:
    mins, lbl = _user_tz()
    t = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=mins)
    return f"{t.strftime('%I:%M %p').lstrip('0')} {lbl}"

def _now_city_str(ck: str) -> str:
    tz = (pw.CITIES.get(ck) or {}).get("tz", 0)
    return (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=tz)).strftime("%H:%M")

def _ts_to_user_str(ts: str) -> Optional[str]:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None
    mins, lbl = _user_tz()
    u = dt + timedelta(minutes=mins)
    return f"{u.strftime('%Y-%m-%d %I:%M %p').replace(' 0', ' ')} {lbl}"


def _report_all_time(data: dict) -> str:
    total = hits = call_total = call_hits = trade_total = trade_wins = 0
    for date, day in data.items():
        for city, rec in day.items():
            s = rec.get("score")
            if not s:
                continue
            total += 1
            hits  += 1 if s["hit"] else 0
            if s.get("made_call"):
                call_total += 1
                call_hits  += 1 if s["hit"] else 0
            if s.get("trade_win") is not None:
                trade_total += 1
                trade_wins  += 1 if s["trade_win"] else 0
    if total == 0:
        return "📊 No settled outcomes yet."
    L = ["📊 <b>LIFETIME LEARNING</b>",
         f"   Days tracked: {len(data)}",
         f"   Top-bucket accuracy: {hits}/{total} ({hits/total*100:.0f}%)"]
    if call_total:
        L.append(f"   🎯 When bot made a call: {call_hits}/{call_total} "
                 f"({call_hits/call_total*100:.0f}%)")
    if trade_total:
        L.append(f"   💰 Trade win rate: {trade_wins}/{trade_total} "
                 f"({trade_wins/trade_total*100:.0f}%)")
    return "\n".join(L)


def report_calibration() -> str:
    """Calibration: does a predicted '70%' actually win ~70% of the time?

    Bins every settled prediction by the bot's stated top-bucket probability and
    compares it to the REALISED hit rate. This is the single most important
    learning signal — if the bot is over-confident (says 80%, wins 55%), you do
    NOT trust its calls yet; if it's well-calibrated, its edges are real.
    """
    data = _load()
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    tally = {b: [0, 0] for b in bins}   # bin -> [hits, total]
    for day in data.values():
        for rec in day.values():
            s = rec.get("score")
            p = rec.get("pred")
            if not s or not p:
                continue
            prob = p.get("top_prob")
            if prob is None:
                continue
            for b in bins:
                if b[0] <= prob < b[1]:
                    tally[b][1] += 1
                    tally[b][0] += 1 if s.get("hit") else 0
                    break
    total = sum(t[1] for t in tally.values())
    if total == 0:
        return "📊 No settled predictions yet — calibration needs a few days of data."

    L = ["📊 <b>CALIBRATION</b>  (predicted confidence vs actual win rate)"]
    for b in bins:
        hits, n = tally[b]
        if n == 0:
            continue
        actual = hits / n * 100
        mid    = (b[0] + b[1]) / 2 * 100
        flag   = "✅" if abs(actual - mid) <= 12 else "⚠️"   # within ~one bin = OK
        L.append(f"   {int(b[0]*100)}–{int(b[1]*100)}% said → {actual:>3.0f}% won "
                 f"({hits}/{n}) {flag}")
    L.append("")
    L.append("✅ = roughly honest · ⚠️ = over/under-confident in that band")
    L.append(f"<i>{total} settled predictions analysed.</i>")
    return "\n".join(L)


def report_sources(city: Optional[str] = None) -> str:
    """Per-API reliability vs settled actuals — which sources to trust.

    For every settled day we stored each API's raw reading for that city. Here we
    compare each one to the actual settled high: mean absolute error (normalised
    to °C so cities in °F and °C can be compared) and how often that source alone
    rounded to the winning bucket. Lower error / higher bucket% = more reliable.
    Pass a city to see reliability for just that location (e.g. NWS for US cities).
    """
    data = _load()
    agg: Dict[str, list] = {}   # source -> [sum_abs_err_C, n, bucket_hits]
    for day in data.values():
        for ck, rec in day.items():
            if city and ck != city:
                continue
            o = rec.get("outcome"); p = rec.get("pred")
            if not o or not p or o.get("actual_high") is None:
                continue
            actual = o["actual_high"]
            abk    = o.get("actual_bucket")
            use_f  = bool((pw.CITIES.get(ck) or {}).get("f", False))
            for src, val in (p.get("forecasts") or {}).items():
                if val is None:
                    continue
                err = abs(val - actual) / (1.8 if use_f else 1.0)   # → °C
                a = agg.setdefault(src, [0.0, 0, 0])
                a[0] += err
                a[1] += 1
                try:
                    if pw.settlement_round(ck, val) == abk:
                        a[2] += 1
                except Exception:
                    pass

    rows = [(se / n, hits / n * 100.0, n, src)
            for src, (se, n, hits) in agg.items() if n > 0]
    if not rows:
        return "📡 No settled data yet — source reliability needs a few settled days."
    rows.sort()                                   # by MAE ascending = best first
    scope = f" [{city_disp(city)}]" if city else " [all cities]"
    L = [f"📡 <b>SOURCE RELIABILITY{scope}</b>  (lower error = better)"]
    for mae, hr, n, src in rows:
        L.append(f"   {src:<13} MAE {mae:.1f}°C  ·  bucket {hr:.0f}%  ({n})")
    L.append("")
    L.append("<i>MAE = avg miss vs settled high · bucket% = how often it alone hit the winner</i>")
    return "\n".join(L)


def report_bias_free() -> str:
    """Re-score the bot's settled TRADE calls using the RAW source consensus
    (median of the stored per-API readings, with NO peak bias) and compare to the
    bot's actual (bias-included) hit rate. Answers: is the peak bias helping?

    Note: the raw consensus is an APPROXIMATION of the no-bias prediction — it
    ignores the live-obs / dead-market adjustments predict() also makes — but it's
    a fair read on whether the +bias nudge changed outcomes.
    """
    data = _load()
    wb_hits = wb_n = nb_hits = nb_n = 0
    flips: List[str] = []
    for day in data.values():
        for ck, rec in day.items():
            o = rec.get("outcome"); p = rec.get("pred"); s = rec.get("score")
            if not o or not p or not s or not s.get("made_call"):
                continue
            abk = o.get("actual_bucket")
            wb_n += 1
            wb_hit = bool(s.get("hit"))
            wb_hits += 1 if wb_hit else 0
            vals = [v for v in (p.get("forecasts") or {}).values() if v is not None]
            if not vals:
                continue
            med = sorted(vals)[len(vals) // 2]
            try:
                nb_bucket = pw.settlement_round(ck, med)
            except Exception:
                continue
            nb_n += 1
            nb_hit = (nb_bucket == abk)
            nb_hits += 1 if nb_hit else 0
            if nb_hit != wb_hit:
                arrow = "bias WON, raw lost" if wb_hit else "raw WON, bias lost"
                flips.append(f"   {city_disp(ck)}: bot {p.get('top_bucket')}° vs raw "
                             f"{nb_bucket}° → actual {abk}°  ({arrow})")
    if wb_n == 0:
        return "📊 No settled calls yet to compare."
    L = ["🧪 <b>PEAK-BIAS CHECK</b>  (bot's calls: with vs without the bias)",
         f"   With bias (live):     {wb_hits}/{wb_n} ({wb_hits/wb_n*100:.0f}%)"]
    if nb_n:
        L.append(f"   Without bias (raw):   {nb_hits}/{nb_n} ({nb_hits/nb_n*100:.0f}%)")
    if flips:
        L.append("")
        L.append("Where they differ:")
        L.extend(flips[:12])
    else:
        L.append("")
        L.append("<i>Peak bias changed no outcomes on settled calls so far.</i>")
    return "\n".join(L)


def report_missed() -> str:
    """The missed-trade tracker: for every buy alert where you had NO position,
    show the $1 what-if result — what you'd have made (or avoided) by taking it.
    """
    data = _load()
    missed, taken = [], []
    for day in data.values():
        for ck, rec in day.items():
            al = rec.get("alert")
            r  = (al or {}).get("result")
            if not al or not r:
                continue
            (taken if al.get("had_position") else missed).append((r, al, ck))
    if not missed and not taken:
        return ("💸 No settled alerts yet. Once the bot alerts a city you don't hold "
                "and that market settles, this shows what you missed (or avoided).")

    L = ["💸 <b>MISSED-TRADE TRACKER</b>  ($1 per alert you weren't holding)"]
    if missed:
        net = sum(r["pnl"] for r, _, _ in missed)
        won = sum(1 for r, _, _ in missed if r["won"])
        L.append("")
        L.append(f"⚠️ I alerted you on {len(missed)} market(s) you had NO position on:")
        for r, al, ck in sorted(missed, key=lambda x: x[0]["pnl"], reverse=True)[:15]:
            u    = al.get("unit", "°")
            cents = (al.get("yes_price") or 0) * 100
            if r["won"]:
                L.append(f"   ✅ {city_disp(ck)} {al.get('bucket')}{u} @ {cents:.0f}¢ WON "
                         f"→ you MISSED +${r['pnl']:.2f}")
            else:
                L.append(f"   ❌ {city_disp(ck)} {al.get('bucket')}{u} @ {cents:.0f}¢ lost "
                         f"→ you AVOIDED -$1.00")
        L.append("")
        L.append(f"<b>Net if you'd taken every $1 alert: {net:+.2f} USD</b>  "
                 f"({won} won / {len(missed)-won} lost)")
        L.append(f"💡 {'You left ~$%.2f on the table by not acting.' % net if net > 0 else 'Skipping these saved you $%.2f.' % abs(net)}")
    if taken:
        L.append("")
        L.append(f"<i>(You held a position on {len(taken)} other alerted market(s) — not counted as missed.)</i>")
    return "\n".join(L)


def _raw_blend(p: dict) -> Optional[float]:
    """The no-bias blend for a snapshot (deb_raw, or deb minus the bias)."""
    raw = p.get("deb_raw")
    if raw is not None:
        return raw
    deb, pb = p.get("deb"), p.get("peak_bias")
    if deb is not None and pb is not None:
        return round(deb - pb, 1)
    return None


def report_city(city: str) -> str:
    """Full prediction-vs-outcome history for ONE city, plus the bias it implies.

    Shows every day's predicted bucket vs the actual, and computes how far the raw
    (no-bias) blend sat from the actual on average — which is exactly the bias to
    set with CITY_BIAS so the blend lands on the real high for that city.
    """
    ck   = pw.resolve_city(city) or (city or "").lower()
    data = _load()
    rows, residuals = [], []
    last_alert_ts = None
    for date in sorted(data.keys()):
        rec = (data.get(date) or {}).get(ck)
        if not rec or "pred" not in rec:
            if rec and rec.get("alert", {}).get("ts"):
                last_alert_ts = rec["alert"]["ts"]
            continue
        if rec.get("alert", {}).get("ts"):
            last_alert_ts = rec["alert"]["ts"]
        p = rec["pred"]; o = rec.get("outcome")
        u = p.get("unit", "°")
        bb, prob = p.get("top_bucket"), p.get("top_prob", 0) * 100
        if o and o.get("actual_high") is not None:
            ab, ah = o.get("actual_bucket"), o.get("actual_high")
            mark = "✅" if bb == ab else "❌"
            rows.append(f"   {date}  pred {bb}{u}@{prob:.0f}% → actual {ab}{u} ({ah}{u}) {mark}")
            raw = _raw_blend(p)
            if raw is not None:
                residuals.append(ah - raw)
        else:
            rows.append(f"   {date}  pred {bb}{u}@{prob:.0f}% → (pending)")
    if not rows:
        return f"📜 No history yet for {city_disp(ck)} — fills in as days settle."

    L = [f"📜 <b>{city_disp(ck)} — prediction history</b>",
         f"🕐 {city_disp(ck)} now {_now_city_str(ck)} · {_now_user_str()}"]
    la = _ts_to_user_str(last_alert_ts) if last_alert_ts else None
    if la:
        L.append(f"🔔 Last alert sent: {la}")
    L.append("")
    L += rows[-14:]
    if residuals:
        mean_res = sum(residuals) / len(residuals)
        where = "BELOW" if mean_res > 0 else "ABOVE"
        L.append("")
        L.append(f"📐 Raw blend ran {abs(mean_res):.1f}°{where} the actual over "
                 f"{len(residuals)} settled day(s).")
        L.append(f"💡 To correct it: <code>CITY_BIAS={ck}:{mean_res:+.1f}</code>")
    return "\n".join(L)


def recent_city_line(city: str, n: int = 3) -> Optional[str]:
    """Short 'pred→actual' track record for the last n settled days — shown in
    the alert so you can see how this city has behaved lately."""
    ck   = pw.resolve_city(city) or (city or "").lower()
    data = _load()
    items = []
    for date in sorted(data.keys(), reverse=True):
        rec = (data.get(date) or {}).get(ck)
        if not rec:
            continue
        p, o = rec.get("pred"), rec.get("outcome")
        if not p or not o or o.get("actual_bucket") is None:
            continue
        u = p.get("unit", "°")
        mark = "✅" if p.get("top_bucket") == o.get("actual_bucket") else "❌"
        items.append(f"{p.get('top_bucket')}→{o.get('actual_bucket')}{u}{mark}")
        if len(items) >= n:
            break
    if not items:
        return None
    return "📜 Recent (pred→actual): " + " · ".join(items)


def settle_and_report() -> str:
    """Settle anything newly complete, then return the latest scoreboard.
    Used by the monitor for the once-a-day Telegram learning digest."""
    try:
        settle()
    except Exception as e:
        print(f"[learn] settle error: {e}")
    return report()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PolyWeather learning tracker")
    ap.add_argument("cmd", choices=["record", "settle", "report", "run"],
                    help="record | settle | report | run")
    ap.add_argument("args", nargs="*", help="cities (for record) or date (for report)")
    ap.add_argument("--all", action="store_true", help="report: lifetime accuracy")
    ap.add_argument("--calib", action="store_true", help="report: calibration table")
    ap.add_argument("--sources", action="store_true", help="report: per-API reliability")
    ap.add_argument("--nobias", action="store_true", help="report: with vs without peak bias")
    a = ap.parse_args()

    if a.cmd == "record":
        record_all(a.args or None)
    elif a.cmd == "settle":
        settle()
    elif a.cmd == "report":
        if a.calib:
            print(_strip(report_calibration()))
        elif a.sources:
            city = next((pw.resolve_city(x) for x in a.args if pw.resolve_city(x)), None)
            print(_strip(report_sources(city)))
        elif a.nobias:
            print(_strip(report_bias_free()))
        else:
            date = next((x for x in a.args if x.count("-") == 2), None)
            print(_strip(report(date, all_time=a.all)))
    elif a.cmd == "run":
        record_all()
        settle()
        print(_strip(report()))


def _strip(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s)


if __name__ == "__main__":
    main()
