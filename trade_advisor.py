"""
trade_advisor.py — degree-distribution trade advisor
====================================================
Turns the bot's bias + no-bias prediction into a concrete, HONEST trade plan:
how to spread your money across temperature degrees and what it really pays.

It gives TWO sizings for a city and recommends one:

  💰 VALUE BET  — bet only the degrees the market UNDER-prices (honest prob > price),
                  sized for equal payout. This is the +EV play: it makes money over
                  many markets. Lower hit-rate / higher variance (the value is usually
                  in the cheaper neighbour degrees, not the pricey favourite).

  🛡️ FULL COVER — spread across the market favourite + the bot's pick + backups so
                  you "win" whichever covered degree settles. Feels safe (high
                  hit-rate) BUT usually -EV, because covering the over-priced
                  favourite is paying to lose slowly. Reported honestly with its EV.

VALIDATION — backtested on the 361 settled markets (budget $4 each):
  FULL COVER : ~54% hit · ~+6% ROI · per-bet std ~$18  → CAPITAL PRESERVATION
               (win more than half the time, small edge). The "don't lose" play
               and the default recommendation. (High variance → treat the ROI as
               "≈break-even-to-slightly-positive", not a guaranteed edge.)
  VALUE BET  : 8% hit · +34% ROI but driven by a few rare big wins (high variance)
               → a lottery, NOT reliable for a small bankroll. Aggressive only.
Neither is a statistically-certain edge on 361 markets — the market is fairly
efficient — but the cover holds your money while you wait for genuine mis-pricings.

Recommendation:
  • full cover available → COVER (default; capital-preservation).
  • else value legs      → VALUE/SPREAD (aggressive).
  • else                 → SKIP.

GROUNDING (361 settled markets)
  actual − bias:    mean −0.16°, std **1.45°**   (the real spread → use this, not the
                                                  bot's over-confident 0.3–0.5° σ)
  actual − no-bias: mean +0.39°  (raw feed under-reads the peak)
So the honest distribution centres on the calibrated bias value with the empirical
±1.45° spread, widened when bias & no-bias disagree.

USAGE
  from trade_advisor import advise, advise_from_prediction, format_advice, one_liner
  a = advise(deb=34.9, deb_raw=34.7, market={34:{"yes":0.39},35:{"yes":0.46},...}, budget=4)
  print(format_advice(a))     # full block
  print(one_liner(a))         # compact, for inline alert use
"""

import math

# ── Empirical calibration (refreshed 2026-07-06 on 871 settled markets) ───────
# actual − deb: mean −0.11° · std 1.38° (was −0.16 / 1.45 on the first 361).
EMP_STD       = 1.38
BIAS_OFFSET   = -0.11
DISAGREE_K    = 0.35
SIGMA_FLOOR   = 1.30

# ── Trade controls ────────────────────────────────────────────────────────────
MIN_PRICE     = 0.01
MAX_PRICE     = 0.97
MIN_EDGE      = 0.04     # a degree counts as "under-priced" if honest prob - price >= this
MIN_COVER_PROB = 0.05    # a degree is worth covering if the MODEL gives it >= this prob
MARKET_ALIVE   = 0.05    # ...OR the MARKET prices it >= this (5¢). A degree the market
                         # treats as live (e.g. 31° @ 27¢) is covered even if our wide-σ
                         # model under-weights it — the market is the better odds here.
MAX_SUM_PRICE = 0.96     # default: stop adding cover degrees once their prices sum past
                         # this. Backtest on 452 settled markets: the tight cover (0.96)
                         # runs ≈break-even, while covering EVERY live degree (~0.99) runs
                         # ≈-4% — you pay the spread for the extra safety. So 0.96 is the
                         # default; "wide" mode (WIDE_SUM_PRICE) is an explicit opt-in.
WIDE_SUM_PRICE = 0.995   # "wide" / full-safety cover: include all live degrees even though
                         # it's ≈break-even-to-slightly-negative (the user's don't-lose play).
MAX_LEGS      = 4
# Polymarket order minimum: a leg must be at least $1 OR at least 5 shares.
MIN_ORDER_USD = 1.0
MIN_SHARES    = 5


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _edges(lo, hi):
    a, b = (lo - 0.5), (hi + 0.5)
    if lo <= -9000: a = -9999.0
    if hi >= 9000:  b = 9999.0
    return a, b

def realistic_distribution(deb, deb_raw, buckets):
    """{value:(lo,hi)} -> ({value: honest prob}, centre, sigma)."""
    center   = deb + BIAS_OFFSET
    disagree = abs(deb - deb_raw) if (deb is not None and deb_raw is not None) else 0.0
    sigma    = max(SIGMA_FLOOR, math.sqrt(EMP_STD ** 2 + (DISAGREE_K * disagree) ** 2))
    dist = {}
    for v, (lo, hi) in buckets.items():
        a, b = _edges(lo, hi)
        dist[v] = max(0.0, _phi((b - center) / sigma) - _phi((a - center) / sigma))
    return dist, center, sigma


def _equal_payout(bucket_list, priced, dist, budget, anchors=frozenset()):
    """Size a set of degrees for EQUAL payout (same money back whichever wins):
    every leg buys S = budget/Σprice shares, so stakeᵢ = S·priceᵢ. Legs that can't
    meet Polymarket's order minimum ($1 OR 5 shares) are DROPPED and the rest
    re-sized — so you never get an un-buyable $0.09 leg. We drop the EXTRA degrees
    first (cheapest non-anchor), protecting the two anchors (bot pick + market
    favourite); only if no extras remain do we drop an anchor. So a small budget
    keeps your core two and a bigger budget ADDS the extra degrees — never the
    other way round."""
    bl = list(bucket_list)
    dropped = []
    while len(bl) > 1:
        sump = sum(priced[v] for v in bl)
        if sump <= 0:
            return None
        S = budget / sump                              # shares per leg (equal payout)
        # a leg is valid if its stake >= $1 OR it buys >= 5 shares
        bad = [v for v in bl if (S * priced[v] < MIN_ORDER_USD - 1e-9) and (S < MIN_SHARES)]
        if not bad:
            break
        # prefer to drop an EXTRA (non-anchor) degree; among those the cheapest =
        # the one the market thinks least likely. Drop an anchor only as a last resort.
        droppable = [v for v in bl if v not in anchors] or bl
        drop = min(droppable, key=lambda v: priced[v])
        bl.remove(drop); dropped.append(drop)
    sump = sum(priced[v] for v in bl)
    if sump <= 0:
        return None
    S = budget / sump
    if len(bl) == 1:                                    # last leg must still be valid
        v = bl[0]
        if S * priced[v] < MIN_ORDER_USD - 1e-9 and S < MIN_SHARES:
            return None
    legs = []
    for v in bl:
        p = priced[v]
        stake = S * p
        legs.append({"bucket": v, "price": round(p, 3), "prob": round(dist.get(v, 0.0), 3),
                     "edge": round(dist.get(v, 0.0) - p, 3),
                     "stake": round(stake, 2), "shares": round(stake / p, 1)})
    total    = round(sum(l["stake"] for l in legs), 2)
    coverage = round(sum(l["prob"] for l in legs), 3)   # MODEL prob mass covered (for EV)
    mkt_cov  = round(min(sump, 1.0), 3)                  # MARKET-implied P(covered) = Σ price
    payout   = round(budget / sump, 2)
    return {"legs": legs, "total": total, "payout": payout, "sum_price": round(sump, 3),
            "coverage": coverage, "mkt_coverage": mkt_cov,
            "profit_if_covered": round(payout - total, 2),
            "loss_if_miss": round(-total, 2),
            "ev": round(coverage * payout - total, 2), "dropped": dropped}


def advise(deb, deb_raw, market, budget=4.0, wide=False):
    """market: {value: {"yes": price, "lo": lo, "hi": hi}} (lo/hi optional).
    wide=True covers ALL live degrees (full-safety, ≈break-even); default is the
    tighter, near-break-even cover that keeps the small edge."""
    if deb is None or not market:
        return {"ok": False, "reason": "no prediction / market"}
    max_sum  = WIDE_SUM_PRICE if wide else MAX_SUM_PRICE
    max_legs = 99 if wide else MAX_LEGS        # wide = cover EVERY live degree, no leg cap
    buckets = {v: (b.get("lo", v), b.get("hi", v)) for v, b in market.items()}
    dist, center, sigma = realistic_distribution(deb, deb_raw, buckets)
    priced = {v: b["yes"] for v, b in market.items()
              if b.get("yes") is not None and MIN_PRICE <= b["yes"] <= MAX_PRICE}
    if not priced:
        return {"ok": False, "reason": "no buyable buckets",
                "center": round(center, 2), "sigma": round(sigma, 2),
                "distribution": dict(sorted(dist.items()))}

    market_fav = max(priced, key=lambda v: priced[v])
    bot_pick   = min(priced, key=lambda v: abs(v - center))

    # ── VALUE BET: only the under-priced degrees, best edge first, Σprice capped ──
    value_order = sorted((v for v in priced if dist.get(v, 0) - priced[v] >= MIN_EDGE),
                         key=lambda v: -(dist.get(v, 0) - priced[v]))
    vbuckets, vp = [], 0.0
    for v in value_order:
        if len(vbuckets) >= MAX_LEGS:
            break
        if vbuckets and vp + priced[v] > max_sum:
            continue
        vbuckets.append(v); vp += priced[v]
    value_bet = _equal_payout(vbuckets, priced, dist, budget) if vbuckets else None

    # ── FULL COVER: bot pick + market favourite + most-likely MEANINGFUL neighbours.
    # Only degrees with a non-trivial honest probability are eligible — never fill a
    # slot with a near-zero-probability cheap bucket (e.g. cover 37° not 36° when the
    # cluster is 38/39). Anchors (bot pick, market favourite) are always included.
    eligible = {bot_pick, market_fav} | {v for v in priced
                if dist.get(v, 0.0) >= MIN_COVER_PROB or priced[v] >= MARKET_ALIVE}
    order, seen = [], set()
    for v in (bot_pick, market_fav):
        if v not in seen:
            order.append(v); seen.add(v)
    for v in sorted(eligible, key=lambda v: -dist.get(v, 0.0)):
        if v not in seen:
            order.append(v); seen.add(v)
    cbuckets, cp = [], 0.0
    for v in order:
        if len(cbuckets) >= max_legs:
            break
        if cbuckets and cp + priced[v] > max_sum:
            continue
        cbuckets.append(v); cp += priced[v]
    full_cover = _equal_payout(cbuckets, priced, dist, budget,
                               anchors=frozenset({bot_pick, market_fav}))

    # Live degrees (the market prices >= MARKET_ALIVE) that the tight cover left out —
    # so the alert can say "31° is live but excluded; go 'wide' to include it".
    covered_set  = {l["bucket"] for l in (full_cover["legs"] if full_cover else [])}
    live_skipped = sorted(v for v in priced
                          if priced[v] >= MARKET_ALIVE and v not in covered_set)

    # ── recommendation ───────────────────────────────────────────────────────
    # Backtest on 452 settled markets (budget $4 each):
    #   TIGHT COVER (default, cap 0.96): ~55% hit · ≈break-even (−0.4%) — the least-bad
    #                play; against an efficient market a cover ≈ parks money.
    #   WIDE COVER  (cap ~0.99, all live degrees): ~60% hit but ≈−4% — you pay the
    #                spread for the extra safety. Opt-in only.
    #   VALUE BET   : rare-big-win lottery, high variance — aggressive option only.
    # So default to the tight cover; surface value/wide as explicit choices.
    if full_cover and full_cover["legs"]:
        rec = "cover"
    elif value_bet:
        rec = "value"
    else:
        rec = "skip"

    return {
        "ok": True, "center": round(center, 2), "sigma": round(sigma, 2),
        "distribution": dict(sorted(dist.items())),
        "market_fav": market_fav, "bot_pick": bot_pick,
        "value_bet": value_bet, "full_cover": full_cover,
        "recommendation": rec, "budget": budget,
        "wide": wide, "live_skipped": live_skipped,
    }


def advise_budgets(deb, deb_raw, market, budgets=(4.0, 5.0, 6.0), wide=False):
    """Compute the cover at several budgets. Default = the tight, near-break-even
    cover; wide=True covers every live degree (full-safety, ≈break-even-to--4%).
    Bigger budget affords more backup degrees (once each clears the order minimum)."""
    base = advise(deb, deb_raw, market, budget=budgets[0], wide=wide)
    rows, hi_skipped = [], base.get("live_skipped") or []
    for b in budgets:
        a = advise(deb, deb_raw, market, budget=b, wide=wide)
        rows.append({"budget": b, "cover": (a.get("full_cover") if a.get("ok") else None)})
        if b == max(budgets):                 # what's left out even at the biggest budget
            hi_skipped = a.get("live_skipped") or []
    valid = [r for r in rows if r["cover"] and r["cover"]["legs"]
             and r["cover"]["profit_if_covered"] > 0]
    # There is no single "best" — it's a tradeoff:
    #   💰 keep the edge: most profit IF it lands in your covered set (fewer degrees).
    #   🛡️ safest:        most degrees covered (least chance of a miss), thinner margin.
    most_profit = max(valid, key=lambda r: (r["cover"]["profit_if_covered"], -r["budget"]),
                      default=None)
    most_cover  = max(valid, key=lambda r: (r["cover"]["mkt_coverage"], -r["budget"]),
                      default=None)
    return {"ok": base.get("ok"), "center": base.get("center"), "sigma": base.get("sigma"),
            "distribution": base.get("distribution"), "wide": wide,
            "live_skipped": hi_skipped,
            "market_fav": base.get("market_fav"), "bot_pick": base.get("bot_pick"),
            "rows": rows,
            "profit_budget": (most_profit["budget"] if most_profit else None),
            "cover_budget":  (most_cover["budget"] if most_cover else None)}


def format_budgets(res, sym="°C"):
    """Multi-budget cover block — each budget shows the exact per-degree $ and
    share allocation (equal-payout) and the guaranteed return if covered."""
    if not res.get("ok"):
        return ""
    short = sym.replace(chr(176), "")
    wide   = res.get("wide")
    n_rows = len([r for r in res["rows"] if r["cover"] and r["cover"]["legs"]])
    if wide:
        title = "🎓 <b>Wide cover</b> (every live degree — full safety):"
    elif n_rows <= 1:
        title = "🎓 <b>Cover</b> (spread $ so whichever covered degree settles, you win):"
    else:
        title = "🎓 <b>Cover options</b> (spread $ so whichever covered degree settles, you win):"
    L = [title]
    any_row = False
    pb, cb = res.get("profit_budget"), res.get("cover_budget")
    # only differentiate budgets when they actually cover DIFFERENT degree sets;
    # if every budget covers the same set, the markers would just mean "more $".
    covs = {round(r["cover"]["mkt_coverage"], 3)
            for r in res["rows"] if r["cover"] and r["cover"]["legs"]}
    differ = len(covs) > 1
    for r in res["rows"]:
        s, b = r["cover"], r["budget"]
        if not s or not s["legs"]:
            L.append(f"   <b>${b:g}</b>: — no buyable cover at this size")
            continue
        any_row = True
        legs = " + ".join(f"{l['bucket']}{short} @ {l['price']*100:.0f}¢ → ${l['stake']:.2f} "
                          f"({l['shares']:g} sh)" for l in s["legs"])
        mark = ""
        if differ and pb != cb:
            if b == cb:
                mark = " 🛡️"
            elif b == pb:
                mark = " 💰"
        drp = (f"  [+{'/'.join(str(d) + short for d in s.get('dropped') or [])} needs more $]"
               if s.get("dropped") else "")
        L.append(f"   <b>${b:g}</b> → {legs} → <b>${s['payout']:.2f}</b> back · "
                 f"{s['mkt_coverage']*100:.0f}% covered · profit {s['profit_if_covered']:+.2f}{mark}{drp}")
    if not any_row:
        return ""
    if differ and pb and cb and pb != cb:
        L.append(f"   💰 <b>${pb:g}</b> = most profit if covered (keeps an edge, but can miss) · "
                 f"🛡️ <b>${cb:g}</b> = most degrees covered (safest, ≈break-even)")
    elif not differ and len([r for r in res["rows"] if r["cover"] and r["cover"]["legs"]]) > 1:
        L.append("   <i>(same degrees at each budget — a bigger budget just scales the "
                 "stake &amp; payout, not the coverage.)</i>")
    skipped = res.get("live_skipped") or []
    degs = "/".join(f"{d}{short}" for d in skipped)
    if skipped and wide:
        # Even in WIDE mode a live degree can be excluded: it would push the covered
        # prices over $1, i.e. you'd pay more than the $1 you can win. Say so — don't
        # claim "every live degree" while silently dropping one.
        L.append(f"   ⚠️ <b>{degs}</b> can't be added — the covered prices already sum to "
                 f"~$1, so including it would cost more than you could win (a guaranteed "
                 f"loss). Covering the rest only; the market sees those as near-certain.")
    elif skipped:
        L.append(f"   ⚠️ <b>{degs}</b> priced live but left out — covering it would push the "
                 f"spread to ~$1 (≈break-even). It's excluded on purpose: backtest says "
                 f"covering every live degree runs ≈-4%. Add <code>wide</code> "
                 f"(e.g. <code>/cover &lt;city&gt; wide</code>) to include it anyway.")
    else:
        L.append("   <i>ℹ️ a cover ≈ parks money — against an efficient market it's roughly "
                 "break-even; the tight cover keeps a small edge.</i>")
    return "\n".join(L)


def _market_from_pred(p):
    pm = (p.get("polymarket") or {}).get("buckets") or {}
    market = {}
    for v, b in pm.items():
        try:
            market[int(v)] = {"yes": b.get("yes"), "lo": b.get("lo", int(v)), "hi": b.get("hi", int(v))}
        except (TypeError, ValueError):
            continue
    return market


def advise_budgets_from_prediction(p, budgets=(4.0, 5.0, 6.0), wide=False):
    return advise_budgets(p.get("deb"), p.get("deb_raw"), _market_from_pred(p),
                          budgets=budgets, wide=wide)


def advise_from_prediction(p, budget=4.0, wide=False):
    return advise(p.get("deb"), p.get("deb_raw"), _market_from_pred(p),
                  budget=budget, wide=wide)


# ── formatting ────────────────────────────────────────────────────────────────
def _legs_str(sizing, sym):
    return " + ".join(f"{l['bucket']}{sym} ${l['stake']:.2f}" for l in sizing["legs"])

def one_liner(a, sym="°C"):
    """Compact advice for inline use in an alert. Returns '' if no plan."""
    if not a.get("ok"):
        return ""
    s = a.get("full_cover")
    if not s or not s["legs"]:
        vb = a.get("value_bet")
        if vb:
            return (f"🎓 <b>Spread</b> (${a['budget']:.0f}): {_legs_str(vb, sym)} "
                    f"→ ${vb['payout']:.2f} if any hits (aggressive)")
        return "🎓 Advisor: nothing cleanly under-priced — sit out."
    short = sym.replace(chr(176), "")
    line = (f"🎓 <b>Cover</b> (${a['budget']:.0f}): {_legs_str(s, sym)} → "
            f"<b>${s['payout']:.2f}</b> back on any · win if it's "
            + "/".join(f"{l['bucket']}{short}" for l in s['legs']))
    drp = s.get("dropped")
    if drp:
        line += (f"\n   (skipped {'/'.join(str(d)+short for d in drp)} — under Polymarket's "
                 f"$1/5-share minimum; raise the budget to include them)")
    return line

def format_advice(a, sym="°C"):
    if not a.get("ok"):
        return "🎓 Advisor: " + a.get("reason", "no data")
    L = [f"📐 Honest read: centre {a['center']}{sym} · σ {a['sigma']}{sym} "
         f"(market favours {a['market_fav']}{sym}, bot picks {a['bot_pick']}{sym})"]
    for v, prob in sorted(a["distribution"].items(), key=lambda x: -x[1])[:5]:
        L.append(f"     {v}{sym}  {prob*100:4.0f}%")
    L.append("")
    vb, fc = a.get("value_bet"), a.get("full_cover")
    if vb:
        star = "⭐ " if a["recommendation"] == "value" else ""
        L.append(f"{star}💰 VALUE BET — under-priced degrees, +EV ({vb['ev']:+.2f}):")
        for l in vb["legs"]:
            sh = f", {l['shares']:g} sh" if l["stake"] < MIN_ORDER_USD else ""
            L.append(f"   • {l['bucket']}{sym} @ {l['price']*100:.0f}¢ → ${l['stake']:.2f}{sh} "
                     f"(edge {l['edge']*100:+.0f}%)")
        L.append(f"   → win any → ${vb['payout']:.2f}  (profit {vb['profit_if_covered']:+.2f}, "
                 f"{vb['coverage']*100:.0f}% hit, lose {vb['loss_if_miss']:.2f} if outside)")
        L.append("")
    if fc:
        star = "⭐ " if a["recommendation"] == "cover" else ""
        tag = "+EV ✅" if fc["ev"] > 0 else f"-EV ⚠️ (overpaying for the favourite)"
        L.append(f"{star}🛡️ FULL COVER — market+bot+backups, {tag}:")
        for l in fc["legs"]:
            note = "✅" if l["edge"] > 0 else "💸"
            sh = f", {l['shares']:g} sh" if l["stake"] < MIN_ORDER_USD else ""
            L.append(f"   • {l['bucket']}{sym} @ {l['price']*100:.0f}¢ → ${l['stake']:.2f}{sh} "
                     f"(edge {l['edge']*100:+.0f}% {note})")
        L.append(f"   → win any → ${fc['payout']:.2f}  ({fc['mkt_coverage']*100:.0f}% covered, "
                 f"EV {fc['ev']:+.2f}, lose {fc['loss_if_miss']:.2f} if outside)")
        L.append("")
    rec = {"value": "👉 Take the VALUE BET — it's the +EV play.",
           "cover": "👉 The COVER is under-priced here — safe and +EV.",
           "skip":  "👉 SKIP — nothing under-priced; covering would just bleed the spread."}[a["recommendation"]]
    L.append(rec)
    return "\n".join(L)


# ── demos ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import re
    def show(t, deb, raw, mkt, bud=4.0):
        print("=" * 68); print(t); print("=" * 68)
        print(re.sub(r"</?b>", "", format_advice(advise(deb, raw, mkt, bud))))
        print("\nINLINE:", re.sub(r"</?b>", "", one_liner(advise(deb, raw, mkt, bud)))); print()
    show("MILAN (bias 34.9, raw 34.7) — settled 34",
         34.9, 34.7, {32:{"yes":0.01},33:{"yes":0.02},34:{"yes":0.39},35:{"yes":0.46},36:{"yes":0.06},37:{"yes":0.01}})
    show("AMSTERDAM (bias 30.1, raw 30.7) — market split 29/30",
         30.1, 30.7, {28:{"yes":0.01},29:{"yes":0.39},30:{"yes":0.43},31:{"yes":0.06},32:{"yes":0.01}})
    show("CHENGDU (bias 25.3, raw 24.1) — 25 @ 81¢ near-decided",
         25.3, 24.1, {24:{"yes":0.01},25:{"yes":0.81},26:{"yes":0.13},27:{"yes":0.03}})
    show("SF RANGE (bias 68.4, raw 68.1) — 68-69 @ 36¢",
         68.4, 68.1, {65:{"yes":0.02,"lo":64,"hi":65},67:{"yes":0.24,"lo":66,"hi":67},
                      69:{"yes":0.36,"lo":68,"hi":69},71:{"yes":0.26,"lo":70,"hi":71},
                      73:{"yes":0.05,"lo":72,"hi":73}})
