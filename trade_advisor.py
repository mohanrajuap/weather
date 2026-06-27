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
  FULL COVER : 49% hit · +2.7% ROI · per-bet std ~$18  → effectively CAPITAL
               PRESERVATION (≈ break-even with a slight edge, win ~half the time).
               This is the "don't lose" play and the default recommendation.
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

# ── Calibration from the 361-settled-market analysis ──────────────────────────
EMP_STD       = 1.45
BIAS_OFFSET   = -0.16
DISAGREE_K    = 0.35
SIGMA_FLOOR   = 1.30

# ── Trade controls ────────────────────────────────────────────────────────────
MIN_PRICE     = 0.01
MAX_PRICE     = 0.97
MIN_EDGE      = 0.04     # a degree counts as "under-priced" if honest prob - price >= this
MAX_SUM_PRICE = 0.92     # stop adding cover degrees once their prices sum past this
MAX_LEGS      = 4
MIN_STAKE     = 0.25     # round legs below this away


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


def _equal_payout(bucket_list, priced, dist, budget):
    """Size a set of degrees for EQUAL payout (same money back whichever wins).
    stakeᵢ = payout·priceᵢ where payout = budget / Σ price over the SET. All buckets
    in the set are kept so the math is consistent (Σ stake == budget exactly)."""
    sump = sum(priced[v] for v in bucket_list)
    if sump <= 0:
        return None
    payout = budget / sump                       # same money back on any covered leg
    legs = []
    for v in bucket_list:
        p = priced[v]
        stake = payout * p                       # NOT dropped — keeps Σ stake == budget
        legs.append({"bucket": v, "price": round(p, 3), "prob": round(dist.get(v, 0.0), 3),
                     "edge": round(dist.get(v, 0.0) - p, 3),
                     "stake": round(stake, 2), "shares": round(stake / p, 1)})
    total    = round(sum(l["stake"] for l in legs), 2)
    coverage = round(sum(l["prob"] for l in legs), 3)
    payout   = round(payout, 2)
    return {"legs": legs, "total": total, "payout": payout, "sum_price": round(sump, 3),
            "coverage": coverage, "profit_if_covered": round(payout - total, 2),
            "loss_if_miss": round(-total, 2),
            "ev": round(coverage * payout - total, 2)}


def advise(deb, deb_raw, market, budget=4.0):
    """market: {value: {"yes": price, "lo": lo, "hi": hi}} (lo/hi optional)."""
    if deb is None or not market:
        return {"ok": False, "reason": "no prediction / market"}
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
        if vbuckets and vp + priced[v] > MAX_SUM_PRICE:
            continue
        vbuckets.append(v); vp += priced[v]
    value_bet = _equal_payout(vbuckets, priced, dist, budget) if vbuckets else None

    # ── FULL COVER: bot pick + market favourite + most-likely neighbours ─────────
    order, seen = [], set()
    for v in (bot_pick, market_fav):
        if v not in seen:
            order.append(v); seen.add(v)
    for v in sorted(priced, key=lambda v: -dist.get(v, 0.0)):
        if v not in seen:
            order.append(v); seen.add(v)
    cbuckets, cp = [], 0.0
    for v in order:
        if len(cbuckets) >= MAX_LEGS:
            break
        if cbuckets and cp + priced[v] > MAX_SUM_PRICE:
            continue
        cbuckets.append(v); cp += priced[v]
    full_cover = _equal_payout(cbuckets, priced, dist, budget)

    # ── recommendation ───────────────────────────────────────────────────────
    # Backtest on 361 settled markets (budget $4 each):
    #   FULL COVER : 49% hit · +2.7% ROI · ~capital-preservation (≈break-even+,
    #                the "don't lose" play that matches your goal).
    #   VALUE BET  : 8% hit · +34% ROI but driven by rare big wins → high-variance
    #                lottery, NOT reliable for a small bankroll.
    # So default to the cover; surface the value bet only as an aggressive option.
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
    }


def advise_from_prediction(p, budget=4.0):
    pm = (p.get("polymarket") or {}).get("buckets") or {}
    market = {}
    for v, b in pm.items():
        try:
            market[int(v)] = {"yes": b.get("yes"), "lo": b.get("lo", int(v)), "hi": b.get("hi", int(v))}
        except (TypeError, ValueError):
            continue
    return advise(p.get("deb"), p.get("deb_raw"), market, budget=budget)


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
    line = (f"🎓 <b>Cover</b> (${a['budget']:.0f}): {_legs_str(s, sym)} → "
            f"<b>${s['payout']:.2f}</b> back on any · win if it's "
            + "/".join(f"{l['bucket']}{sym.replace(chr(176),'')}" for l in s['legs']))
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
            L.append(f"   • {l['bucket']}{sym} @ {l['price']*100:.0f}¢ → ${l['stake']:.2f} "
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
            L.append(f"   • {l['bucket']}{sym} @ {l['price']*100:.0f}¢ → ${l['stake']:.2f} "
                     f"(edge {l['edge']*100:+.0f}% {note})")
        L.append(f"   → win any → ${fc['payout']:.2f}  ({fc['coverage']*100:.0f}% covered, "
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
