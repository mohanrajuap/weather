"""
Rendered image charts for Telegram signal cards (revamp).

Self-contained matplotlib (Agg backend) → PNG bytes. Every public function returns
None on ANY failure (missing matplotlib, bad data, render error) so a chart problem
can NEVER break an alert — the caller just skips the photo and the text card still
sends. Nothing here does network I/O.

Two charts:
  • signal_png(p)   — model probability vs market price per degree (the core
                       model-vs-market picture) + a forecast strip underneath.
  • cover_png(res)  — the equal-payout cover: $ per degree and the guaranteed
                       return whichever covered degree settles.
"""
import io

# Flat, Telegram-friendly palette (works on light & dark since it's a baked image
# with its own near-white panel background).
_MODEL   = "#1D9E75"   # teal   — the model
_MARKET  = "#EF9F27"   # amber  — the market
_BLEND   = "#378ADD"   # blue   — model blend line
_LIVE    = "#D85A30"   # coral  — live observation
_INK     = "#26261F"   # near-black text
_MUTE    = "#8A8A80"   # muted text / gridlines
_PANEL   = "#FBFBF8"   # panel background
_GRID    = "#E7E7E0"


def _mpl():
    """Import + configure matplotlib lazily; None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": _GRID,
            "axes.linewidth": 0.8,
            "figure.facecolor": _PANEL,
            "axes.facecolor": _PANEL,
            "savefig.facecolor": _PANEL,
        })
        return plt
    except Exception:
        return None


def _save(fig, plt) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.25, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _deg_label(v, lo, hi, short):
    if lo is not None and hi is not None and lo != hi and -9000 < lo and hi < 9000:
        return f"{lo}-{hi}{short}"
    return f"{v}{short}"


def signal_png(p):
    """Model % vs market ¢ per degree, with a forecast strip. Returns PNG bytes or None."""
    plt = _mpl()
    if plt is None:
        return None
    try:
        return _render_signal(p, plt)
    except Exception:
        return None


def _render_signal(p, plt):
    sym   = p.get("temp_unit", "°C")
    short = sym.replace("°", "")
    dist  = p.get("distribution") or []
    pm    = (p.get("polymarket") or {}).get("buckets") or {}

    # Merge model probability and market price per degree.
    rows = {}
    for b in dist:
        v = b.get("value")
        if v is None:
            continue
        rows[v] = {"model": (b.get("probability") or 0.0) * 100.0, "mkt": 0.0,
                   "lo": b.get("lo", v), "hi": b.get("hi", v)}
    for k, mb in pm.items():
        try:
            kk = int(k)
        except (TypeError, ValueError):
            continue
        y = (mb or {}).get("yes")
        if y is None:
            continue
        r = rows.setdefault(kk, {"model": 0.0, "mkt": 0.0,
                                 "lo": (mb or {}).get("lo", kk), "hi": (mb or {}).get("hi", kk)})
        r["mkt"] = float(y) * 100.0

    items = [(k, v) for k, v in rows.items() if v["model"] >= 1.0 or v["mkt"] >= 2.0]
    items.sort(key=lambda x: x[0])
    if not items:
        return None

    labels = [_deg_label(k, v["lo"], v["hi"], short) for k, v in items]
    model  = [v["model"] for _, v in items]
    market = [v["mkt"] for _, v in items]

    fig, ax = plt.subplots(figsize=(7.4, max(2.8, 0.62 * len(items) + 1.6)))
    ypos = range(len(items))
    h = 0.38
    ax.barh([i + h / 2 for i in ypos], model,  height=h, color=_MODEL,  zorder=3)
    ax.barh([i - h / 2 for i in ypos], market, height=h, color=_MARKET, zorder=3)

    # value labels at the end of each bar
    for i, (m, k) in enumerate(zip(model, market)):
        if m >= 1:
            ax.text(m + 1.2, i + h / 2, f"{m:.0f}%", va="center", ha="left",
                    fontsize=9.5, color=_MODEL, zorder=4)
        if k >= 1:
            ax.text(k + 1.2, i - h / 2, f"{k:.0f}¢", va="center", ha="left",
                    fontsize=9.5, color="#B5760F", zorder=4)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=11, color=_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(100, max(model + market) + 12))
    ax.set_xlabel("model probability  vs  market price", fontsize=10, color=_MUTE)
    ax.tick_params(axis="x", colors=_MUTE, labelsize=9)
    ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Title block: city, verdict, model pick vs market favourite.
    city = (p.get("display") or p.get("city") or "").title()
    pick = p.get("top_bucket")
    prob = (p.get("top_prob") or 0) * 100
    fav  = max(pm, key=lambda k: (pm[k] or {}).get("yes") or 0, default=None)
    try:
        fav = int(fav) if fav is not None else None
    except (TypeError, ValueError):
        fav = None
    favp = ((pm.get(fav) or pm.get(str(fav)) or {}).get("yes") if fav is not None else None)
    sub = f"model {pick}{short} · {prob:.0f}%"
    if fav is not None and favp is not None:
        sub += f"     market {fav}{short} · {favp*100:.0f}¢"
    ax.set_title(f"{city} · {p.get('target_date','')}\n{sub}",
                 fontsize=13, color=_INK, loc="left", pad=10, fontweight="bold")

    # Legend + footer with the blend / live obs (context in one line).
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_MODEL, label="Model"),
                       Patch(color=_MARKET, label="Market")],
              loc="lower right", frameon=False, fontsize=9.5)
    live = p.get("live") or {}
    foot = f"blend {p.get('deb')}{sym}"
    if p.get("deb_raw") is not None and p.get("deb_raw") != p.get("deb"):
        foot += f"  (no-bias {p.get('deb_raw')}{sym})"
    if live.get("max_so_far") is not None:
        foot += f"   ·   live max {live.get('max_so_far')}{sym}"
    fig.text(0.01, -0.02, foot, fontsize=9, color=_MUTE, ha="left")
    return _save(fig, plt)


def cover_png(res, sym="°C"):
    """Equal-payout cover: $ per degree + the return if covered. res = advise_budgets
    result (uses the FIRST budget's cover). Returns PNG bytes or None."""
    plt = _mpl()
    if plt is None:
        return None
    try:
        rows = res.get("rows") or []
        cover = next((r["cover"] for r in rows if r.get("cover") and r["cover"].get("legs")), None)
        if not cover:
            return None
        short = sym.replace("°", "")
        legs  = cover["legs"]
        labels = [f"{l['bucket']}{short}" for l in legs]
        stakes = [l["stake"] for l in legs]
        fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.55 * len(legs) + 1.4)))
        ypos = range(len(legs))
        ax.barh(list(ypos), stakes, color=_MODEL, height=0.55, zorder=3)
        for i, l in enumerate(legs):
            ax.text(l["stake"] + max(stakes) * 0.02, i,
                    f"${l['stake']:.2f} · {l['shares']:g} sh @ {l['price']*100:.0f}¢",
                    va="center", ha="left", fontsize=9.5, color=_INK, zorder=4)
        ax.set_yticks(list(ypos)); ax.set_yticklabels(labels, fontsize=11, color=_INK)
        ax.invert_yaxis()
        ax.set_xlim(0, max(stakes) * 1.5)
        ax.tick_params(axis="x", colors=_MUTE, labelsize=9)
        ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        title = (f"Cover ${cover['total']:.0f} → ${cover['payout']:.2f} back "
                 f"· {cover['mkt_coverage']*100:.0f}% covered · +${cover['profit_if_covered']:.2f}")
        ax.set_title(title, fontsize=12.5, color=_INK, loc="left", pad=10, fontweight="bold")
        return _save(fig, plt)
    except Exception:
        return None
