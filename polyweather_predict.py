"""
polyweather_predict.py  v2.0
============================
Run once → scans ALL 51 cities, shows probabilities + trading signals.

FIXES in v2.0:
  - CORRECT TARGET DATE: after peak hours → automatically predicts TOMORROW
  - Shows which Polymarket date to look for (Jun 13 / Jun 14 etc.)
  - Multi-model fetches correct date index (today vs tomorrow)
  - Ensemble fetches correct date index
  - METAR reads today's observations for live temp / max_so_far
  - Dead market only triggers when predicting TODAY (not tomorrow)
  - All display output shows target date clearly

Usage:
    python polyweather_predict.py                    <- scan all cities
    python polyweather_predict.py tokyo london       <- specific cities
    python polyweather_predict.py --workers 10       <- faster
    python polyweather_predict.py --min-prob 0.5     <- high confidence only
    python polyweather_predict.py --detail istanbul  <- full detail
    python polyweather_predict.py --json             <- raw JSON
    python polyweather_predict.py --list             <- list cities
    python polyweather_predict.py --record-actual istanbul 2026-06-13 20

Requirements:
    pip install httpx requests python-dotenv
"""

import sys
import os
import math
import json
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any, Tuple

import os as _os_for_bias
# Default upward nudge applied to the model blend when no history exists yet.
# Numerical weather models systematically under-predict the daily MAX because
# they smooth the afternoon peak. Override with env PEAK_BIAS (e.g. "0.4").
DEFAULT_PEAK_BIAS = float(_os_for_bias.environ.get("PEAK_BIAS", "0.3"))

# Manual per-city bias overrides (°), e.g. CITY_BIAS="manila:2.0,karachi:0.5".
# When a city is listed here it REPLACES the learned/default bias for that city —
# use it after /history shows the bot is consistently off for a city. Backed up to
# GitHub via the learning data path is the *history*; this override lives in env.
def _parse_city_bias() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in _os_for_bias.environ.get("CITY_BIAS", "").split(","):
        if ":" in part:
            name, val = part.split(":", 1)
            try:
                out[name.strip().lower()] = float(val)
            except Exception:
                pass
    return out
CITY_BIAS = _parse_city_bias()

# A Polymarket bucket priced at/above this YES means the market has effectively
# DECIDED the outcome. If our model disagrees with an already-decided market,
# the model is almost certainly the one that's wrong (it's predicting a peak the
# market has already priced past) — suppress those signals entirely.
MARKET_DECIDED_YES = float(_os_for_bias.environ.get("MARKET_DECIDED_YES", "0.90"))
# A softer band: market leans hard one way but isn't fully locked. If the model
# disagrees here, downgrade a TRADE to WAIT rather than firing a buy.
MARKET_LEAN_YES    = float(_os_for_bias.environ.get("MARKET_LEAN_YES", "0.75"))

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL KEYED FORECAST SOURCES — configure in Railway variables
# ──────────────────────────────────────────────────────────────────────────────
# Each of these weather APIs is FREE-tier but needs an API key. Set the matching
# env var in Railway and that source automatically joins the blend for every city
# (more independent models → tighter σ, better agreement, better trades). Leave a
# key unset and that source is simply skipped. All values are auto-converted to
# each city's settlement unit (°C/°F) via _convert_temp, so you never have to
# worry about which unit an API returns.
#
#   OPENWEATHER_API_KEY   → OpenWeatherMap   (free 1M/mo)  openweathermap.org/api
#   WEATHERAPI_KEY        → WeatherAPI.com   (free 1M/mo)  weatherapi.com
#   VISUALCROSSING_KEY    → Visual Crossing  (free 1k/day) visualcrossing.com
#   TOMORROW_API_KEY      → Tomorrow.io      (free ~500/day) tomorrow.io
# ══════════════════════════════════════════════════════════════════════════════
OPENWEATHER_API_KEY = _os_for_bias.environ.get("OPENWEATHER_API_KEY", "").strip()
WEATHERAPI_KEY      = _os_for_bias.environ.get("WEATHERAPI_KEY", "").strip()
VISUALCROSSING_KEY  = _os_for_bias.environ.get("VISUALCROSSING_KEY", "").strip()
TOMORROW_API_KEY    = _os_for_bias.environ.get("TOMORROW_API_KEY", "").strip()

# ── Edge & learning knobs (all configurable in Railway) ───────────────────────
# Minimum model-vs-market edge (probability points) to call a bucket tradeable.
# Raise it to be MORE selective — fewer but higher-quality trades. Default 10%.
EDGE_MIN     = float(_os_for_bias.environ.get("EDGE_MIN", "0.10"))
# DEB learner memory: how many past days each source's bias is learned from, and
# how fast older days fade (0-1; higher = longer memory). More days = steadier
# learning once enough history exists.
DEB_LOOKBACK = int(_os_for_bias.environ.get("DEB_LOOKBACK", "10"))
DEB_DECAY    = float(_os_for_bias.environ.get("DEB_DECAY", "0.85"))

# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS & SOURCE TOGGLES — nothing is hard-coded; everything below is a
# Railway-overridable variable that simply DEFAULTS to the current value. Turn a
# source off with ENABLE_<SOURCE>=0, or point it at a new URL if an API changes.
# ══════════════════════════════════════════════════════════════════════════════
def _env(name: str, default: str) -> str:
    v = _os_for_bias.environ.get(name)
    return v.strip() if v else default

def _env_bool(name: str, default: bool = True) -> bool:
    v = _os_for_bias.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

# ── Endpoints (override only if an API changes its URL) ───────────────────────
OPEN_METEO_URL          = _env("OPEN_METEO_URL",          "https://api.open-meteo.com/v1/forecast")
OPEN_METEO_ENSEMBLE_URL = _env("OPEN_METEO_ENSEMBLE_URL", "https://ensemble-api.open-meteo.com/v1/ensemble")
METNO_URL               = _env("METNO_URL",               "https://api.met.no/weatherapi/locationforecast/2.0/compact")
NWS_POINTS_URL          = _env("NWS_POINTS_URL",          "https://api.weather.gov/points")
SEVENTIMER_URL          = _env("SEVENTIMER_URL",          "https://www.7timer.info/bin/api.pl")
OPENWEATHER_URL         = _env("OPENWEATHER_URL",         "https://api.openweathermap.org/data/2.5/forecast")
WEATHERAPI_URL          = _env("WEATHERAPI_URL",          "https://api.weatherapi.com/v1/forecast.json")
VISUALCROSSING_URL      = _env("VISUALCROSSING_URL",      "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline")
TOMORROW_URL            = _env("TOMORROW_URL",            "https://api.tomorrow.io/v4/weather/forecast")
METAR_URL               = _env("METAR_URL",               "https://aviationweather.gov/api/data/metar")
WUNDERGROUND_URL        = _env("WUNDERGROUND_URL",        "https://api.weather.com/v1/location")
GAMMA = _env("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")
CLOB  = _env("POLYMARKET_CLOB_URL",  "https://clob.polymarket.com")
DATA  = _env("POLYMARKET_DATA_URL",  "https://data-api.polymarket.com")
# Override Gamma's stale last-trade prices with the LIVE CLOB order book (one batch
# call per market). Set 0 to fall back to Gamma's outcomePrices.
LIVE_CLOB_PRICES = _env("LIVE_CLOB_PRICES", "1") == "1"

# ── Per-source on/off (free no-key sources default ON; keyed ones turn on when
#    their API key is set). Flip any to 0 in Railway to drop that source. ──────
# Trade on the RAW (no-bias) blend — ignore the peak/learned/CITY bias for the
# actual decision (verdict, edges, best trade). The bias values are still shown
# for reference. Set USE_NOBIAS=1 if you trust the raw model over the bias.
USE_NOBIAS        = _env_bool("USE_NOBIAS", False)
ENABLE_OPEN_METEO = _env_bool("ENABLE_OPEN_METEO", True)
ENABLE_ENSEMBLE   = _env_bool("ENABLE_ENSEMBLE",   True)
ENABLE_MULTIMODEL = _env_bool("ENABLE_MULTIMODEL", True)
ENABLE_METNO      = _env_bool("ENABLE_METNO",      True)
ENABLE_NWS        = _env_bool("ENABLE_NWS",        True)
ENABLE_7TIMER     = _env_bool("ENABLE_7TIMER",     True)

# How far (in °C) a source may sit from the consensus before it's treated as
# suspect — likely a broken API or the wrong unit. Raise to be more lenient.
SOURCE_OUTLIER_TOL = float(_os_for_bias.environ.get("SOURCE_OUTLIER_TOL", "8.0"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import httpx
except ImportError:
    print("ERROR: pip install httpx requests")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CITY REGISTRY — 51 cities
# key fields: lat/lon, icao, tz (UTC offset seconds), f (Fahrenheit?),
#             settlement source, peak_end (local hour when day's max is done)
# ══════════════════════════════════════════════════════════════════════════════
CITIES = {
    "ankara":        {"lat": 40.1281, "lon": 32.9951,   "icao": "LTAC",  "tz": 10800,  "f": False, "settlement": "metar", "peak_end": 17},
    "istanbul":      {"lat": 41.2749, "lon": 28.7323,   "icao": "LTFM",  "tz": 10800,  "f": False, "settlement": "noaa",  "peak_end": 17},
    "moscow":        {"lat": 55.5915, "lon": 37.2615,   "icao": "UUWW",  "tz": 10800,  "f": False, "settlement": "metar", "peak_end": 17},
    "london":        {"lat": 51.5048, "lon": 0.0522,    "icao": "EGLC",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 17},
    "paris":         {"lat": 48.9694, "lon": 2.4414,    "icao": "LFPB",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 17},
    "munich":        {"lat": 48.3538, "lon": 11.7861,   "icao": "EDDM",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 17},
    "milan":         {"lat": 45.6306, "lon": 8.7231,    "icao": "LIMC",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 17},
    "warsaw":        {"lat": 52.1672, "lon": 20.9679,   "icao": "EPWA",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 17},
    "madrid":        {"lat": 40.4719, "lon": -3.5626,   "icao": "LEMD",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 18},
    "tel aviv":      {"lat": 32.0055, "lon": 34.8854,   "icao": "LLBG",  "tz": 7200,   "f": False, "settlement": "metar", "peak_end": 16},
    "amsterdam":     {"lat": 52.3105, "lon": 4.7683,    "icao": "EHAM",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 17},
    "helsinki":      {"lat": 60.3183, "lon": 24.9497,   "icao": "EFHK",  "tz": 7200,   "f": False, "settlement": "metar", "peak_end": 17},
    "lagos":         {"lat": 6.5774,  "lon": 3.3212,    "icao": "DNMM",  "tz": 3600,   "f": False, "settlement": "metar", "peak_end": 15},
    "cape town":     {"lat": -33.9648,"lon": 18.6017,   "icao": "FACT",  "tz": 7200,   "f": False, "settlement": "metar", "peak_end": 16},
    "jeddah":        {"lat": 21.6796, "lon": 39.1565,   "icao": "OEJN",  "tz": 10800,  "f": False, "settlement": "metar", "peak_end": 15},
    "seoul":         {"lat": 37.4602, "lon": 126.4407,  "icao": "RKSI",  "tz": 32400,  "f": False, "settlement": "metar", "peak_end": 16},
    "busan":         {"lat": 35.1796, "lon": 128.9380,  "icao": "RKPK",  "tz": 32400,  "f": False, "settlement": "metar", "peak_end": 16},
    "hong kong":     {"lat": 22.3019, "lon": 114.1742,  "icao": "VHHH",  "tz": 28800,  "f": False, "settlement": "hko", "obs": "hko", "peak_end": 15},
    "taipei":        {"lat": 25.0777, "lon": 121.5737,  "icao": "RCSS",  "tz": 28800,  "f": False, "settlement": "cwa",   "peak_end": 15},
    "shanghai":      {"lat": 31.1443, "lon": 121.8083,  "icao": "ZSPD",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "beijing":       {"lat": 40.0799, "lon": 116.5847,  "icao": "ZBAA",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "qingdao":       {"lat": 36.2661, "lon": 120.3744,  "icao": "ZSQD",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "wuhan":         {"lat": 30.7838, "lon": 114.2080,  "icao": "ZHHH",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "chengdu":       {"lat": 30.5785, "lon": 103.9470,  "icao": "ZUUU",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "chongqing":     {"lat": 29.7192, "lon": 106.6423,  "icao": "ZUCK",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "shenzhen":      {"lat": 22.6395, "lon": 113.8105,  "icao": "ZGSZ",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "guangzhou":     {"lat": 23.3924, "lon": 113.2988,  "icao": "ZGGG",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "singapore":     {"lat": 1.3644,  "lon": 103.9915,  "icao": "WSSS",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "tokyo":         {"lat": 35.5523, "lon": 139.7798,  "icao": "RJTT",  "tz": 32400,  "f": False, "settlement": "metar", "peak_end": 16},
    "kuala lumpur":  {"lat": 2.7456,  "lon": 101.7099,  "icao": "WMKK",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "jakarta":       {"lat": -6.1275, "lon": 106.6537,  "icao": "WIII",  "tz": 25200,  "f": False, "settlement": "metar", "peak_end": 15},
    "manila":        {"lat": 14.5086, "lon": 121.0194,  "icao": "RPLL",  "tz": 28800,  "f": False, "settlement": "metar", "peak_end": 15},
    "wellington":    {"lat": -41.3272,"lon": 174.8051,  "icao": "NZWN",  "tz": 46800,  "f": False, "settlement": "metar", "peak_end": 17},
    "toronto":       {"lat": 43.6777, "lon": -79.6248,  "icao": "CYYZ",  "tz": -18000, "f": False, "settlement": "metar", "peak_end": 16},
    "new york":      {"lat": 40.7769, "lon": -73.8740,  "icao": "KLGA",  "tz": -18000, "f": True,  "settlement": "metar", "peak_end": 16},  # Polymarket settles on LaGuardia, not JFK
    "los angeles":   {"lat": 33.9425, "lon": -118.4081, "icao": "KLAX",  "tz": -28800, "f": True,  "settlement": "metar", "peak_end": 16},
    "san francisco": {"lat": 37.6213, "lon": -122.3790, "icao": "KSFO",  "tz": -28800, "f": True,  "settlement": "metar", "peak_end": 16},
    "aurora":        {"lat": 39.8561, "lon": -104.6737, "icao": "KBKF",  "tz": -25200, "f": True,  "settlement": "metar", "peak_end": 17},
    "austin":        {"lat": 30.1975, "lon": -97.6664,  "icao": "KAUS",  "tz": -21600, "f": True,  "settlement": "metar", "peak_end": 17},
    "houston":       {"lat": 29.6454, "lon": -95.2789,  "icao": "KHOU",  "tz": -21600, "f": True,  "settlement": "metar", "peak_end": 17},  # Polymarket settles on Hobby, not Bush/IAH
    "chicago":       {"lat": 41.9742, "lon": -87.9073,  "icao": "KORD",  "tz": -21600, "f": True,  "settlement": "metar", "peak_end": 16},
    "dallas":        {"lat": 32.8481, "lon": -96.8512,  "icao": "KDAL",  "tz": -21600, "f": True,  "settlement": "metar", "peak_end": 17},  # Polymarket settles on Love Field, not DFW
    "miami":         {"lat": 25.7959, "lon": -80.2870,  "icao": "KMIA",  "tz": -18000, "f": True,  "settlement": "metar", "peak_end": 16},
    "atlanta":       {"lat": 33.6407, "lon": -84.4277,  "icao": "KATL",  "tz": -18000, "f": True,  "settlement": "metar", "peak_end": 16},
    "seattle":       {"lat": 47.4502, "lon": -122.3088, "icao": "KSEA",  "tz": -28800, "f": True,  "settlement": "metar", "peak_end": 17},
    "mexico city":   {"lat": 19.4361, "lon": -99.0719,  "icao": "MMMX",  "tz": -21600, "f": False, "settlement": "metar", "peak_end": 15},
    "buenos aires":  {"lat": -34.8222,"lon": -58.5358,  "icao": "SAEZ",  "tz": -10800, "f": False, "settlement": "metar", "peak_end": 16},
    "sao paulo":     {"lat": -23.4356,"lon": -46.4731,  "icao": "SBGR",  "tz": -10800, "f": False, "settlement": "metar", "peak_end": 15},
    "panama city":   {"lat": 8.9733,  "lon": -79.5556,  "icao": "MPMG",  "tz": -18000, "f": False, "settlement": "metar", "peak_end": 15},  # Polymarket settles on Albrook/Marcos Gelabert, not Tocumen
    "lucknow":       {"lat": 26.7606, "lon": 80.8893,   "icao": "VILK",  "tz": 19800,  "f": False, "settlement": "metar", "peak_end": 16},
    "karachi":       {"lat": 24.9008, "lon": 67.1681,   "icao": "OPKC",  "tz": 18000,  "f": False, "settlement": "metar", "wu_station": "OPMR", "peak_end": 16},  # PM settles on Masroor (WU); OPKC kept for METAR/forecast fallback (Masroor has no METAR)
}

# ══════════════════════════════════════════════════════════════════════════════
# CITY ALIASES — handles spacing, abbreviations, common spellings
# So "hongkong", "hong kong", "hk", "HONGKONG" all resolve to "hong kong"
# ══════════════════════════════════════════════════════════════════════════════
ALIASES = {
    "hongkong": "hong kong", "hk": "hong kong",
    "newyork": "new york", "nyc": "new york", "ny": "new york",
    "losangeles": "los angeles", "la": "los angeles",
    "sanfrancisco": "san francisco", "sf": "san francisco",
    "telaviv": "tel aviv",
    "capetown": "cape town",
    "kualalumpur": "kuala lumpur", "kl": "kuala lumpur",
    "mexicocity": "mexico city",
    "buenosaires": "buenos aires", "ba": "buenos aires",
    "saopaulo": "sao paulo", "sãopaulo": "sao paulo",
    "panamacity": "panama city",
    "laufaushan": "lau fau shan",
    "istanbul": "istanbul", "ist": "istanbul",
    "moscow": "moscow", "mos": "moscow",
}

def resolve_city(name: str) -> Optional[str]:
    """Normalize a user-typed city name to a registry key."""
    if not name:
        return None
    raw = name.lower().strip()
    # direct match
    if raw in CITIES:
        return raw
    # alias match (handles 'hongkong', 'nyc', etc.)
    nospace = raw.replace(" ", "").replace("_", "").replace("-", "")
    if nospace in ALIASES:
        return ALIASES[nospace]
    if raw in ALIASES:
        return ALIASES[raw]
    # try matching ignoring spaces against registry keys
    for key in CITIES:
        if key.replace(" ", "") == nospace:
            return key
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SETTLEMENT ROUNDING
# ══════════════════════════════════════════════════════════════════════════════
def wu_round(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))

def settlement_round(city: str, value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    meta = CITIES.get(city.lower().strip(), {})
    if meta.get("settlement") == "hko":
        return int(math.floor(float(value)))
    return wu_round(float(value))

# ══════════════════════════════════════════════════════════════════════════════
# HTTP
# ══════════════════════════════════════════════════════════════════════════════
_SESSION = httpx.Client(
    timeout=12.0,
    follow_redirects=True,
    headers={"User-Agent": "PolyWeatherPredict/2.0 (+https://polyweather.top)"},
    limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
)

# ── Response cache ────────────────────────────────────────────────────────────
# The monitor calls predict() far more often than the data actually changes:
# the 20-min scan PLUS the position-watch, signal-watch and position-update loops
# all re-fetch the same Open-Meteo forecasts for the same cities. Open-Meteo only
# refreshes hourly, so without caching we hammer the free quota and get the 429
# storm seen in the logs. Cache successful GETs for HTTP_CACHE_TTL seconds, keyed
# on URL+params, and on a 429 serve the last cached value (even if stale) so a
# rate-limit degrades gracefully instead of nuking the signal.
import time as _time
_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = float(os.environ.get("HTTP_CACHE_TTL", "1800"))  # 30 min default —
# bridges the 20-min scan interval so consecutive scans reuse forecasts instead
# of re-hitting Open-Meteo; daily-max forecasts barely move within half an hour.
_CACHE_LOCK = threading.Lock()

# ── Circuit breaker ───────────────────────────────────────────────────────────
# When a host's quota is fully exhausted, EVERY call returns 429 — so the cache
# (which only stores successes) never helps, and a single scan fires ~150 doomed
# Open-Meteo calls. After RATE_LIMIT_TRIP consecutive 429s from a host we "open
# the circuit": stop calling that host for RATE_LIMIT_COOLDOWN seconds and serve
# cache instead. This is per-host, so Met.no / METAR / Polymarket are unaffected,
# and it auto-closes the moment a real request to that host succeeds again.
_CB_LOCK = threading.Lock()
_CB_FAILS: Dict[str, int]   = {}     # host -> consecutive 429 count
_CB_UNTIL: Dict[str, float] = {}     # host -> blocked-until epoch seconds
_CB_TRIP     = int(os.environ.get("RATE_LIMIT_TRIP", "3"))
_CB_COOLDOWN = float(os.environ.get("RATE_LIMIT_COOLDOWN", "600"))   # 10 min

def _cache_key(url: str, params: dict) -> str:
    return url + "?" + json.dumps(params or {}, sort_keys=True, default=str)

def _cb_fail(host: str, now: float, why: str) -> bool:
    """Record a failure against a host; open the breaker past the threshold.
    Returns True if THIS failure tripped the breaker (so the caller can log it)."""
    with _CB_LOCK:
        _CB_FAILS[host] = _CB_FAILS.get(host, 0) + 1
        fails   = _CB_FAILS[host]
        tripped = fails >= _CB_TRIP and now >= _CB_UNTIL.get(host, 0.0)
        if tripped:
            _CB_UNTIL[host] = now + _CB_COOLDOWN
    if tripped:
        print(f"[http] ⛔ circuit breaker OPEN for {host} after {fails} {why} "
              f"— pausing calls for {int(_CB_COOLDOWN)}s (running on cache / fallbacks)")
    return tripped

def _cb_ok(host: str):
    """A success closes the breaker for this host."""
    with _CB_LOCK:
        if _CB_FAILS.pop(host, 0):
            _CB_UNTIL.pop(host, None)

def _get(url: str, params: dict = None, timeout: float = 10.0,
         cache_ttl: float = None) -> Optional[Any]:
    key  = _cache_key(url, params)
    host = httpx.URL(url).host
    now  = _time.time()
    ttl  = _CACHE_TTL if cache_ttl is None else cache_ttl   # caller may want fresher
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    # Circuit open for this host? Skip the doomed call entirely, serve cache.
    with _CB_LOCK:
        blocked = now < _CB_UNTIL.get(host, 0.0)
    if blocked:
        return hit[1] if hit else None
    try:
        r = _SESSION.get(url, params=params or {}, timeout=timeout)
        # 429 (rate limit) and 5xx (server error / outage) both count toward the
        # breaker — a host that's down or throttling shouldn't be hammered 50×/scan.
        if r.status_code == 429:
            if not _cb_fail(host, now, "rate-limits"):
                print(f"[http] 429 RATE LIMITED by {host} — quota likely exhausted"
                      f"{' (serving cached)' if hit else ''}")
            return hit[1] if hit else None
        if r.status_code >= 500:
            if not _cb_fail(host, now, "server errors"):
                print(f"[http] {r.status_code} from {host}"
                      f"{' (serving cached)' if hit else ''}")
            return hit[1] if hit else None
        r.raise_for_status()
        data = r.json()
        with _CACHE_LOCK:
            _CACHE[key] = (now, data)
        _cb_ok(host)               # success → close the breaker
        return data
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        # 404 is benign (e.g. NWS has no grid for non-US cities) — don't log it.
        if code != 404:
            print(f"[http] {code} from {host}")
        return hit[1] if hit else None
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
        # Timeouts/connection failures count too — this is what made the broken
        # Polymarket search hang the scan one city at a time.
        if not _cb_fail(host, now, "timeouts"):
            print(f"[http] timeout/conn error from {host}")
        return hit[1] if hit else None
    except Exception:
        return hit[1] if hit else None

def _post(url: str, json_body, timeout: float = 10.0,
          cache_ttl: float = None) -> Optional[Any]:
    """POST with the same cache + circuit-breaker behaviour as _get (used for the
    CLOB batch price endpoint)."""
    key  = _cache_key(url, {"__body__": json.dumps(json_body, sort_keys=True)[:500]})
    host = httpx.URL(url).host
    now  = _time.time()
    ttl  = _CACHE_TTL if cache_ttl is None else cache_ttl
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    with _CB_LOCK:
        blocked = now < _CB_UNTIL.get(host, 0.0)
    if blocked:
        return hit[1] if hit else None
    try:
        r = _SESSION.post(url, json=json_body, timeout=timeout)
        if r.status_code == 429:
            _cb_fail(host, now, "rate-limits"); return hit[1] if hit else None
        if r.status_code >= 500:
            _cb_fail(host, now, "server errors"); return hit[1] if hit else None
        r.raise_for_status()
        data = r.json()
        with _CACHE_LOCK:
            _CACHE[key] = (now, data)
        _cb_ok(host)
        return data
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
        _cb_fail(host, now, "timeouts"); return hit[1] if hit else None
    except Exception:
        return hit[1] if hit else None

def _sf(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def _convert_temp(value: Optional[float], src_unit: str, dst_is_fahrenheit: bool) -> Optional[float]:
    """Convert a temperature from a source's native unit to the CITY's unit.

    Every forecast source reports in either °C or °F. Cities settle in different
    units (CITIES[..]['f']), so each source's value must be converted to match
    before it goes into the blend — otherwise a US city in °F would be averaged
    against a °C reading and the model would be wildly wrong. `src_unit` is "C"
    or "F"; `dst_is_fahrenheit` is the city's settlement unit.
    """
    if value is None:
        return None
    src_is_f = str(src_unit).strip().upper().startswith("F")
    if src_is_f and not dst_is_fahrenheit:
        return round((value - 32.0) * 5.0 / 9.0, 1)      # F → C
    if (not src_is_f) and dst_is_fahrenheit:
        return round(value * 9.0 / 5.0 + 32.0, 1)        # C → F
    return round(value, 1)                                # already matching

def _series_daily_max(points, target_date: Optional[str], tz_offset: int,
                      time_fn, temp_fn, src_unit: str = "C",
                      use_fahrenheit: bool = False) -> Optional[float]:
    """Daily MAX from a sub-daily forecast series, converted to the city's unit.

    Shared by every source that returns timestamped points (Met.no, OpenWeather,
    7Timer): walk the points, keep those whose LOCAL date matches target_date,
    take the max temperature, convert from src_unit to the city's unit.
      time_fn(point) -> naive UTC datetime (or None to skip)
      temp_fn(point) -> temperature in src_unit (or None to skip)
    """
    highs: List[float] = []
    for pt in points or []:
        dt = time_fn(pt)
        if dt is None:
            continue
        local_d = (dt + timedelta(seconds=tz_offset)).strftime("%Y-%m-%d")
        if target_date and local_d != target_date:
            continue
        t = _sf(temp_fn(pt))
        if t is not None:
            highs.append(t)
    if not highs:
        return None
    return _convert_temp(max(highs), src_unit, use_fahrenheit)

def _vet_forecasts(forecasts: Dict[str, float], use_fahrenheit: bool):
    """Validate every source's forecast before it enters the blend.

    This is the safety net for configurable/new APIs: a source can return garbage
    or the WRONG UNIT (°F where we expected °C). For each value we:
      1. drop it if it's missing / not a finite number,
      2. drop it if it's outside a plausible temperature range (bad API),
      3. if it's a big outlier vs the consensus of the other sources BUT a C↔F
         flip brings it in line → AUTO-CORRECT it and warn (wrong-unit detection),
      4. otherwise drop it and warn that the API looks broken.
    Returns (clean_forecasts, warnings). The blend only ever sees clean values.
    """
    warnings: List[str] = []
    sym = "°F" if use_fahrenheit else "°C"
    # Plausible daily-max range expressed in the CITY's unit (≈ -70..60 °C).
    lo, hi = (-94.0, 140.0) if use_fahrenheit else (-70.0, 60.0)
    def plausible(x: float) -> bool:
        return lo <= x <= hi
    def as_other_unit(x: float) -> float:
        # Re-interpret x as the OTHER unit and express it in the city's unit:
        #   °C city → x was probably °F  → convert F→C
        #   °F city → x was probably °C  → convert C→F
        return (x - 32.0) * 5.0 / 9.0 if not use_fahrenheit else x * 9.0 / 5.0 + 32.0

    # Pass 1: collect well-formed numbers; anchors are those already plausible.
    prelim: Dict[str, float] = {}
    anchors: List[float] = []
    for label, v in forecasts.items():
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            warnings.append(f"{label}: no/invalid value — dropped")
            continue
        v = float(v)
        prelim[label] = v
        if plausible(v):
            anchors.append(v)

    consensus = sorted(anchors)[len(anchors) // 2] if len(anchors) >= 3 else None
    tol       = SOURCE_OUTLIER_TOL * (9.0 / 5.0 if use_fahrenheit else 1.0)

    # Pass 2: keep / auto-correct / drop each source.
    clean: Dict[str, float] = {}
    for label, v in prelim.items():
        # Not enough anchors to judge outliers — keep what's in range, drop the rest.
        if consensus is None:
            if plausible(v):
                clean[label] = round(v, 1)
            else:
                warnings.append(f"{label}: {v}{sym} out of plausible range — dropped (bad API?)")
            continue
        if plausible(v) and abs(v - consensus) <= tol:
            clean[label] = round(v, 1)
            continue
        flipped = as_other_unit(v)                    # maybe it's the wrong unit
        if plausible(flipped) and abs(flipped - consensus) <= tol:
            clean[label] = round(flipped, 1)
            warnings.append(f"{label}: {v}{sym} looked like the wrong unit "
                            f"— auto-corrected to {clean[label]}{sym}")
        elif plausible(v):
            warnings.append(f"{label}: {v}{sym} is {abs(v-consensus):.0f}° from consensus "
                            f"{consensus:.0f}{sym} — dropped (API may be broken/wrong unit)")
        else:
            warnings.append(f"{label}: {v}{sym} out of plausible range — dropped (bad API?)")
    return clean, warnings

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ══════════════════════════════════════════════════════════════════════════════
# TARGET DATE LOGIC  ← THE CORE FIX
#
# If it's past the city's peak_end hour locally, today's max is already done.
# We predict TOMORROW instead — which is what Polymarket's open market is for.
# ══════════════════════════════════════════════════════════════════════════════
def resolve_target(city_key: str) -> Dict[str, Any]:
    """
    Returns:
        target_date  : "YYYY-MM-DD" in city local time — the market date
        target_idx   : 0=today, 1=tomorrow in Open-Meteo daily array
        predicting   : "today" or "tomorrow"
        local_now    : current local datetime
        local_hour   : current local hour
        is_after_peak: True if today's max is already settled
    """
    meta       = CITIES[city_key]
    tz         = meta["tz"]
    peak_end   = meta.get("peak_end", 17)

    local_now  = _now_utc() + timedelta(seconds=tz)
    local_hour = local_now.hour

    # After peak_end → today is done → predict tomorrow
    if local_hour >= peak_end:
        target_dt   = local_now + timedelta(days=1)
        target_idx  = 1
        predicting  = "tomorrow"
        is_after    = True
    else:
        target_dt   = local_now
        target_idx  = 0
        predicting  = "today"
        is_after    = False

    return {
        "target_date":   target_dt.strftime("%Y-%m-%d"),
        "target_idx":    target_idx,
        "predicting":    predicting,
        "local_now":     local_now,
        "local_hour":    local_hour,
        "local_date":    local_now.strftime("%Y-%m-%d"),
        "is_after_peak": is_after,
        "peak_end":      peak_end,
    }

# ══════════════════════════════════════════════════════════════════════════════
# TIMING ADVISOR — is NOW a good time to predict this city?
#
# Golden window = morning of the target day, local time:
#   models ran overnight, METAR starting, peak still hours away (market cheap).
#
#   06:00–11:00 local  → GOLDEN  (best edge)
#   11:00–peak_end     → GOOD    (market tightening)
#   peak_end–22:00     → TOMORROW (long lead, less sharp — recheck AM)
#   22:00–06:00        → EARLY   (overnight, wait for morning run)
# ══════════════════════════════════════════════════════════════════════════════
def _user_tz_offset_seconds() -> int:
    """User's local UTC offset. Override with POLYWEATHER_USER_TZ_OFFSET (seconds)."""
    env = os.environ.get("POLYWEATHER_USER_TZ_OFFSET")
    if env is not None:
        try:
            return int(env)
        except Exception:
            pass
    try:
        local = datetime.now().astimezone()
        return int(local.utcoffset().total_seconds())
    except Exception:
        return 0

def timing_advice(city_key: str, is_tomorrow: bool = False) -> Dict[str, Any]:
    """
    Whether NOW gives a RELIABLE signal, based on how much of the day's
    peak has actually been observed.

    If is_tomorrow=True, the target market's peak is ~a day away — the signal
    is ALWAYS forecast-only and never 'reliable' yet, regardless of clock.
    """
    meta      = CITIES[city_key]
    tz        = meta["tz"]
    peak_end  = meta.get("peak_end", 17)
    peak_start = max(11, peak_end - 3)

    local_now = _now_utc() + timedelta(seconds=tz)
    h         = local_now.hour + local_now.minute / 60.0

    # ── Tomorrow's market: peak is ~a day out, never reliable yet ──
    if is_tomorrow:
        return {
            "quality":             "FORECAST",
            "ok_to_trade":         False,
            "reliable":            False,
            "message":             ("Tomorrow's market — its peak is ~a day away. "
                                    "Forecast only; recheck during tomorrow's peak window."),
            "city_local_now":      local_now.strftime("%H:%M"),
            "peak_window":         f"{peak_start:02d}:00–{peak_end:02d}:00 (tomorrow)",
            "hours_until_golden":  None,
            "next_run_city_local": f"{peak_start:02d}:00 tomorrow",
            "next_run_user_local": None,
        }

    if h < 6:
        quality, ok, reliable = "OVERNIGHT", False, False
        msg = "Overnight — models not refreshed, no live obs yet. Wait."
        hint = 6
    elif h < peak_start:
        quality, ok, reliable = "SPECULATIVE", False, False
        msg = (f"Pre-peak ({local_now.strftime('%H:%M')}). Forecast only — the day's high "
               f"hasn't formed yet. A 70% now can still flip. Wait until ~{peak_start}:00.")
        hint = peak_start
    elif h < peak_end:
        quality, ok, reliable = "FIRMING", True, False
        msg = (f"Peak window ({peak_start}:00–{peak_end}:00). High is forming now — "
               f"signal firming up. Tradeable, but watch live obs.")
        hint = None
    elif h < 22:
        quality, ok, reliable = "RELIABLE", True, True
        msg = (f"Post-peak ({local_now.strftime('%H:%M')}). Day's high is essentially "
               f"locked — most reliable window. Settles ~next morning.")
        hint = None
    else:
        quality, ok, reliable = "OVERNIGHT", False, False
        msg = "Late night — today is done. Tomorrow's market is forecast-only until its peak."
        hint = peak_start

    next_city, next_user, hours_until = None, None, None
    if hint is not None:
        target_local = local_now.replace(hour=int(hint), minute=0, second=0, microsecond=0)
        if target_local <= local_now:
            target_local += timedelta(days=1)
        target_utc  = target_local - timedelta(seconds=tz)
        hours_until = round((target_utc - _now_utc()).total_seconds() / 3600.0, 1)
        next_city   = target_local.strftime("%H:%M %a")
        user_tz     = _user_tz_offset_seconds()
        next_user   = (target_utc + timedelta(seconds=user_tz)).strftime("%H:%M %a")

    return {
        "quality":             quality,       # OVERNIGHT/SPECULATIVE/FIRMING/RELIABLE
        "ok_to_trade":         ok,
        "reliable":            reliable,      # True only post-peak
        "message":             msg,
        "city_local_now":      local_now.strftime("%H:%M"),
        "peak_window":         f"{peak_start:02d}:00–{peak_end:02d}:00",
        "hours_until_golden":  hours_until,
        "next_run_city_local": next_city,
        "next_run_user_local": next_user,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES  — all accept target_idx to pick correct day
# ══════════════════════════════════════════════════════════════════════════════
def fetch_open_meteo(lat: float, lon: float,
                     use_fahrenheit: bool = False,
                     target_idx: int = 0) -> Optional[Dict]:
    """Fetch forecast. Returns today's max (idx=0) or tomorrow's (idx=1)."""
    unit = "fahrenheit" if use_fahrenheit else "celsius"
    data = _get(OPEN_METEO_URL, {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "hourly": "temperature_2m",
        "current": "temperature_2m",
        "temperature_unit": unit,
        "timezone": "UTC",
        "forecast_days": 3,
    }, timeout=8.0)
    if not data:
        return None
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    if target_idx < len(highs) and highs[target_idx] is not None:
        data["_target_max"] = round(float(highs[target_idx]), 1)
        data["_target_date"] = dates[target_idx] if target_idx < len(dates) else None
    return data

def fetch_ensemble(lat: float, lon: float,
                   use_fahrenheit: bool = False,
                   target_idx: int = 0) -> Optional[Dict]:
    """Fetch ensemble spread for target day."""
    unit = "fahrenheit" if use_fahrenheit else "celsius"
    data = _get(OPEN_METEO_ENSEMBLE_URL, {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": unit,
        "timezone": "UTC",
        "models": "icon_seamless",
        "forecast_days": 3,
    }, timeout=8.0)
    if not data:
        return None
    daily   = data.get("daily", {})
    members = {k: v for k, v in daily.items() if k.startswith("temperature_2m_max_member")}
    if not members:
        return None
    vals = sorted(
        float(v[target_idx]) for v in members.values()
        if isinstance(v, list) and target_idx < len(v) and v[target_idx] is not None
    )
    if not vals:
        return None
    n = len(vals)
    return {
        "p10":     round(vals[max(0, int(n * 0.10))], 1),
        "median":  round(vals[n // 2], 1),
        "p90":     round(vals[min(n - 1, int(n * 0.90))], 1),
        "members": n,
    }

def fetch_multi_model(lat: float, lon: float,
                      use_fahrenheit: bool = False,
                      target_idx: int = 0) -> Optional[Dict[str, float]]:
    """Fetch individual NWP model forecasts for target day.

    All four models are requested in a SINGLE Open-Meteo call (comma-separated
    `models=`). The response suffixes each daily field with the model name, e.g.
    `temperature_2m_max_gfs_seamless`. This cuts the per-city call count from 4
    to 1, which keeps us well under Open-Meteo's free daily limit.
    """
    unit = "fahrenheit" if use_fahrenheit else "celsius"
    model_names = {
        "ecmwf_ifs025":         "ECMWF",
        "gfs_seamless":         "GFS",
        "icon_seamless":        "ICON",
        "gem_seamless":         "GEM",
        "meteofrance_seamless": "MeteoFrance",   # often best on hot continental days
        "ukmo_seamless":        "UKMO",
    }
    d = _get(OPEN_METEO_URL, {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": unit,
        "timezone": "UTC",
        "models": ",".join(model_names.keys()),
        "forecast_days": 3,               # need 3 days so idx=1 works
    }, timeout=8.0)
    if not d:
        return None
    daily = d.get("daily") or {}
    forecasts = {}
    for api_name, display_name in model_names.items():
        vals = daily.get(f"temperature_2m_max_{api_name}") or []
        if target_idx < len(vals) and vals[target_idx] is not None:
            forecasts[display_name] = round(float(vals[target_idx]), 1)
    return forecasts if forecasts else None

def fetch_metno(lat: float, lon: float, tz_offset: int = 0,
                use_fahrenheit: bool = False,
                target_date: Optional[str] = None) -> Optional[float]:
    """MET Norway (Yr) locationforecast — FREE, no API key, global coverage.

    Independent ECMWF-based model. Used as a fallback/extra source so the bot
    still has a forecast (and real model spread for σ) when Open-Meteo is
    rate-limited. MET requires an identifying User-Agent — already set on
    _SESSION — and rejects coordinates with >4 decimals, so we round.

    Returns the forecast daily MAX for the target LOCAL date (°C, or °F if the
    city settles in Fahrenheit), or None.
    """
    data = _get(METNO_URL,
                {"lat": round(lat, 4), "lon": round(lon, 4)}, timeout=8.0)
    if not data:
        return None
    series = ((data.get("properties") or {}).get("timeseries")) or []

    def _time(entry):
        try:
            return datetime.fromisoformat(str(entry.get("time")).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    def _temp(entry):
        return (((entry.get("data") or {}).get("instant") or {})
                .get("details") or {}).get("air_temperature")

    return _series_daily_max(series, target_date, tz_offset, _time, _temp,
                             src_unit="C", use_fahrenheit=use_fahrenheit)

def fetch_nws(lat: float, lon: float, use_fahrenheit: bool = False,
              target_date: Optional[str] = None) -> Optional[float]:
    """US National Weather Service (api.weather.gov) — FREE, no API key.

    US-only (returns None elsewhere). Highest-quality source for the US cities.
    Two steps: /points/{lat},{lon} → the gridpoint forecast URL → daytime period
    high for the target LOCAL day. NWS `startTime` is already in the location's
    local timezone, so its date matches our city-local target_date directly.
    """
    pts = _get(f"{NWS_POINTS_URL}/{round(lat, 4)},{round(lon, 4)}",
               timeout=8.0)
    if not pts:
        return None
    fc_url = (pts.get("properties") or {}).get("forecast")
    if not fc_url:
        return None
    fc = _get(fc_url, timeout=8.0)
    if not fc:
        return None
    for p in ((fc.get("properties") or {}).get("periods") or []):
        if not p.get("isDaytime"):          # daytime period high ≈ daily max
            continue
        try:
            d = datetime.fromisoformat(str(p.get("startTime"))).strftime("%Y-%m-%d")
        except Exception:
            continue
        if target_date and d != target_date:
            continue
        t = _sf(p.get("temperature"))
        if t is None:
            continue
        unit = (p.get("temperatureUnit") or "F").upper()
        if unit == "F" and not use_fahrenheit:
            t = (t - 32) * 5.0 / 9.0
        elif unit == "C" and use_fahrenheit:
            t = t * 9.0 / 5.0 + 32.0
        return round(t, 1)
    return None

def fetch_7timer(lat: float, lon: float, tz_offset: int = 0,
                 use_fahrenheit: bool = False,
                 target_date: Optional[str] = None) -> Optional[float]:
    """7Timer! (free, no API key, GLOBAL) — coarse but independent third opinion.

    Returns the forecast daily MAX for the target LOCAL date. `dataseries` gives
    temp2m (°C, integer) at 3-hourly steps measured from `init` (UTC).
    """
    data = _get(SEVENTIMER_URL,
                {"lon": round(lon, 3), "lat": round(lat, 3),
                 "product": "civil", "output": "json"}, timeout=8.0)
    if not data:
        return None
    init   = data.get("init")               # "YYYYMMDDHH" in UTC
    series = data.get("dataseries") or []
    if not init or not series:
        return None
    try:
        init_dt = datetime.strptime(str(init), "%Y%m%d%H")
    except Exception:
        return None

    def _time(e):
        tp = e.get("timepoint")
        return init_dt + timedelta(hours=float(tp)) if tp is not None else None

    return _series_daily_max(series, target_date, tz_offset,
                             _time, lambda e: e.get("temp2m"),
                             src_unit="C", use_fahrenheit=use_fahrenheit)

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL KEYED SOURCES — each enabled by its Railway env var (see top of file).
# All share one signature so predict() can drive them generically, and all return
# the forecast daily MAX already converted to the city's settlement unit.
#   fetch_xxx(lat, lon, tz_offset, use_fahrenheit, target_date, api_key) -> float|None
# ══════════════════════════════════════════════════════════════════════════════
def fetch_openweather(lat: float, lon: float, tz_offset: int = 0,
                      use_fahrenheit: bool = False, target_date: Optional[str] = None,
                      api_key: str = "") -> Optional[float]:
    """OpenWeatherMap 5-day/3-hour forecast (free tier). Requested in °C, then
    converted to the city's unit. Daily max = max of the 3-hourly highs on the
    target LOCAL day."""
    if not api_key:
        return None
    data = _get(OPENWEATHER_URL,
                {"lat": round(lat, 4), "lon": round(lon, 4),
                 "appid": api_key, "units": "metric"}, timeout=8.0)
    if not data:
        return None

    def _time(row):
        ts = row.get("dt")
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None) if ts is not None else None
        except Exception:
            return None
    def _temp(row):
        main = row.get("main") or {}
        return main.get("temp_max", main.get("temp"))

    return _series_daily_max(data.get("list") or [], target_date, tz_offset,
                             _time, _temp, src_unit="C", use_fahrenheit=use_fahrenheit)

def fetch_weatherapi(lat: float, lon: float, tz_offset: int = 0,
                     use_fahrenheit: bool = False, target_date: Optional[str] = None,
                     api_key: str = "") -> Optional[float]:
    """WeatherAPI.com 3-day forecast (free 1M/mo). It returns BOTH maxtemp_c and
    maxtemp_f, so we just pick the city's unit — no conversion math needed."""
    if not api_key:
        return None
    data = _get(WEATHERAPI_URL,
                {"key": api_key, "q": f"{round(lat, 4)},{round(lon, 4)}",
                 "days": 3, "aqi": "no", "alerts": "no"}, timeout=8.0)
    if not data:
        return None
    days = ((data.get("forecast") or {}).get("forecastday")) or []
    chosen = None
    for d in days:
        if target_date and d.get("date") != target_date:
            continue
        chosen = d
        break
    if chosen is None and days:
        chosen = days[0]                      # target out of range → nearest day
    if chosen is None:
        return None
    day = chosen.get("day") or {}
    t = _sf(day.get("maxtemp_f")) if use_fahrenheit else _sf(day.get("maxtemp_c"))
    return round(t, 1) if t is not None else None

def fetch_visualcrossing(lat: float, lon: float, tz_offset: int = 0,
                         use_fahrenheit: bool = False, target_date: Optional[str] = None,
                         api_key: str = "") -> Optional[float]:
    """Visual Crossing Timeline API (free 1k records/day). Requested in metric
    (°C) for the exact target day, then converted to the city's unit."""
    if not api_key or not target_date:
        return None
    loc = f"{round(lat, 4)},{round(lon, 4)}"
    data = _get(f"{VISUALCROSSING_URL}/{loc}/{target_date}",
                {"key": api_key, "unitGroup": "metric", "include": "days",
                 "elements": "datetime,tempmax"}, timeout=10.0)
    if not data:
        return None
    for d in (data.get("days") or []):
        if d.get("datetime") == target_date or target_date is None:
            t = _sf(d.get("tempmax"))
            if t is not None:
                return _convert_temp(t, "C", use_fahrenheit)
    return None

def fetch_tomorrow(lat: float, lon: float, tz_offset: int = 0,
                   use_fahrenheit: bool = False, target_date: Optional[str] = None,
                   api_key: str = "") -> Optional[float]:
    """Tomorrow.io daily forecast (free ~500/day). Requested in metric (°C),
    converted to the city's unit. `temperatureMax` per daily timeline entry."""
    if not api_key:
        return None
    data = _get(TOMORROW_URL,
                {"location": f"{round(lat, 4)},{round(lon, 4)}",
                 "apikey": api_key, "timesteps": "1d", "units": "metric"}, timeout=8.0)
    if not data:
        return None
    for d in (((data.get("timelines") or {}).get("daily")) or []):
        day = str(d.get("time") or "")[:10]
        if target_date and day != target_date:
            continue
        t = _sf((d.get("values") or {}).get("temperatureMax"))
        if t is not None:
            return _convert_temp(t, "C", use_fahrenheit)
    return None

def _keyed_sources():
    """Return the optional keyed sources that are configured (have a key set).
    Each entry: (label, fetch_fn, api_key). Adding a new API = add one line here
    plus its fetch_xxx function and env var — fully driven by Railway variables."""
    out = []
    if OPENWEATHER_API_KEY:
        out.append(("OpenWeather",    fetch_openweather,    OPENWEATHER_API_KEY))
    if WEATHERAPI_KEY:
        out.append(("WeatherAPI",     fetch_weatherapi,     WEATHERAPI_KEY))
    if VISUALCROSSING_KEY:
        out.append(("VisualCrossing", fetch_visualcrossing, VISUALCROSSING_KEY))
    if TOMORROW_API_KEY:
        out.append(("Tomorrow",       fetch_tomorrow,       TOMORROW_API_KEY))
    return out

# ══════════════════════════════════════════════════════════════════════════════
# WUNDERGROUND — the ACTUAL settlement source Polymarket uses.
# Tries the public api.weather.com endpoint. If it fails, caller falls back
# to METAR. Returns same shape as fetch_metar so it's a drop-in.
# ══════════════════════════════════════════════════════════════════════════════
# ICAO → ISO country code (for the WU station path STATION:9:COUNTRY)
_ICAO_COUNTRY = {
    "LT": "TR", "UU": "RU", "EG": "GB", "LF": "FR", "ED": "DE", "LI": "IT",
    "EP": "PL", "LE": "ES", "LL": "IL", "EH": "NL", "EF": "FI", "DN": "NG",
    "FA": "ZA", "OE": "SA", "RK": "KR", "VH": "HK", "RC": "TW", "ZS": "CN",
    "ZB": "CN", "ZH": "CN", "ZU": "CN", "ZG": "CN", "WS": "SG", "RJ": "JP",
    "WM": "MY", "WI": "ID", "RP": "PH", "NZ": "NZ", "CY": "CA", "KJ": "US",
    "KL": "US", "KS": "US", "KB": "US", "KA": "US", "KI": "US", "KO": "US",
    "KD": "US", "KM": "US", "KAT": "US", "MM": "MX", "SA": "AR", "SB": "BR",
    "MP": "PA", "VI": "IN", "OP": "PK", "K": "US",
}

# Wunderground (api.weather.com) keys — provided via env so no secret lives in
# the (public) code. Comma-separated in WU_KEYS. If unset, Wunderground settlement
# is skipped and the bot falls back to METAR / Open-Meteo analysis.
_WU_KEYS = [k.strip() for k in _os_for_bias.environ.get("WU_KEYS", "").split(",") if k.strip()]

def _icao_country(icao: str) -> str:
    """Best-effort ISO country for a WU station path."""
    for prefix in (icao[:3], icao[:2], icao[:1]):
        if prefix in _ICAO_COUNTRY:
            return _ICAO_COUNTRY[prefix]
    return "US"

def fetch_wunderground(icao: str, tz_offset: int = 0,
                       use_fahrenheit: bool = False,
                       local_date: str = None) -> Optional[Dict]:
    """
    Pull today's observations from Wunderground (Polymarket's settlement source).
    Returns the same dict shape as fetch_metar, or None on failure.
    """
    country = _icao_country(icao)
    units   = "e" if use_fahrenheit else "m"   # e=imperial(F), m=metric(C)
    # Query date: explicit local_date ("YYYY-MM-DD") if given, else today (local).
    # Passing a PAST date returns that whole day's observations, so max_so_far
    # becomes the settled daily high — used by fetch_actual_high() for learning.
    if local_date:
        query_date = local_date.replace("-", "")
    else:
        query_date = (_now_utc() + timedelta(seconds=tz_offset)).strftime("%Y%m%d")

    obs = None
    for key in _WU_KEYS:
        try:
            r = _SESSION.get(
                f"{WUNDERGROUND_URL}/{icao}:9:{country}/observations/historical.json",
                params={"apiKey": key, "units": units, "startDate": query_date},
                timeout=8.0,
            )
            if r.status_code == 200:
                j = r.json()
                if j.get("observations"):
                    obs = j["observations"]
                    break
        except Exception:
            continue
    if not obs:
        return None

    # Build the same structure fetch_metar returns
    rows = []
    for o in obs:
        temp = _sf(o.get("temp"))
        if temp is None:
            continue
        ts = o.get("valid_time_gmt")
        try:
            obs_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None) if ts else None
        except Exception:
            obs_dt = None
        local_t = (obs_dt + timedelta(seconds=tz_offset)).strftime("%H:%M") if obs_dt else "?"
        rows.append({"temp": round(temp, 1), "time": local_t, "obs_dt": obs_dt})

    rows = [r for r in rows if r["obs_dt"] is not None]
    if not rows:
        return None
    rows.sort(key=lambda x: x["obs_dt"], reverse=True)

    max_so_far = max(r["temp"] for r in rows)
    curr = rows[0]
    recent = [(r["time"], r["temp"]) for r in rows[:4]]
    trend = "unknown"
    if len(recent) >= 2:
        diff = recent[0][1] - recent[1][1]
        trend = "rising" if diff > 0.1 else "falling" if diff < -0.1 else "stagnant"

    return {
        "current_temp": curr["temp"],
        "max_so_far":   max_so_far,
        "obs_time":     curr["time"],
        "recent_temps": recent,
        "trend":        trend,
        "obs_count":    len(rows),
        "source":       "wunderground",
    }

def _om_past_max(lat: float, lon: float, date: str,
                 use_fahrenheit: bool = False) -> Optional[float]:
    """Fallback actual: Open-Meteo's analysed daily max for a recent past date."""
    unit = "fahrenheit" if use_fahrenheit else "celsius"
    data = _get(OPEN_METEO_URL, {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": unit,
        "timezone": "UTC",
        "past_days": 7,
        "forecast_days": 1,
    }, timeout=8.0)
    if not data:
        return None
    daily = data.get("daily", {})
    dates = daily.get("time", []) or []
    highs = daily.get("temperature_2m_max", []) or []
    for i, d in enumerate(dates):
        if d == date and i < len(highs) and highs[i] is not None:
            return round(float(highs[i]), 1)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION PROVIDERS — per-city settlement-station overrides
# ══════════════════════════════════════════════════════════════════════════════
# Most cities settle on (or near) their airport, so the default METAR/Wunderground
# obs are correct. A few settle on a SPECIFIC non-airport station — e.g. Hong Kong
# settles on the Hong Kong Observatory (HKO HQ, urban), ~35 km from the airport,
# which can read 1°+ cooler at the afternoon peak. Set "obs": "<provider>" on such
# a city and the bot pulls live obs + settlement from that station instead.
#
# To fix another market the same way: write fetch_<x>_obs / fetch_<x>_actual,
# register them in _OBS_PROVIDERS, and add "obs": "<x>" to that city's config.
HKO_URL = _env("HKO_URL", "https://data.weather.gov.hk/weatherAPI/opendata/weather.php")

# Persistent running daily-max per (city, local-date). Some stations (HKO) publish
# only the CURRENT temperature, so we track the day's max ourselves across scans.
def _resolve_obsmax_file() -> str:
    env = os.environ.get("OBS_MAX_FILE")
    if env:
        return env
    if os.path.isdir("/data"):
        return "/data/obs_max.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs_max.json")

_OBSMAX_FILE = _resolve_obsmax_file()
_OBSMAX_LOCK = threading.Lock()

def _obsmax_load() -> dict:
    try:
        if os.path.exists(_OBSMAX_FILE):
            with open(_OBSMAX_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _obsmax_update(city_key: str, date: str, temp: Optional[float]) -> Optional[float]:
    """Fold `temp` into the running daily max for (city, date); return the max."""
    if temp is None:
        return None
    with _OBSMAX_LOCK:
        d = _obsmax_load()
        key = f"{city_key}|{date}"
        prev = d.get(key)
        mx = temp if prev is None else max(prev, temp)
        if mx != prev:
            d[key] = mx
            for old in sorted(d.keys())[:-300]:    # keep the file small
                d.pop(old, None)
            try:
                with open(_OBSMAX_FILE, "w") as f:
                    json.dump(d, f)
            except Exception:
                pass
        return mx

def _obsmax_get(city_key: str, date: str) -> Optional[float]:
    return _obsmax_load().get(f"{city_key}|{date}")


# ── Hong Kong Observatory (HKO open data) provider ────────────────────────────
def _hko_current(station: str = "Hong Kong Observatory") -> Optional[float]:
    """Current temperature (°C) at an HKO station from the open-data feed."""
    d = _get(HKO_URL, {"dataType": "rhrread", "lang": "en"}, timeout=10.0, cache_ttl=300)
    if not isinstance(d, dict):
        return None
    for t in (d.get("temperature") or {}).get("data", []):
        if t.get("place") == station:
            return _sf(t.get("value"))
    return None

def fetch_hko_obs(city_key, tz, use_f, local_date=None) -> Optional[Dict]:
    """Live obs for an HKO-settled city from the Hong Kong Observatory station (the
    actual settlement source), with a tracked running daily max."""
    cur = _hko_current("Hong Kong Observatory")
    if cur is None:
        return None
    local_now = _now_utc() + timedelta(seconds=tz)
    today = local_now.strftime("%Y-%m-%d")
    mx = _obsmax_update(city_key, today, cur)
    return {
        "current_temp": cur,
        "max_so_far":   mx if mx is not None else cur,
        "obs_time":     local_now.strftime("%H:%M"),
        "recent_temps": [(local_now.strftime("%H:%M"), cur)],
        "trend":        "unknown",
        "obs_count":    1,
        "source":       "hko",
    }

def _hko_daily_max(date: str) -> Optional[float]:
    """HKO 'Absolute Daily Maximum Temperature' from the official Daily Extract
    (CLMMAXT) — the EXACT source Polymarket uses to resolve HK markets, to 0.1°C.
    Returns None until HKO publishes that day (same delay Polymarket waits on)."""
    try:
        y, m, dd = date.split("-")
        dd = int(dd)
    except Exception:
        return None
    d = _get("https://data.weather.gov.hk/weatherAPI/opendata/opendata.php",
             {"dataType": "CLMMAXT", "lang": "en", "rformat": "json",
              "station": "HKO", "year": int(y), "month": int(m)},
             timeout=15.0, cache_ttl=3600)
    if not isinstance(d, dict):
        return None
    for row in (d.get("data") or []):
        # row = [year, month, day, value, completeness]
        try:
            if int(row[2]) == dd:
                v = _sf(row[3])
                return round(v, 1) if v is not None else None
        except Exception:
            continue
    return None

def fetch_hko_actual(city_key, date) -> Optional[float]:
    """Settled HKO daily max for a past date. Order of trust:
    1) CLMMAXT — the EXACT 'Absolute Daily Max' Polymarket resolves on (once HKO
       publishes it). This makes the bot's settled value match Polymarket exactly.
    2) our tracked running max from live HKO readings (real-time proxy before #1).
    3) Open-Meteo reanalysis at HKO HQ (last-resort estimate)."""
    v = _hko_daily_max(date)              # 1) exact Polymarket source
    if v is not None:
        return v
    mx = _obsmax_get(city_key, date)      # 2) live running-max proxy
    if mx is not None:
        return round(float(mx), 1)
    meta = CITIES.get(city_key) or {}     # 3) estimate
    return _om_past_max(meta.get("lat"), meta.get("lon"), date, meta.get("f", False))


# Registry: city config "obs" -> {live, actual}. Add new stations here to fix more.
_OBS_PROVIDERS = {
    "hko": {"live": fetch_hko_obs, "actual": fetch_hko_actual},
}


def fetch_actual_high(city_key: str, date: str) -> Optional[float]:
    """
    The SETTLED daily high for a completed past date, used to teach the model
    its per-city bias (see deb_blend / _signed_bias). Uses the city's configured
    observation provider (e.g. HKO) when set, else Wunderground/airport, falling
    back to Open-Meteo's analysis. `date` is the local date "YYYY-MM-DD".
    """
    meta = CITIES.get(city_key)
    if not meta:
        return None
    prov = _OBS_PROVIDERS.get(meta.get("obs"))
    if prov:
        v = prov["actual"](city_key, date)
        if v is not None:
            return round(float(v), 1)
    icao  = meta["icao"]
    tz    = meta["tz"]
    use_f = meta.get("f", False)
    # 1) Wunderground historical at the settlement station (wu_station override) → max
    wu = fetch_wunderground(meta.get("wu_station") or icao, tz, use_f, local_date=date)
    if wu and wu.get("max_so_far") is not None:
        return round(float(wu["max_so_far"]), 1)
    # 2) Open-Meteo analysed max as fallback
    return _om_past_max(meta["lat"], meta["lon"], date, use_f)

# ── Batched METAR cache ───────────────────────────────────────────────────────
# aviationweather.gov accepts MANY station ids in one call (ids=RJTT,KJFK,...).
# Fetching all 51 stations in a SINGLE request per scan — instead of one request
# per city — keeps the free API from rate-limiting/timing out (which previously
# tripped the circuit breaker for the whole host). prefetch_metars() fills this
# cache once per scan; fetch_metar() serves from it and only hits the network for
# stations the batch didn't cover.
_METAR_BATCH: Dict[str, list] = {}
_METAR_BATCH_LOCK = threading.Lock()
_METAR_BATCH_TS = 0.0
_METAR_BATCH_TTL = float(os.environ.get("METAR_BATCH_TTL", "900"))   # 15 min

def prefetch_metars(icaos) -> int:
    """One batched aviationweather.gov call for ALL given stations. Returns the
    number of stations that came back with data. Never raises."""
    global _METAR_BATCH_TS
    ids = sorted({(i or "").upper() for i in icaos if i})
    if not ids:
        return 0
    data = _get(METAR_URL, {"ids": ",".join(ids), "format": "json", "hours": 26},
                timeout=20.0)
    if not isinstance(data, list):
        return 0
    grouped: Dict[str, list] = {}
    for row in data:
        k = (row.get("icaoId") or "").upper()
        if k:
            grouped.setdefault(k, []).append(row)
    with _METAR_BATCH_LOCK:
        _METAR_BATCH.clear()
        _METAR_BATCH.update(grouped)
        for i in ids:                      # mark every requested id as fetched,
            _METAR_BATCH.setdefault(i, [])  # even empty, so we don't re-call per city
        _METAR_BATCH_TS = _time.time()
    print(f"[metar] prefetched {len(grouped)}/{len(ids)} stations in 1 call")
    return len(grouped)

def _batched_rows(icao: str):
    """Raw rows for `icao` from a FRESH batch, or None if the batch is stale or
    never covered this station (caller should then fetch it itself)."""
    with _METAR_BATCH_LOCK:
        if _time.time() - _METAR_BATCH_TS > _METAR_BATCH_TTL:
            return None
        return _METAR_BATCH.get((icao or "").upper())   # [] = fetched, no obs


def _parse_metar_rows(rows: list, tz_offset: int, use_fahrenheit: bool = False) -> Optional[Dict]:
    """Turn raw aviationweather rows into the live-obs dict (today's max etc.).
    METAR temps are always °C; convert to °F for Fahrenheit-settled cities."""
    today_local = (_now_utc() + timedelta(seconds=tz_offset)).strftime("%Y-%m-%d")
    results = []
    for row in rows:
        temp = _sf(row.get("temp"))
        if temp is None:
            continue
        if use_fahrenheit:
            temp = temp * 9.0 / 5.0 + 32.0      # METAR is °C → city settles in °F
        obs_raw = row.get("reportTime") or row.get("receiptTime") or row.get("observation_time")
        if not obs_raw:
            continue
        try:
            obs_dt   = datetime.fromisoformat(str(obs_raw).replace("Z", "+00:00"))
            local_dt = obs_dt + timedelta(seconds=tz_offset)
            if local_dt.strftime("%Y-%m-%d") != today_local:
                continue
            results.append({
                "temp":         round(temp, 1),
                "time":         local_dt.strftime("%H:%M"),
                "obs_dt":       obs_dt,
                "humidity":     _sf(row.get("relh")),
                "wind_dir":     _sf(row.get("wdir")),
                "wind_speed_kt":_sf(row.get("wspd")),
            })
        except Exception:
            continue

    if not results:
        return None

    results.sort(key=lambda x: x["obs_dt"], reverse=True)
    max_so_far = max(r["temp"] for r in results)
    curr       = results[0]
    recent     = [(r["time"], r["temp"]) for r in results[:4]]
    trend      = "unknown"
    if len(recent) >= 2:
        diff  = recent[0][1] - recent[1][1]
        trend = "rising" if diff > 0.1 else "falling" if diff < -0.1 else "stagnant"

    return {
        "current_temp":  curr["temp"],
        "max_so_far":    max_so_far,
        "obs_time":      curr["time"],
        "recent_temps":  recent,
        "trend":         trend,
        "obs_count":     len(results),
        "humidity":      curr.get("humidity"),
        "wind_speed_kt": curr.get("wind_speed_kt"),
    }


def fetch_metar(icao: str, tz_offset: int = 0, local_date: str = None,
                use_fahrenheit: bool = False) -> Optional[Dict]:
    """
    Live METAR for TODAY's observations (current temp / max_so_far). Serves from
    the once-per-scan batched cache when available, else makes a single per-station
    call covering the whole local day. METAR is °C; pass use_fahrenheit=True for
    cities that settle in °F so the obs come back in the right unit.
    """
    rows = _batched_rows(icao)              # [] = fetched-but-empty, None = not cached
    if rows is None:
        local_now  = _now_utc() + timedelta(seconds=tz_offset)
        hours_back = min(26, max(8, local_now.hour + 2))
        data = _get(METAR_URL, {"ids": icao, "format": "json", "hours": hours_back},
                    timeout=10.0)
        rows = data if isinstance(data, list) else None
    if not rows:
        return None
    return _parse_metar_rows(rows, tz_offset, use_fahrenheit)

# ══════════════════════════════════════════════════════════════════════════════
# POLYMARKET LIVE PRICES  (Gamma API for discovery + CLOB API for prices)
# Both public, no auth needed.
#
# Flow:
#   1. Gamma /events?search="Tokyo temperature June 14"  → find the event
#   2. event.markets[] → each bucket market has clobTokenIds + outcomes
#   3. CLOB /price?token_id=<YES token>&side=buy → live YES price
# ══════════════════════════════════════════════════════════════════════════════
# (GAMMA / CLOB / DATA are configured at the top of the file — env-overridable)

# Month names for building search queries
_MONTHS = ["January","February","March","April","May","June",
           "July","August","September","October","November","December"]

def _pm_display_name(city_key: str) -> str:
    """Map registry key to the name Polymarket uses in market titles."""
    overrides = {
        "new york": "NYC",
        "hong kong": "Hong Kong",
    }
    return overrides.get(city_key, city_key.title())

def _pm_slug_city(city_key: str) -> str:
    """City as it appears in Polymarket temperature-market slugs (lowercase, hyphenated)."""
    overrides = {
        "new york": "nyc",
        "aurora":   "denver",     # Polymarket lists this metro as 'denver'
    }
    return overrides.get(city_key, city_key).replace(" ", "-")

def fetch_polymarket_market(city_key: str, target_date: str, debug: bool = False,
                            cache_ttl: float = None) -> Optional[Dict]:
    """
    Find the temperature event for a city+date and return per-bucket YES prices.
    Pass a small cache_ttl (e.g. 60) for fresher prices (price watches).
    """
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        date_phrase = f"{_MONTHS[dt.month-1]} {dt.day}"
        month_l     = _MONTHS[dt.month-1].lower()
    except Exception:
        return None

    city_disp = _pm_display_name(city_key)

    # Polymarket's full-text search (/public-search and /events?search=) is
    # unreliable — it intermittently 500s and TIMES OUT, which used to hang the
    # entire scan one city at a time. Look the event up by its DETERMINISTIC slug
    # instead, e.g. 'highest-temperature-in-chicago-on-june-19-2026' — one fast,
    # reliable call. Fall back to filtering the active-events list client-side.
    event = None
    slug = f"highest-temperature-in-{_pm_slug_city(city_key)}-on-{month_l}-{dt.day}-{dt.year}"
    ev = _get(f"{GAMMA}/events", {"slug": slug}, timeout=8.0, cache_ttl=cache_ttl)
    if isinstance(ev, list) and ev:
        event = ev[0]
        if debug:
            print(f"  [debug] slug hit: {slug}")

    if event is None:
        # Fallback: scan currently-active events, match city + date in the title.
        lst = _get(f"{GAMMA}/events",
                   {"active": "true", "closed": "false", "limit": 500}, timeout=12.0)
        if isinstance(lst, list):
            cd, dl = city_disp.lower(), date_phrase.lower()
            for e in lst:
                title = (e.get("title") or "").lower()
                if cd in title and ("temperature" in title or "temp" in title) and dl in title:
                    event = e
                    break
            if event is None:           # looser: city + temperature keyword only
                for e in lst:
                    title = (e.get("title") or "").lower()
                    if cd in title and ("temperature" in title or "temp" in title):
                        event = e
                        break
        if debug and event is not None:
            print(f"  [debug] list fallback matched: {event.get('title')}")

    if event is None:
        if debug:
            print(f"  [debug] no Polymarket market for {city_key} on {target_date}")
        return None

    if debug:
        print(f"  [debug] matched event: {event.get('title')}")

    markets = event.get("markets") or []
    if debug:
        print(f"  [debug] event has {len(markets)} bucket markets")

    buckets: Dict[int, Dict] = {}
    for m in markets:
        question = m.get("question") or m.get("groupItemTitle") or m.get("title") or ""
        temp = _extract_temp_from_title(question)
        if temp is None:
            continue
        try:
            token_ids = json.loads(m.get("clobTokenIds") or "[]")
            outcomes  = json.loads(m.get("outcomes") or "[]")
            prices    = json.loads(m.get("outcomePrices") or "[]")
        except Exception:
            token_ids, outcomes, prices = [], [], []

        yes_price = no_price = token_yes = None
        for i, oc in enumerate(outcomes):
            ocl = str(oc).lower()
            pr  = _sf(prices[i]) if i < len(prices) else None
            tid = token_ids[i] if i < len(token_ids) else None
            if ocl == "yes":
                yes_price, token_yes = pr, tid
            elif ocl == "no":
                no_price = pr

        vol = _sf(m.get("volume") or m.get("volumeNum") or m.get("volume24hr")) or 0.0
        rng = _extract_range_from_title(question)
        lo, hi = rng if rng else (temp, temp)
        buckets[temp] = {
            "yes":       round(yes_price, 3) if yes_price is not None else None,
            "no":        round(no_price, 3) if no_price is not None else None,
            "token_yes": token_yes,
            "vol":       round(vol, 0),
            "lo":        lo,    # bucket spans [lo, hi] degrees (lo==hi = single)
            "hi":        hi,
        }

    if debug:
        print(f"  [debug] parsed {len(buckets)} buckets: {sorted(buckets.keys())}")

    if not buckets:
        return None

    # ── Override Gamma's stale last-trade prices with the LIVE order book ──
    # Gamma `outcomePrices` lags badly on thin markets (saw 12°C @ 28¢ while live was
    # 56¢ → fake edge). One batch CLOB call gives the real bid/ask; use the midpoint
    # (matches the % Polymarket shows). Falls back to the Gamma price per bucket if
    # the CLOB call fails, so there is never a regression.
    if LIVE_CLOB_PRICES:
        try:
            live = fetch_clob_prices([b.get("token_yes") for b in buckets.values()],
                                     cache_ttl=cache_ttl)
            for b in buckets.values():
                tok = b.get("token_yes")
                lp  = live.get(tok)
                # Fallback to the PROVEN single /price endpoint for live (non-dead)
                # buckets the batch didn't cover — guarantees the fix works even if
                # the batch response shape differs from what we expect.
                if (lp is None or lp.get("buy") is None) and tok and (b.get("yes") or 0) >= 0.02:
                    ask = fetch_clob_price(tok, "buy",  cache_ttl=cache_ttl)
                    bid = fetch_clob_price(tok, "sell", cache_ttl=cache_ttl)
                    vals = [x for x in (ask, bid) if x is not None]
                    if vals:
                        lp = {"buy": ask, "sell": bid, "mid": round(sum(vals) / len(vals), 3)}
                if not lp:
                    continue
                # Use the ASK ("Buy Yes") — what you ACTUALLY pay to enter, the number
                # Polymarket shows on the buy button. (The midpoint is the % display,
                # but you can't buy at the mid.) Keep the mid for reference.
                ask, mid = lp.get("buy"), lp.get("mid")
                price = ask if ask is not None else mid
                if price is not None:
                    b["yes"]     = round(price, 3)
                    b["yes_mid"] = mid
                    if lp.get("sell") is not None:
                        b["no"] = round(1 - lp["sell"], 3)   # Buy No ask = 1 − Yes bid
        except Exception as e:
            if debug:
                print(f"  [debug] CLOB live-price override failed: {e}")

    slug = event.get("slug") or ""
    return {
        "title":   event.get("title"),
        "buckets": buckets,
        "url":     f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com",
    }

def _market_is_range(pm_data) -> bool:
    """True only if the market uses TRUE finite multi-degree range buckets, e.g.
    San Francisco's '68-69°F'. Open-ended edge buckets ('35°C or below') alone do
    NOT count — those exist on normal single-degree markets, which keep their
    existing behaviour untouched."""
    if not pm_data or not pm_data.get("buckets"):
        return False
    for b in pm_data["buckets"].values():
        lo, hi = b.get("lo"), b.get("hi")
        if lo is None or hi is None:
            continue
        if -9000 < lo and hi < 9000 and hi != lo:   # finite span of 2+ degrees
            return True
    return False

def _remap_distribution(dist, pm_data):
    """Aggregate a single-degree model distribution into the MARKET's buckets.
    Each market bucket gets the SUM of the model's per-degree probabilities whose
    value falls in its [lo, hi] span — so a '68-69°F' bucket gets P(68)+P(69).
    No-op (returns dist unchanged) for normal single-degree markets, so existing
    behaviour is untouched."""
    if not dist or not _market_is_range(pm_data):
        return dist
    out = []
    for key, b in pm_data["buckets"].items():
        lo = b.get("lo"); hi = b.get("hi")
        if lo is None or hi is None:
            lo = hi = key
        psum = sum(d["probability"] for d in dist if lo <= d["value"] <= hi)
        out.append({"value": key, "probability": round(psum, 4), "lo": lo, "hi": hi})
    out.sort(key=lambda x: x["probability"], reverse=True)
    return out

def _extract_temp_from_title(text: str) -> Optional[int]:
    """Pull the integer temperature out of a market question like '29°C' or '84-85°F'."""
    import re
    if not text:
        return None
    # match patterns like "29°C", "29C", "29 °C", "84°F"
    m = re.search(r"(\d{1,3})\s*°?\s*[CF]", text)
    if m:
        return int(m.group(1))
    # "or below" / "or higher" style — grab first number
    m = re.search(r"(\d{1,3})", text)
    if m:
        return int(m.group(1))
    return None

def _extract_range_from_title(text: str):
    """Bucket span (lo, hi) for a market question. Handles 2-degree range buckets
    (e.g. San Francisco '68-69°F' → (68, 69)), open-ended buckets ('80°F or higher'
    → (80, 9999); '61°F or below' → (-9999, 61)) and single buckets ('32°C' → (32,32)).
    Returns None if no number is present."""
    import re
    if not text:
        return None
    # explicit "68-69" / "84–85" range (hyphen or en-dash)
    m = re.search(r"(\d{1,3})\s*[-–]\s*(\d{1,3})", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    # open-ended upward
    m = re.search(r"(\d{1,3})\s*°?\s*[CF]?\s*or\s+(?:higher|above|more|greater|warmer)", text, re.I)
    if m:
        return (int(m.group(1)), 9999)
    # open-ended downward
    m = re.search(r"(\d{1,3})\s*°?\s*[CF]?\s*or\s+(?:below|lower|less|under|colder)", text, re.I)
    if m:
        return (-9999, int(m.group(1)))
    v = _extract_temp_from_title(text)
    return (v, v) if v is not None else None

def fetch_clob_price(token_id: str, side: str = "buy",
                     cache_ttl: float = None) -> Optional[float]:
    """Live price from CLOB order book for a token (more current than Gamma).
    Pass a small cache_ttl (e.g. 60) for near-real-time reads like price watches."""
    if not token_id:
        return None
    d = _get(f"{CLOB}/price", {"token_id": token_id, "side": side}, timeout=5.0,
             cache_ttl=cache_ttl)
    if isinstance(d, dict):
        return _sf(d.get("price"))
    return None

def fetch_clob_prices(token_ids, cache_ttl: float = None) -> Dict[str, Dict]:
    """Batch LIVE order-book prices for many YES tokens in ONE call. Gamma's
    `outcomePrices` is the last-trade / cached value and lags the live book badly on
    thin temperature buckets (it once showed 12°C @ 28¢ while the live book was 56¢),
    which manufactures phantom edges. The CLOB /prices endpoint returns the real
    best bid/ask per token. Returns {token_id: {"buy": ask, "sell": bid, "mid": m}}."""
    toks = [t for t in dict.fromkeys(token_ids) if t]
    if not toks:
        return {}
    body = []
    for t in toks:
        body.append({"token_id": t, "side": "BUY"})
        body.append({"token_id": t, "side": "SELL"})
    d = _post(f"{CLOB}/prices", body, timeout=10.0, cache_ttl=cache_ttl)
    out: Dict[str, Dict] = {}

    def _side(row, s):                      # tolerate BUY/buy/Buy key casing
        if not isinstance(row, dict):
            return None
        return _sf(row.get(s) or row.get(s.lower()) or row.get(s.capitalize()))

    if isinstance(d, dict):
        for t in toks:
            row = d.get(t) or {}
            ask, bid = _side(row, "BUY"), _side(row, "SELL")
            vals = [x for x in (ask, bid) if x is not None]
            if vals:
                out[t] = {"buy": ask, "sell": bid, "mid": round(sum(vals) / len(vals), 3)}
    return out

# ══════════════════════════════════════════════════════════════════════════════
# YOUR POLYMARKET POSITIONS  (public on-chain data — needs only wallet ADDRESS)
#
# Uses the public Data API. Requires only your PUBLIC wallet address (0x...).
# NEVER needs your private key, password, or seed phrase.
# Set it once: export POLYMARKET_WALLET=0xYourAddress
# ══════════════════════════════════════════════════════════════════════════════
def get_wallet_address(cli_arg: Optional[str] = None) -> Optional[str]:
    """Resolve wallet address from CLI arg or POLYMARKET_WALLET env var."""
    addr = cli_arg or os.environ.get("POLYMARKET_WALLET")
    if not addr:
        return None
    addr = addr.strip()
    if not addr.lower().startswith("0x") or len(addr) < 10:
        return None
    return addr

def fetch_positions(wallet: str, weather_only: bool = True) -> Optional[List[Dict]]:
    """
    Fetch open positions for a wallet from the public Data API.
    weather_only=True filters to temperature/weather markets.
    """
    if not wallet:
        return None
    data = _get(f"{DATA}/positions", {
        "user": wallet,
        "limit": 200,
        "sizeThreshold": 0.1,
    }, timeout=10.0)
    if not isinstance(data, list):
        return None

    positions = []
    for pos in data:
        title = (pos.get("title") or "")
        slug  = (pos.get("slug") or "") + (pos.get("eventSlug") or "")
        is_weather = ("temperature" in title.lower() or "temperature" in slug.lower()
                      or "temp" in slug.lower() or "weather" in slug.lower())
        if weather_only and not is_weather:
            continue
        positions.append({
            "title":        title,
            "outcome":      pos.get("outcome"),            # Yes / No
            "size":         _sf(pos.get("size")) or 0.0,   # shares held
            "avg_price":    _sf(pos.get("avgPrice")),      # your entry price
            "cur_price":    _sf(pos.get("curPrice")),      # current price
            "initial_value":_sf(pos.get("initialValue")),  # what you paid
            "current_value":_sf(pos.get("currentValue")),  # worth now
            "cash_pnl":     _sf(pos.get("cashPnl")),        # profit/loss $
            "percent_pnl":  _sf(pos.get("percentPnl")),     # profit/loss %
            "redeemable":   pos.get("redeemable", False),   # settled & claimable
            "event_slug":   pos.get("eventSlug"),
        })
    return positions

def compute_edges(distribution: List[Dict], pm: Optional[Dict],
                  temp_sym: str, confidence: float = 1.0,
                  edge_min: Optional[float] = None) -> List[Dict]:
    """
    Match model probabilities against Polymarket prices, compute edge per bucket.

    confidence (0-1): how much to trust the model right now (from σ / agreement /
      timing). Lower confidence REQUIRES a bigger edge before a bucket is called
      tradeable — this is the main protection against marginal losing bets.
    edge_min: base edge threshold (defaults to EDGE_MIN env). The effective
      threshold is edge_min / confidence, so a shaky read must clear a higher bar.

    Each edge also carries `ev` (expected return per $1 staked) and `kelly` (full
    Kelly fraction) so position size can be tied to the real edge, not a guess.
    Returns list of edge dicts sorted by best_edge descending.
    """
    if not pm or not pm.get("buckets"):
        return []
    base_min = EDGE_MIN if edge_min is None else edge_min
    req      = base_min / max(0.5, min(1.0, confidence))   # higher bar when unsure
    # No model distribution = no data (e.g. all weather fetches failed). Without
    # this guard, model prob falls through as 0 for every bucket, and the BUY NO
    # branch below (which only needs mp <= 0.30) manufactures phantom "edges"
    # against an empty model — the bogus edge+38% lines seen in the logs.
    if not distribution:
        return []
    model = {b["value"]: b["probability"] for b in distribution}
    pm_buckets = pm["buckets"]

    edges = []
    all_temps = set(model.keys()) | set(pm_buckets.keys())
    for temp in all_temps:
        mp = model.get(temp, 0.0)                    # model probability
        pb = pm_buckets.get(temp, {})
        yes = pb.get("yes")                          # market YES price
        no  = pb.get("no")
        vol = pb.get("vol", 0)
        if yes is None:
            continue

        # ── Tradeability guards ──────────────────────────────────────────────
        # A price of 0¢ or 100¢ means the order book is empty at that level —
        # you CANNOT actually buy it. Treat extremes as non-tradeable.
        yes_buyable = (yes is not None) and (0.02 <= yes <= 0.98)
        no_buyable  = (no  is not None) and (0.02 <= no  <= 0.98)

        edge_yes = mp - yes                          # +ve → BUY YES
        edge_no  = (1 - mp) - (no if no is not None else (1 - yes))

        action = "skip"
        best_edge = max(edge_yes, edge_no)

        # Only suggest a side if BOTH: edge clears the (confidence-adjusted) bar
        # AND it's actually buyable AND the model genuinely supports that side.
        if edge_yes >= req and yes_buyable and mp >= 0.40:
            action, best_edge = "BUY YES", edge_yes
        elif edge_no >= req and no_buyable and mp <= 0.30:
            action, best_edge = "BUY NO", edge_no
        else:
            action = "skip"
            best_edge = max(edge_yes if yes_buyable else -1,
                            edge_no  if no_buyable  else -1)
            if best_edge < 0:
                best_edge = 0

        # ── EV + Kelly sizing for the chosen side ────────────────────────────
        # ev    = expected return per $1 staked (e.g. 0.25 = +25% expected)
        # kelly = full Kelly fraction of bankroll; USE A FRACTION of it (quarter-
        #         Kelly) in practice — full Kelly is too aggressive and one bad
        #         streak ruins it. kelly_quarter is provided for convenience.
        ev = kelly = 0.0
        if action == "BUY YES" and yes and yes < 1.0:
            ev    = (mp - yes) / yes
            kelly = max(0.0, (mp - yes) / (1.0 - yes))
        elif action == "BUY NO" and no and no < 1.0:
            qn    = 1.0 - mp
            ev    = (qn - no) / no
            kelly = max(0.0, (qn - no) / (1.0 - no))

        # liquidity flag
        thin = vol < 500

        edges.append({
            "temp":        temp,
            "model_prob":  round(mp, 3),
            "yes_price":   yes,
            "no_price":    no,
            "edge_yes":    round(edge_yes, 3),
            "edge_no":     round(edge_no, 3),
            "best_edge":   round(best_edge, 3),
            "action":      action,
            "ev":          round(ev, 3),                 # expected return per $1
            "kelly":       round(kelly, 3),              # full Kelly fraction
            "kelly_quarter": round(kelly / 4.0, 3),      # safer suggested stake
            "vol":         vol,
            "yes_buyable": yes_buyable,
            "no_buyable":  no_buyable,
            "thin":        thin,
        })

    edges.sort(key=lambda x: x["best_edge"], reverse=True)
    return edges

# ══════════════════════════════════════════════════════════════════════════════
# DEB — Dynamic Error Blending
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_history_file() -> str:
    """Where to persist learned forecast/actual history.

    On Railway the code directory is wiped on every redeploy, which would reset
    the model's learning each deploy. Prefer an explicit DEB_HISTORY_FILE, then
    the persistent /data volume (same place the monitor's state DB lives), and
    only fall back to the code dir for local runs.
    """
    env = os.environ.get("DEB_HISTORY_FILE")
    if env:
        return env
    if os.path.isdir("/data"):
        return "/data/deb_history.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "deb_history.json")

_HISTORY_FILE = _resolve_history_file()

def _load_history() -> dict:
    try:
        if os.path.exists(_HISTORY_FILE):
            with open(_HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_history(data: dict):
    try:
        with open(_HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def deb_blend(city: str, forecasts: Dict[str, float],
              target_date: str = None,
              lookback: int = DEB_LOOKBACK, decay: float = DEB_DECAY
              ) -> Tuple[Optional[float], str, float]:
    """Returns (blended_with_bias, info, bias_applied). The RAW (no-bias) blend is
    blended_with_bias - bias_applied, so callers can show both numbers."""
    if not forecasts:
        return None, "no forecasts", 0.0
    history   = _load_history()
    city_data = history.get(city.lower(), {})
    skip_date = target_date or _now_utc().strftime("%Y-%m-%d")
    errors: Dict[str, list] = {m: [] for m in forecasts}
    days_used = 0
    for date in sorted(city_data.keys(), reverse=True):
        if date >= skip_date:
            continue
        rec    = city_data[date]
        actual = _sf(rec.get("actual_high"))
        if actual is None:
            continue
        past = rec.get("forecasts", {})
        w    = decay ** days_used
        for model in forecasts:
            if model in past and past[model] is not None:
                errors[model].append((abs(float(past[model]) - actual), w))
        days_used += 1
        if days_used >= lookback:
            break

    manual = CITY_BIAS.get((city or "").strip().lower())   # explicit per-city override

    if days_used < 2:
        n       = len(forecasts)
        blended = sum(forecasts.values()) / n
        # A manual CITY_BIAS overrides everything; otherwise apply the default
        # upward nudge (models smooth the afternoon peak, so they run low). The
        # learned signed-bias takes over once history accumulates.
        bias = manual if manual is not None else DEFAULT_PEAK_BIAS
        blended += bias
        tag = f"manual {bias:+.1f}°" if manual is not None else f"+{DEFAULT_PEAK_BIAS}° peak-bias"
        return (round(blended, 1),
                f"equal-weight({n} models, {days_used}d history, {tag})",
                bias)

    maes = {}
    for model, errs in errors.items():
        if errs:
            tw = sum(w for _, w in errs)
            maes[model] = sum(e * w for e, w in errs) / tw if tw > 0 else 2.0
        else:
            maes[model] = 2.0

    inv       = {m: 1.0 / (mae + 0.1) for m, mae in maes.items() if m in forecasts}
    total_inv = sum(inv.values())
    if total_inv == 0:
        return round(sum(forecasts.values()) / len(forecasts), 1), "equal-weight(fallback)", 0.0
    weights   = {m: v / total_inv for m, v in inv.items()}
    blended   = sum(forecasts[m] * weights[m] for m in weights)

    # ── SIGNED bias correction ──
    # MAE measures error magnitude but not DIRECTION. Models systematically
    # under-predict the daily MAX (they smooth the peak hour). Compute the
    # signed mean error (actual - forecast) from history and shift the blend.
    # A manual CITY_BIAS override wins over the learned value when set.
    if manual is not None:
        bias = manual
    else:
        bias = _signed_bias(city_data, list(forecasts.keys()), skip_date, lookback, decay)
    if bias is not None:
        blended += bias

    top       = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    info      = " | ".join(f"{m}({w*100:.0f}%,MAE:{maes[m]:.1f}°)" for m, w in top)
    if bias is not None and abs(bias) >= 0.05:
        info += f" | {'manual ' if manual is not None else ''}bias{bias:+.1f}°"
    return round(blended, 1), info, (bias or 0.0)

def _signed_bias(city_data: dict, models: list, skip_date: str,
                 lookback: int = DEB_LOOKBACK, decay: float = DEB_DECAY) -> Optional[float]:
    """
    Mean signed error (actual - model_mean) over history, decay-weighted.
    Positive → models ran COLD (actual was higher) → shift blend UP.
    Returns None if not enough history.
    """
    diffs = []
    days_used = 0
    for date in sorted(city_data.keys(), reverse=True):
        if date >= skip_date:
            continue
        rec    = city_data[date]
        actual = _sf(rec.get("actual_high"))
        if actual is None:
            continue
        past = rec.get("forecasts", {})
        mvals = [float(past[m]) for m in models if m in past and past[m] is not None]
        if not mvals:
            continue
        model_mean = sum(mvals) / len(mvals)
        w = decay ** days_used
        diffs.append(((actual - model_mean), w))
        days_used += 1
        if days_used >= lookback:
            break
    if len(diffs) < 2:
        return None
    tw = sum(w for _, w in diffs)
    if tw == 0:
        return None
    bias = sum(d * w for d, w in diffs) / tw
    # clamp to a sane range so one weird day can't swing it wildly
    return max(-1.5, min(1.5, bias))

def record_actual(city: str, date: str, actual_high: float):
    history  = _load_history()
    city_key = city.lower()
    history.setdefault(city_key, {}).setdefault(date, {})["actual_high"] = actual_high
    cutoff   = (_now_utc() - timedelta(days=180)).strftime("%Y-%m-%d")
    history[city_key] = {d: v for d, v in history[city_key].items() if d >= cutoff}
    _save_history(history)
    print(f"✅ Recorded {city} {date} actual_high={actual_high}")

# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY ENGINE — Gaussian buckets
# ══════════════════════════════════════════════════════════════════════════════
def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def compute_probabilities(mu: float, sigma: float,
                          max_so_far: Optional[float],
                          city: str,
                          use_hko: bool = False,
                          predicting_tomorrow: bool = False) -> List[Dict]:
    if mu is None or sigma is None:
        return []
    # When predicting tomorrow, max_so_far is irrelevant (today's data)
    min_settle = -999
    if not predicting_tomorrow and max_so_far is not None:
        v = settlement_round(city, max_so_far)
        if v is not None:
            min_settle = v

    target_mu = settlement_round(city, mu) or int(round(mu))
    search    = max(3, int(sigma * 3))
    probs     = {}
    for n in range(target_mu - search, target_mu + search + 1):
        if n < min_settle:
            continue
        if use_hko:
            p   = _norm_cdf(n + 1.0, mu, sigma) - _norm_cdf(n, mu, sigma)
            rng = f"[{n}.0~{n+1}.0)"
        else:
            p   = _norm_cdf(n + 0.5, mu, sigma) - _norm_cdf(n - 0.5, mu, sigma)
            rng = f"[{n-0.5}~{n+0.5})"
        if p > 0.005:
            probs[n] = (p, rng)

    total = sum(p for p, _ in probs.values())
    if total <= 0:
        return []
    result = [{"value": n, "probability": round(p / total, 3), "range": rng}
              for n, (p, rng) in probs.items()]
    result.sort(key=lambda x: x["probability"], reverse=True)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def estimate_peak_window(lat: float) -> Tuple[int, int]:
    a = abs(lat)
    if a < 23:   return 13, 15
    elif a < 45: return 14, 16
    else:        return 15, 17

def get_peak_status(hour: int, fp: int, lp: int) -> str:
    if hour > lp:             return "past"
    elif fp <= hour <= lp:    return "in_window"
    return "before"

def is_dead_market_check(local_hour: int, peak_end: int,
                          max_so_far: Optional[float],
                          cur: Optional[float],
                          predicting_tomorrow: bool) -> bool:
    # Dead market only possible when predicting TODAY
    if predicting_tomorrow:
        return False
    if max_so_far is None or cur is None:
        return False
    drop = max_so_far - cur
    return (local_hour >= 21 and drop >= 3.0) or (local_hour >= peak_end and drop >= 1.5)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def predict(city_name: str, fetch_prices: bool = False,
            fresh_prices: bool = False) -> Dict[str, Any]:
    city_key = resolve_city(city_name)
    meta     = CITIES.get(city_key) if city_key else None
    if not meta:
        return {"city": city_name, "error": f"Unknown city: '{city_name}'"}

    lat, lon   = meta["lat"], meta["lon"]
    tz         = meta["tz"]
    use_f      = meta.get("f", False)
    settlement = meta.get("settlement", "metar")
    icao       = meta["icao"]
    use_hko    = settlement == "hko"
    sym        = "°F" if use_f else "°C"

    # ── resolve target date ──────────────────────────────────────────────────
    t           = resolve_target(city_key)
    target_date = t["target_date"]
    target_idx  = t["target_idx"]
    predicting  = t["predicting"]           # "today" or "tomorrow"
    local_now   = t["local_now"]
    local_hour  = t["local_hour"]
    local_date  = t["local_date"]
    is_tomorrow = predicting == "tomorrow"

    # ── timing advice (is now a good time?) ──────────────────────────────────
    timing = timing_advice(city_key, is_tomorrow=is_tomorrow)

    # ── fetch all sources in parallel ────────────────────────────────────────
    res = {}
    def _om():  res["om"]    = fetch_open_meteo(lat, lon, use_f, target_idx)
    def _ens(): res["ens"]   = fetch_ensemble(lat, lon, use_f, target_idx)
    def _mm():  res["mm"]    = fetch_multi_model(lat, lon, use_f, target_idx)
    def _mn():  res["mn"]    = fetch_metno(lat, lon, tz, use_f, target_date)
    def _nws(): res["nws"]   = fetch_nws(lat, lon, use_f, target_date)
    def _7t():  res["t7"]    = fetch_7timer(lat, lon, tz, use_f, target_date)
    _provider = _OBS_PROVIDERS.get(meta.get("obs"))
    def _met():
        # If the city settles on a specific non-airport station (e.g. Hong Kong →
        # HKO Observatory), pull live obs from THAT station — it's the settlement
        # source. Otherwise: Wunderground first, METAR fallback.
        if _provider:
            obs = _provider["live"](city_key, tz, use_f, local_date)
            if obs:
                res["metar"] = obs
                return
        # Wunderground at the settlement station (wu_station overrides the airport
        # ICAO when Polymarket settles on a different WU station), METAR fallback.
        wu = fetch_wunderground(meta.get("wu_station") or icao, tz, use_f, local_date)
        res["metar"] = wu if wu else fetch_metar(icao, tz, local_date, use_f)

    def _air():
        # Raw airport METAR (aviationweather.gov — the same feed metar-taf.com
        # shows). Fetched in its OWN thread so it runs in parallel with _met
        # instead of serialising two HTTP calls. Cached, so if _met also needed
        # METAR this is a cache hit. Lets the alert show the exact station reading.
        res["airport_metar"] = fetch_metar(icao, tz, local_date, use_f)

    # Free no-key sources, each toggleable via ENABLE_<SOURCE>. _met (live obs) and
    # _air (raw airport METAR) are always on — they provide max_so_far, not forecasts.
    fetchers = [_met, _air]
    if ENABLE_OPEN_METEO: fetchers.append(_om)
    if ENABLE_ENSEMBLE:   fetchers.append(_ens)
    if ENABLE_MULTIMODEL: fetchers.append(_mm)
    if ENABLE_METNO:      fetchers.append(_mn)
    if ENABLE_NWS:        fetchers.append(_nws)
    if ENABLE_7TIMER:     fetchers.append(_7t)

    # Optional keyed sources, configured via Railway env vars. Each one that has a
    # key set adds an independent forecast (auto-converted to the city's unit).
    keyed = _keyed_sources()
    def _make_keyed(label, fn, key):
        def run():
            try:
                res[label] = fn(lat, lon, tz, use_f, target_date, key)
            except Exception:
                res[label] = None
        return run
    fetchers += [_make_keyed(label, fn, key) for label, fn, key in keyed]

    threads = [threading.Thread(target=fn) for fn in fetchers]
    for t_ in threads: t_.start()
    for t_ in threads: t_.join(timeout=15)

    om      = res.get("om") or {}
    ens     = res.get("ens") or {}
    multi   = res.get("mm") or {}
    metar   = res.get("metar") or {}
    airport = res.get("airport_metar") or {}   # raw aviationweather.gov station read

    # ── build forecasts dict using target day ────────────────────────────────
    forecasts: Dict[str, float] = {}

    om_target = _sf(om.get("_target_max"))
    if om_target is not None:
        forecasts["Open-Meteo"] = om_target

    for model, val in (multi or {}).items():
        if val is not None:
            forecasts[model] = round(float(val), 1)

    # MET Norway (Yr) — independent ECMWF-based source, free + no key. Always
    # included when available so the blend is genuinely multi-provider; it also
    # carries the forecast alone when Open-Meteo is rate-limited (429).
    metno_val = _sf(res.get("mn"))
    if metno_val is not None:
        forecasts["MET.no"] = round(metno_val, 1)

    # US NWS (weather.gov) — best-quality US source; None for non-US cities.
    nws_val = _sf(res.get("nws"))
    if nws_val is not None:
        forecasts["NWS"] = round(nws_val, 1)

    # 7Timer — free global third opinion, so non-US cities still get ≥2 sources
    # (MET.no + 7Timer) even when Open-Meteo is rate-limited.
    t7_val = _sf(res.get("t7"))
    if t7_val is not None:
        forecasts["7Timer"] = round(t7_val, 1)

    # Optional keyed sources (already converted to the city's unit by each fetcher).
    for label, _fn, _key in keyed:
        v = _sf(res.get(label))
        if v is not None:
            forecasts[label] = round(v, 1)

    # ── Validate every source before blending ────────────────────────────────
    # Drops garbage, AUTO-CORRECTS wrong-unit values, and flags broken APIs so a
    # newly-configured source can never silently poison the consensus.
    forecasts, source_warnings = _vet_forecasts(forecasts, use_f)
    for w in source_warnings:
        print(f"  ⚠️ source check [{city_key}] — {w}")

    # ── DEB blend ────────────────────────────────────────────────────────────
    deb, deb_weights, peak_bias = deb_blend(city_key, forecasts, target_date=target_date)
    # Raw blend BEFORE any bias was applied — so we can show both numbers.
    deb_raw = round(deb - peak_bias, 1) if deb is not None else None
    # USE_NOBIAS: decide on the RAW blend (drop the bias from the trading center).
    if USE_NOBIAS and deb is not None:
        deb = deb_raw
        peak_bias = 0.0

    # ── ensemble spread ───────────────────────────────────────────────────────
    p10     = _sf(ens.get("p10"))
    p90     = _sf(ens.get("p90"))
    ens_med = _sf(ens.get("median"))
    if p10 and p90:
        # Best case: real ensemble P10–P90 spread.
        sigma = max(0.3, (p90 - p10) / 2.56)
    else:
        # Ensemble endpoint unavailable (e.g. rate-limited). Do NOT fall back to a
        # blanket σ=1.2 — that always trips the `σ>0.7` uncertainty gate below and
        # forces EVERY city to WAIT, so the bot can never trade while the ensemble
        # API is down. Instead estimate σ from the spread of the individual NWP
        # models we DID get (ECMWF/GFS/ICON/GEM/Open-Meteo): tight model agreement
        # → low σ (tradeable), wide disagreement → high σ (correctly cautious).
        _fvals = [v for v in forecasts.values() if v is not None]
        if len(_fvals) >= 2:
            _mean  = sum(_fvals) / len(_fvals)
            _stdev = (sum((x - _mean) ** 2 for x in _fvals) / len(_fvals)) ** 0.5
            # Floor at 0.5° (models are correlated and understate true uncertainty);
            # cap at 2.0° so one outlier model can't peg σ absurdly high.
            sigma = max(0.5, min(2.0, _stdev))
        else:
            # Truly no spread information (≤1 model) — stay cautious.
            sigma = 1.2

    # Fallback: if the multi-model fetch returned nothing, deb_blend gave None.
    # Use the ensemble median so the prediction never shows "None°C" and the
    # downstream probability math still has a center to work from.
    if deb is None:
        deb = ens_med if ens_med is not None else _sf(om.get("_target_max"))
        if deb is not None:
            deb_raw = round(deb, 1)                    # the fallback center, no bias
            if not USE_NOBIAS:
                deb = round(deb + DEFAULT_PEAK_BIAS, 1)
                peak_bias = DEFAULT_PEAK_BIAS
            deb_weights = "ensemble-median fallback (multi-model fetch empty)"

    # ── live METAR (always today's observations) ──────────────────────────────
    cur_temp   = _sf(metar.get("current_temp"))
    max_so_far = _sf(metar.get("max_so_far"))
    trend      = metar.get("trend", "unknown")
    recent     = metar.get("recent_temps", [])

    # raw airport METAR reading, surfaced alongside the primary source so a
    # divergence (e.g. HKO observatory vs the airport ~35 km away) is visible.
    air_temp = _sf(airport.get("current_temp"))
    air_max  = _sf(airport.get("max_so_far"))
    live_source_disagree = (
        metar.get("source") in ("wunderground", "hko")
        and max_so_far is not None and air_max is not None
        and abs(max_so_far - air_max) >= 1.0
    )

    # ── peak window ───────────────────────────────────────────────────────────
    fp, lp      = estimate_peak_window(lat)
    peak_end    = meta.get("peak_end", 17)
    peak_status = get_peak_status(local_hour, fp, lp) if not is_tomorrow else "before"

    # ── dead market (only for today predictions) ──────────────────────────────
    dead = is_dead_market_check(local_hour, peak_end, max_so_far, cur_temp, is_tomorrow)

    # ── LIVE vs MODEL conflict detection ─────────────────────────────────────
    live_model_conflict = False
    stale_reading = False
    conflict_gap = None
    if not is_tomorrow and max_so_far is not None and deb is not None:
        conflict_gap = max_so_far - deb
        # live reading is >1.5° above what every model predicts
        if conflict_gap >= 1.5 and peak_status in ("before", "in_window"):
            live_model_conflict = True
        # Stale check: flat reading is ONLY suspicious when it does NOT match
        # the models. A temp sitting flat AT its peak (and matching the model
        # consensus) is normal plateau behaviour, not stale data.
        if recent and len(recent) >= 3:
            vals = [r[1] for r in recent[:3]]
            flat = len(set(vals)) == 1
            # how far the flat live value is from the model blend
            gap_from_model = abs(max_so_far - deb)
            # flat AND far from models (>1.5°) AND not yet at peak = likely cached
            if flat and gap_from_model >= 1.5 and peak_status == "before":
                stale_reading = True
            # flat reading that MATCHES models during/after peak = legit plateau, NOT stale

    # ── compute mu (probability center) ─────────────────────────────────────
    if dead and max_so_far is not None:
        mu = max_so_far
    elif is_tomorrow:
        # For tomorrow: use DEB or ensemble median — ignore today's live temp
        mu = deb or ens_med or om_target
    elif max_so_far is not None and deb is not None:
        if peak_status in ("past", "in_window") and max_so_far < (deb or 0) - 2.0:
            mu = max_so_far if (trend == "falling" or peak_status == "past") else max_so_far + 0.5
        elif live_model_conflict:
            # Live reading suspiciously high vs models → DON'T blindly trust it.
            # Blend halfway between models and live, and widen uncertainty.
            mu = (deb + max_so_far) / 2.0
            sigma = max(sigma, abs(conflict_gap) / 2.0)  # widen σ to reflect doubt
        else:
            mu = deb
            if max_so_far > mu:
                # Live obs has already exceeded the forecast → the forecast ran low.
                # If the peak is still forming and the temp is RISING, more climb is
                # coming before it tops out — anticipate it instead of trailing by
                # 0.3° (this is the "airport already at 33 climbing to 34, but the
                # model said 32" failure). Past peak / not rising → just track live.
                if peak_status == "in_window" and trend == "rising":
                    mu = max_so_far + 1.0
                else:
                    mu = max_so_far + (0.3 if trend != "falling" else 0.0)
            elif (max_so_far < mu and peak_status in ("in_window", "past")
                  and trend in ("stagnant", "falling")):
                # Live obs is BELOW the forecast and has STALLED (or is falling)
                # during/after the peak → the forecast overshot. Pull the centre
                # toward the live level and widen σ so the call isn't over-confident
                # on the forecast bucket (Milan: forecast 34.9, live stuck at 34 →
                # market correctly leaning 34, not 35).
                gap = mu - max_so_far
                if peak_status == "past" or trend == "falling":
                    mu = max_so_far + 0.2            # peak essentially in
                    sigma = max(sigma, gap * 0.7)    # genuinely uncertain
                else:
                    # in_window + stagnant: the temp can STILL climb to the forecast
                    # before the window closes, so only pull toward the stalled obs in
                    # proportion to how much of the peak window has ELAPSED. Early in
                    # the window a flat reading just means the day's heat hasn't arrived
                    # yet (Madrid 14:53: obs stuck at 32° but 10 models AND the market
                    # say 34-35°) — keep mu near the forecast and pull harder only as
                    # the window runs out. Previously this hedged to obs+0.45·gap the
                    # instant the window opened, dumping a 34.8° call onto 32-33°.
                    span    = max(1.0, lp - fp)
                    try:
                        hour_f = local_now.hour + local_now.minute / 60.0
                    except Exception:
                        hour_f = float(local_hour)
                    elapsed = min(1.0, max(0.0, (hour_f - fp) / span))
                    mu      = mu - gap * 0.45 * elapsed
                    sigma   = max(sigma, gap * 0.7 * (0.4 + 0.6 * elapsed))
    else:
        mu = deb or ens_med or om_target

    # Time-decay sigma (only for today predictions)
    if not is_tomorrow:
        if   peak_status == "past":      sigma *= 0.3
        elif peak_status == "in_window": sigma *= 0.7

    # ── probability distribution ──────────────────────────────────────────────
    dist_raw = []
    nobias_note = None
    if dead and max_so_far is not None:
        sv   = settlement_round(city_key, max_so_far)
        dist = [{"value": sv, "probability": 1.0, "range": "LOCKED"}]
    elif mu is not None:
        dist = compute_probabilities(mu, sigma,
                                     max_so_far if not is_tomorrow else None,
                                     city_key, use_hko, is_tomorrow)
        # Same distribution with the bias removed (centre shifted down by it), so
        # the alert can show what the model says BEFORE the per-city bias nudge.
        # Skip it when LIVE obs already drive the prediction (max_so_far >= blend):
        # there the bias is irrelevant and a "no-bias" graph would be misleading.
        bias_relevant = (is_tomorrow or max_so_far is None
                         or (deb is not None and max_so_far < deb))
        if peak_bias and bias_relevant:
            dist_raw = compute_probabilities(mu - peak_bias, sigma,
                                             max_so_far if not is_tomorrow else None,
                                             city_key, use_hko, is_tomorrow)
        elif peak_bias and max_so_far is not None:
            # Bias exists but live obs already drive the call — say so instead of
            # silently dropping the no-bias graph.
            nobias_note = (f"live {settlement_round(city_key, max_so_far)}{sym} already "
                           f"≥ blend — this call is driven by live obs, not the bias")
    else:
        dist = []

    # ── boundary alert (only for today, only when max is known) ──────────────
    boundary = None
    if not is_tomorrow and max_so_far is not None:
        frac = max_so_far - int(max_so_far)
        sv   = settlement_round(city_key, max_so_far)
        if use_hko:
            d = 1.0 - frac
            if d <= 0.3:
                boundary = f"only {d:.1f}° from rounding UP to {(sv or 0)+1}!"
        else:
            d = abs(frac - 0.5)
            if d <= 0.3:
                boundary = (
                    f"only {0.5-frac:.1f}° from rounding UP to {(sv or 0)+1}!"
                    if frac < 0.5 else
                    f"only {frac-0.5:.1f}° from rounding DOWN to {(sv or 0)-1}!"
                )

    # ── Range-bucket markets (e.g. SF '68-69°F') ─────────────────────────────
    # Fetch the market now (before the verdict) and, if it uses multi-degree
    # ranges, re-bin the model's single-degree distribution onto the market's
    # buckets so prob / verdict / edges all reflect the RANGE, not one degree.
    pm_data = None
    range_market = False
    if fetch_prices:
        try:
            # fresh_prices=True (e.g. a 🔄 Refresh tap) bypasses the price cache so
            # the cover/edges are computed on the live order book, not a stale copy.
            pm_data = fetch_polymarket_market(city_key, target_date,
                                              cache_ttl=0 if fresh_prices else None)
        except Exception:
            pm_data = None
        if pm_data and _market_is_range(pm_data):
            range_market = True
            dist     = _remap_distribution(dist, pm_data)
            dist_raw = _remap_distribution(dist_raw, pm_data)

    top        = dist[0] if dist else {}
    model_prob = top.get("probability") or 0.0
    confidence = "HIGH" if model_prob >= 0.60 else "MEDIUM" if model_prob >= 0.40 else "LOW"

    # ── model vs ensemble agreement check ────────────────────────────────────
    # If individual models (DEB) and the ensemble median disagree a lot,
    # the forecast is unreliable — flag it.
    agreement = "unknown"
    disagreement = None
    if deb is not None and ens_med is not None:
        disagreement = abs(deb - ens_med)
        if disagreement <= 0.7:
            agreement = "strong"
        elif disagreement <= 1.5:
            agreement = "moderate"
        else:
            agreement = "weak"

    # ── boundary risk: is mu sitting on a rounding edge? ──────────────────────
    # Irrelevant for range-bucket markets — a mid-range µ isn't a coin-flip there.
    on_boundary = False
    if mu is not None and not use_hko and not range_market:
        frac = mu - math.floor(mu)
        # WU rounds at .5; danger zone is .35–.65 (model can't tell which bucket)
        on_boundary = 0.35 <= frac <= 0.65

    # ── TRADE VERDICT — the smart decision ───────────────────────────────────
    # Combines: confidence, sigma, model agreement, boundary risk, timing
    reasons = []
    verdict = "TRADE"

    if dead:
        verdict = "TRADE"
        reasons.append("dead market — outcome locked")
    else:
        if model_prob < 0.55:
            verdict = "SKIP"
            reasons.append(f"top bucket only {model_prob*100:.0f}% (need ≥55%)")
        if sigma > 0.7:
            if verdict != "SKIP":
                verdict = "WAIT"
            reasons.append(f"high uncertainty σ={sigma:.2f}")
        if agreement == "weak":
            verdict = "SKIP"
            reasons.append(f"models disagree by {disagreement:.1f}° (DEB {deb} vs ens {ens_med})")
        elif agreement == "moderate" and verdict == "TRADE":
            # Moderate model spread only matters when the bucket is BORDERLINE.
            # If the top bucket is already high-probability (≥75%), a moderate
            # spread between DEB and ensemble doesn't change the call — the
            # distribution still concentrates on one bucket. Only downgrade
            # moderate-agreement signals when probability is in the soft zone.
            if model_prob < 0.75:
                verdict = "WAIT"
                reasons.append(f"models differ {disagreement:.1f}° and prob only {model_prob*100:.0f}%")
            # else: high prob + moderate spread → still tradeable, no downgrade
        if on_boundary and verdict == "TRADE":
            verdict = "WAIT"
            reasons.append(f"μ={mu:.1f} on rounding boundary (coin-flip between buckets)")
        # live-vs-model conflict: live reading way above forecast consensus
        if live_model_conflict:
            verdict = "WAIT"
            reasons.append(f"live obs ({max_so_far:.0f}°) is {conflict_gap:.1f}° above model consensus "
                          f"(DEB {deb:.1f}°) — either unusually hot OR stale reading")
        if stale_reading and verdict == "TRADE":
            verdict = "WAIT"
            reasons.append(f"live reading flat ({max_so_far:.0f}° x3) — possibly stale/cached data")
        # ── timing / reliability gate ──
        # A 70% signal pre-peak is NOT reliable — the day's high hasn't formed.
        # Only allow TRADE when the peak is forming (FIRMING) or done (RELIABLE).
        tq = timing.get("quality")
        if not is_tomorrow and verdict == "TRADE":
            if tq in ("SPECULATIVE", "OVERNIGHT"):
                verdict = "WAIT"
                reasons.append(f"pre-peak — day's high not formed yet "
                              f"(wait until ~{timing.get('peak_window','peak')})")
            elif tq == "FIRMING":
                # tradeable but tag as not-yet-fully-reliable
                reasons.append("peak forming — firming up, watch live obs")
        if is_tomorrow and verdict == "TRADE":
            # tomorrow's market is always forecast-only — never fully reliable yet
            reasons.append("tomorrow's market — forecast only, recheck near its peak")

    if verdict == "TRADE" and not reasons:
        reasons.append("clear signal — strong bucket, low uncertainty, models agree")

    # ── Polymarket edge calculation ──────────────────────────────────────────
    # pm_data was already fetched above (and the distribution re-binned to the
    # market's buckets for range markets), so reuse it — no second fetch.
    edges   = []
    best_trade = None
    if fetch_prices and pm_data:
        try:
            # Confidence (0-1) from how clean the read is: shaky reads (high σ,
            # models disagree, pre-peak) must clear a HIGHER edge bar to trade.
            conf = 1.0
            if agreement == "weak":
                conf *= 0.55
            elif agreement == "moderate":
                conf *= 0.80
            if sigma and sigma > 0.7:
                conf *= 0.80
            if not timing.get("reliable", False):
                conf *= 0.85
            edges = compute_edges(dist, pm_data, sym, confidence=conf)
            # best actionable trade = highest edge that cleared the bar
            for e in edges:
                if e["action"] in ("BUY YES", "BUY NO"):
                    best_trade = e
                    break
        except Exception:
            pass

    # ── MARKET-DECIDED GUARD ──────────────────────────────────────────────────
    # The market is the source of truth once it concentrates on a bucket. If one
    # bucket's YES is very high and it's NOT the bucket our model favours, the
    # model is predicting a peak the market has already settled past — the "edge"
    # is fake (e.g. Wellington 15°@73% "edge+69%" while the market had 16°@97%
    # and priced 15° at 0.1¢). Trust the market: kill the trade.
    market_decided = False
    pm_top_bucket  = None
    pm_top_yes     = None
    model_bucket   = top.get("value")
    if pm_data and pm_data.get("buckets"):
        for b_temp, b in pm_data["buckets"].items():
            y = _sf(b.get("yes"))
            if y is not None and (pm_top_yes is None or y > pm_top_yes):
                pm_top_yes, pm_top_bucket = y, b_temp
        if pm_top_yes is not None and pm_top_bucket != model_bucket:
            if pm_top_yes >= MARKET_DECIDED_YES:
                market_decided = True
                verdict = "SKIP"
                reasons.append(
                    f"market already decided: {pm_top_bucket}{sym} at "
                    f"{pm_top_yes*100:.0f}¢ — model's {model_bucket}{sym} disagrees, "
                    f"model is wrong here")
                best_trade = None
            elif pm_top_yes >= MARKET_LEAN_YES and verdict == "TRADE":
                verdict = "WAIT"
                reasons.append(
                    f"market leans {pm_top_bucket}{sym} at {pm_top_yes*100:.0f}¢ "
                    f"but model favours {model_bucket}{sym} — wait for them to agree")

    # ── save forecasts to DEB history ─────────────────────────────────────────
    if forecasts:
        history = _load_history()
        history.setdefault(city_key, {}).setdefault(target_date, {})["forecasts"] = forecasts
        _save_history(history)

    return {
        # identity
        "city":           city_key,
        "display":        city_name.title(),
        "local_datetime": local_now.strftime("%Y-%m-%d %H:%M"),
        "local_date":     local_date,
        "target_date":    target_date,       # ← the market date on Polymarket
        "predicting":     predicting,        # "today" or "tomorrow"
        "temp_unit":      sym,
        "settlement":     settlement.upper(),
        "icao":           icao,
        # timing
        "timing":         timing,
        "reliable":       timing.get("reliable", False),
        # forecasts
        "forecasts":      forecasts,
        "source_warnings": source_warnings,
        "deb":            deb,
        "deb_raw":        deb_raw,          # blend WITHOUT peak bias
        "peak_bias":      round(peak_bias, 2),
        "no_bias_mode":   USE_NOBIAS,
        "deb_weights":    deb_weights,
        "ensemble":       {"p10": p10, "median": ens_med, "p90": p90,
                           "members": ens.get("members", 0)},
        "agreement":      agreement,
        "disagreement":   round(disagreement, 1) if disagreement is not None else None,
        "on_boundary":    on_boundary,
        "live_model_conflict": live_model_conflict,
        "stale_reading":  stale_reading,
        "conflict_gap":   round(conflict_gap, 1) if conflict_gap is not None else None,
        # live (always today's observations)
        "live": {
            "current_temp": cur_temp,
            "max_so_far":   max_so_far,
            "obs_time":     metar.get("obs_time"),
            "trend":        trend,
            "recent":       recent[:3],
            "source":       metar.get("source", "metar"),
            # raw airport station read (aviationweather.gov METAR = metar-taf.com)
            "airport_icao":     icao,
            "airport_temp":     air_temp,
            "airport_max":      air_max,
            "airport_obs_time": airport.get("obs_time"),
        },
        "live_source_disagree": live_source_disagree,
        # analysis
        "mu":             round(mu, 2) if mu is not None else None,
        "sigma":          round(sigma, 2),
        "peak_window":    f"{fp:02d}:00–{lp:02d}:00",
        "peak_status":    peak_status,
        "is_dead_market": dead,
        # result
        "distribution":   dist[:6],
        "distribution_raw": dist_raw[:6],     # no-bias version of the distribution
        "nobias_note":    nobias_note,        # why no-bias graph is omitted (if so)
        "top_bucket":     top.get("value"),
        "top_lo":         top.get("lo"),
        "top_hi":         top.get("hi"),
        "range_market":   range_market,
        "top_prob":       model_prob,
        "confidence":     confidence,
        "boundary_alert": boundary,
        "settled_pred":   settlement_round(city_key, mu) if mu is not None else None,
        # ── THE SMART VERDICT ──
        "verdict":        verdict,    # TRADE / WAIT / SKIP
        "verdict_reasons": reasons,
        # ── POLYMARKET LIVE ──
        "polymarket":     pm_data,
        "edges":          edges,
        "best_trade":     best_trade,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
CONF_EMOJI  = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
TREND_EMOJI = {"rising": "📈", "falling": "📉", "stagnant": "➡️", "unknown": "❓"}

def print_positions(wallet: str, positions: Optional[List[Dict]], weather_only: bool = True):
    """Print your live Polymarket weather positions with P&L."""
    masked = wallet[:6] + "…" + wallet[-4:] if wallet else "?"
    scope  = "WEATHER" if weather_only else "ALL"
    print(f"\n{'═'*70}")
    print(f"  💼 YOUR POLYMARKET POSITIONS [{scope}] — wallet {masked}")
    print(f"{'═'*70}")

    if positions is None:
        print(f"  ❌ Could not fetch positions (check wallet address / network)")
        print(f"{'─'*70}")
        return
    if not positions:
        print(f"  (no open {scope.lower()} positions found)")
        print(f"{'─'*70}")
        return

    total_paid = 0.0
    total_now  = 0.0
    total_pnl  = 0.0

    print(f"  {'MARKET':<38} {'SIDE':>4} {'SHARES':>7} {'ENTRY':>6} {'NOW':>5} {'VALUE':>7} {'P&L':>8}")
    print(f"  {'─'*38} {'─'*4} {'─'*7} {'─'*6} {'─'*5} {'─'*7} {'─'*8}")
    for p in positions:
        title = p["title"][:37]
        side  = (p.get("outcome") or "?")[:3]
        shares= p.get("size") or 0
        entry = p.get("avg_price")
        now   = p.get("cur_price")
        val   = p.get("current_value") or 0
        pnl   = p.get("cash_pnl") or 0
        ppnl  = p.get("percent_pnl") or 0

        total_paid += p.get("initial_value") or 0
        total_now  += val
        total_pnl  += pnl

        entry_s = f"{entry*100:.0f}¢" if entry is not None else "—"
        now_s   = f"{now*100:.0f}¢"   if now   is not None else "—"
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        redeem  = " ✅claimable" if p.get("redeemable") else ""

        print(f"  {title:<38} {side:>4} {shares:>7.1f} {entry_s:>6} {now_s:>5} "
              f"${val:>6.2f} {pnl_emoji}${pnl:>+6.2f}{redeem}")

    print(f"  {'─'*70}")
    tot_emoji = "🟢" if total_pnl >= 0 else "🔴"
    roi = (total_pnl / total_paid * 100) if total_paid > 0 else 0
    print(f"  {'TOTAL':<38} {'':>4} {'':>7} {'':>6} {'':>5} "
          f"${total_now:>6.2f} {tot_emoji}${total_pnl:>+6.2f}")
    print(f"  Invested: ${total_paid:.2f}   Now worth: ${total_now:.2f}   "
          f"P&L: {tot_emoji} ${total_pnl:+.2f} ({roi:+.1f}%)")
    print(f"{'═'*70}")

    # claimable winnings
    claimable = [p for p in positions if p.get("redeemable")]
    if claimable:
        print(f"\n  ✅ {len(claimable)} position(s) settled & claimable — redeem on Polymarket:")
        for p in claimable:
            print(f"     • {p['title'][:50]}  → ${p.get('current_value',0):.2f}")
    print()


def print_city_detail(p: Dict):
    if "error" in p:
        print(f"\n❌ {p['city']}: {p['error']}")
        return

    sym  = p["temp_unit"]
    live = p.get("live") or {}
    ens  = p.get("ensemble") or {}
    dist = p.get("distribution") or []
    ce   = CONF_EMOJI.get(p.get("confidence", "LOW"), "⚪")
    te   = TREND_EMOJI.get(live.get("trend", "unknown"), "❓")
    tmrw = p.get("predicting") == "tomorrow"
    tim  = p.get("timing") or {}
    date_label = f"📅 {p.get('target_date')} ({'TOMORROW' if tmrw else 'TODAY'})"

    print(f"\n{'═'*62}")
    print(f"  📍 {p['city'].upper():<20} {p['local_datetime']}  [{p['settlement']}]")
    print(f"  {date_label}  ← this is the Polymarket market date")
    print(f"{'═'*62}")

    # ── TIMING BANNER ──
    q = tim.get("quality", "?")
    q_emoji = {"RELIABLE": "🟢", "FIRMING": "🟡", "SPECULATIVE": "🟠", "OVERNIGHT": "🔴"}.get(q, "⚪")
    print(f"\n  {q_emoji} TIMING [{q}] — city local time {tim.get('city_local_now','?')}")
    print(f"     {tim.get('message','')}")
    if tim.get("next_run_user_local"):
        print(f"     ⏰ Best to run again at: {tim['next_run_user_local']} YOUR time "
              f"(= {tim.get('next_run_city_local')} {p['city']} time, in ~{tim.get('hours_until_golden')}h)")

    if p.get("forecasts"):
        print(f"\n  📊 MODEL FORECASTS (for {p.get('target_date')}):")
        for m, v in p["forecasts"].items():
            print(f"     {m:<15} {v}{sym}")

    if p.get("deb") is not None:
        print(f"\n  🧬 DEB BLEND: {p['deb']}{sym}")
        print(f"     {p['deb_weights']}")

    if ens.get("p10") is not None:
        print(f"  📉 ENSEMBLE:  P10={ens['p10']}{sym}  "
              f"Median={ens['median']}{sym}  P90={ens['p90']}{sym}  "
              f"({ens.get('members',0)} members)")

    # Model agreement
    agr = p.get("agreement", "unknown")
    if agr != "unknown":
        agr_emoji = {"strong": "✅", "moderate": "⚠️", "weak": "❌"}.get(agr, "")
        dis = p.get("disagreement")
        print(f"  {agr_emoji} MODEL AGREEMENT: {agr.upper()} "
              f"(DEB vs ensemble differ by {dis}°)")

    # Live always shows TODAY's observations
    if live.get("current_temp") is not None:
        src = live.get("source", "metar")
        src_label = "🎯 Wunderground (settlement source)" if src == "wunderground" else "METAR (aviation)"
        obs_label = "TODAY live obs" if not tmrw else "TODAY live obs (context only)"
        print(f"\n  🌡️  {obs_label} [{live.get('obs_time','?')}] — {src_label}:")
        print(f"     Current={live['current_temp']}{sym}  "
              f"Max so far={live.get('max_so_far','?')}{sym}  {te} {live.get('trend','?')}")
        if live.get("recent"):
            print(f"     " + "  →  ".join(f"{t}{sym}@{tm}" for tm, t in live["recent"]))

    print(f"\n  🔍 μ={p.get('mu')}{sym}  σ={p.get('sigma')}{sym}  "
          f"Peak:{p.get('peak_window')}  Status:[{p.get('peak_status').upper()}]")

    if p.get("is_dead_market"):
        print(f"  🔒 DEAD MARKET — today's max is already locked in!")
    if p.get("boundary_alert"):
        print(f"  ⚖️  BOUNDARY: {p['boundary_alert']}")
    if p.get("on_boundary"):
        print(f"  ⚠️  μ sits on a rounding boundary — model can't tell which bucket wins")
    if p.get("live_model_conflict"):
        print(f"  🚨 CONFLICT: live obs is {p.get('conflict_gap')}° ABOVE model consensus")
        print(f"     → Either today is unusually hot OR the live reading is stale/cached.")
        print(f"     → μ was blended (not blindly anchored to live) and σ widened. Trust reduced.")
    if p.get("stale_reading"):
        print(f"  🚨 STALE: live reading is flat (same value 3x) — likely cached/coarse data")
    if tmrw:
        print(f"  ℹ️  Predicting TOMORROW — today's peak window already passed")

    if dist:
        print(f"\n  🎲 SETTLEMENT PROBABILITIES (market date: {p.get('target_date')}):")
        for b in dist:
            bar = "█" * int(b["probability"] * 32)
            print(f"     {b['value']:>4}{sym}  {bar:<32} {b['probability']*100:5.1f}%")

    # ── THE VERDICT ──
    verdict = p.get("verdict", "?")
    v_emoji = {"TRADE": "🟢🟢", "WAIT": "🟠", "SKIP": "🔴"}.get(verdict, "⚪")
    print(f"\n  {v_emoji} VERDICT: {verdict}")
    for r in p.get("verdict_reasons", []):
        print(f"     • {r}")

    # ── LIVE POLYMARKET EDGE TABLE ──
    pm    = p.get("polymarket")
    edges = p.get("edges") or []
    if pm and edges:
        print(f"\n  💰 LIVE POLYMARKET EDGE  ({pm.get('title','')[:40]}...)")
        print(f"     {'BUCKET':>7}  {'MODEL':>6}  {'YES':>5}  {'NO':>5}  "
              f"{'EDGE':>6}  {'ACTION':>8}  {'VOL':>8}")
        print(f"     {'─'*7}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}")
        for e in edges[:8]:
            yes_s  = f"{e['yes_price']*100:.0f}¢" if e.get("yes_price") is not None else "  —"
            no_s   = f"{e['no_price']*100:.0f}¢"  if e.get("no_price")  is not None else "  —"
            best   = e["best_edge"]
            be_s   = f"{best*100:+.0f}%"
            act    = e["action"]
            act_e  = "🟢" if act in ("BUY YES","BUY NO") else "  "
            flag   = ""
            if e.get("thin"):
                flag = " ⚠️thin"
            # mark unbuyable extremes
            if act == "skip" and not e.get("yes_buyable") and not e.get("no_buyable"):
                flag = " (priced out)"
            print(f"     {e['temp']:>5}{sym}  {e['model_prob']*100:>5.0f}%  "
                  f"{yes_s:>5}  {no_s:>5}  {be_s:>6}  {act_e}{act:>6}  ${e['vol']:>7,.0f}{flag}")
        bt = p.get("best_trade")
        if bt:
            print(f"\n     🎯 BEST TRADE: {bt['action']} on {bt['temp']}{sym} "
                  f"@ {bt['yes_price']*100:.0f}¢ → {bt['best_edge']*100:+.0f}% edge "
                  f"(model {bt['model_prob']*100:.0f}%)")
            if bt.get("thin"):
                print(f"     ⚠️  Low volume (${bt['vol']:,.0f}) — keep size tiny, you may move the price")
            print(f"     🔗 {pm.get('url')}")
        else:
            print(f"\n     ⚠️  No clean tradeable edge — buckets with 'edge' are priced out (0¢/100¢)")
            print(f"        or the market already agrees with the model. Skip this one.")
            print(f"     🔗 {pm.get('url')}")
    elif pm is None and verdict == "TRADE":
        # prices weren't fetched (no --prices flag) or market not found
        print(f"\n  ℹ️  Run with --prices to auto-fetch Polymarket edge")

    if verdict == "TRADE" and not (pm and p.get("best_trade")):
        print(f"\n  {ce} MODEL SIGNAL: {p.get('top_bucket')}{sym} at {p.get('top_prob',0)*100:.0f}% "
              f"(check Polymarket {p.get('target_date')})")
    elif verdict == "WAIT":
        nxt = tim.get("next_run_user_local")
        print(f"\n  ⏳ HOLD OFF — signal not clean enough yet.")
        if nxt:
            print(f"     Re-run at {nxt} your time for a sharper read.")
    elif verdict == "SKIP":
        print(f"\n  ⛔ SKIP this city — no clean edge right now.")
    print(f"{'─'*62}")


def print_summary_table(results: List[Dict], min_prob: float = 0.0):
    valid  = [p for p in results if "error" not in p]
    errors = [p for p in results if "error" in p]

    # sort by verdict (TRADE first), then confidence, then prob
    vorder = {"TRADE": 0, "WAIT": 1, "SKIP": 2}
    order  = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    valid.sort(key=lambda p: (
        vorder.get(p.get("verdict", "SKIP"), 3),
        0 if p.get("is_dead_market") else 1,
        order.get(p.get("confidence", "LOW"), 2),
        -p.get("top_prob", 0),
    ))

    now_str = _now_utc().strftime("%Y-%m-%d %H:%M UTC")
    W = 104
    print(f"\n{'═'*W}")
    print(f"  🌍 POLYWEATHER SCAN — {len(valid)} cities — {now_str}")
    print(f"{'═'*W}")
    print(f"  {'VERDICT':>7} {'CITY':<15} {'LOCAL':>5}  {'MKT DATE':>10}  {'DEB':>7}  "
          f"{'BUCKET':>7}  {'PROB':>5}  {'AGREE':>6}  {'TIMING':>8}")
    print(f"  {'─'*7} {'─'*15} {'─'*5}  {'─'*10}  {'─'*7}  "
          f"{'─'*7}  {'─'*5}  {'─'*6}  {'─'*8}")

    V_EMOJI = {"TRADE": "🟢", "WAIT": "🟠", "SKIP": "🔴"}
    AGR_EMOJI = {"strong": "✅", "moderate": "⚠️", "weak": "❌", "unknown": "  "}

    shown = 0
    for p in valid:
        prob = p.get("top_prob", 0)
        if prob < min_prob:
            continue
        shown += 1
        sym    = p["temp_unit"]
        v      = p.get("verdict", "SKIP")
        ve     = V_EMOJI.get(v, "⚪")
        agr    = p.get("agreement", "unknown")
        ae     = AGR_EMOJI.get(agr, "  ")
        tim    = p.get("timing") or {}
        tq     = tim.get("quality", "?")

        deb_s  = f"{p['deb']}{sym}"          if p.get("deb")        is not None else "  —  "
        bkt_s  = f"{p.get('top_bucket')}{sym}" if p.get("top_bucket") is not None else "  —  "
        mkt_dt = p.get("target_date", "?")
        lhour  = (_now_utc() + timedelta(seconds=CITIES.get(p["city"], {}).get("tz", 0))).strftime("%H:%M")
        dead   = "🔒" if p.get("is_dead_market") else ""

        print(f"  {ve}{v:>5} {p['city']:<15} {lhour}  "
              f"{mkt_dt:>10}  {deb_s:>8}  {bkt_s:>7}  {prob*100:>4.0f}%  "
              f"{ae}{agr[:4]:>5}  {tq:>8} {dead}")

    if shown == 0:
        print(f"  (no cities with probability >= {min_prob*100:.0f}%)")

    print(f"{'─'*W}")
    print(f"  VERDICT: 🟢TRADE=act now  🟠WAIT=signal not clean  🔴SKIP=no edge")
    print(f"  AGREE: ✅strong ⚠️moderate ❌weak (DEB vs ensemble)")
    print(f"  TIMING: 🔴SPECULATIVE(pre-peak) 🟡FIRMING(peak now) 🟢RELIABLE(post-peak,locked)")
    print(f"{'═'*W}")

    # ── ACTIONABLE TRADES ──
    trades = [p for p in valid if p.get("verdict") == "TRADE" and p.get("top_prob", 0) >= min_prob]
    if trades:
        print(f"\n  ✅ TRADES YOU CAN TAKE NOW ({len(trades)}):")
        for p in trades:
            sym  = p["temp_unit"]
            dead = "🔒 DEAD MARKET " if p.get("is_dead_market") else ""
            bt   = p.get("best_trade")
            if bt:
                # we have live polymarket edge
                print(f"     🟢 {p['city'].upper():<16} [{p.get('target_date')}]  "
                      f"{bt['action']} {bt['temp']}{sym} @ {bt['yes_price']*100:.0f}¢  "
                      f"→ EDGE {bt['best_edge']*100:+.0f}%  "
                      f"(model {bt['model_prob']*100:.0f}%)  {dead}")
            else:
                edge_note = ""
                if p.get("polymarket") is not None:
                    edge_note = "  (no >10% edge — market efficient)"
                print(f"     🟢 {p['city'].upper():<16} [{p.get('target_date')}]  "
                      f"BUY YES = {p.get('top_bucket')}{sym}  "
                      f"({p.get('top_prob',0)*100:.0f}% prob){edge_note}  {dead}")
    else:
        print(f"\n  ⚠️  No clean TRADE signals right now.")

    # ── WAIT cities — show when to recheck ──
    waits = [p for p in valid if p.get("verdict") == "WAIT" and p.get("top_prob", 0) >= min_prob]
    if waits:
        print(f"\n  ⏳ WAIT — recheck these later ({len(waits)}):")
        for p in waits[:10]:
            tim = p.get("timing") or {}
            nxt = tim.get("next_run_user_local")
            reason = (p.get("verdict_reasons") or ["—"])[0]
            when = f"recheck {nxt} your time" if nxt else "recheck in golden window"
            print(f"     🟠 {p['city'].upper():<16} {reason}  →  {when}")

    if errors:
        print(f"\n  ⚠️  Failed ({len(errors)}): {', '.join(p['city'] for p in errors)}")

    print(f"\n  💡 For each TRADE: Polymarket → search city → click MKT DATE tab → find bucket")
    print(f"     Edge = model prob% - Polymarket YES price%  →  Buy YES if edge > 10%\n")


# ══════════════════════════════════════════════════════════════════════════════
# SCAN ALL
# ══════════════════════════════════════════════════════════════════════════════
def scan_all(cities: List[str], workers: int = 6, fetch_prices: bool = False) -> List[Dict]:
    results = []
    total   = len(cities)
    done    = [0]
    lock    = threading.Lock()

    print(f"\n  ⏳ Scanning {total} cities ({workers} workers)...")
    print(f"  Each city: auto-detects if predicting TODAY or TOMORROW\n")

    def _fetch_one(city):
        p = predict(city, fetch_prices=fetch_prices)
        with lock:
            done[0] += 1
            s   = "✓" if "error" not in p else "✗"
            pct = done[0] / total
            bar = "█" * int(pct * 30)
            print(f"  [{bar:<30}] {done[0]:>2}/{total}  {s} {city}", end="\r", flush=True)
        return p

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, c): c for c in cities}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"city": futures[f], "error": str(e)})

    print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="PolyWeather v2.0 — auto detects today vs tomorrow per city",
        epilog=(
            "Examples:\n"
            "  python polyweather_predict.py                        scan all 51 cities\n"
            "  python polyweather_predict.py tokyo london           specific cities\n"
            "  python polyweather_predict.py --min-prob 0.6         HIGH confidence only\n"
            "  python polyweather_predict.py --detail istanbul       full detail\n"
            "  python polyweather_predict.py --workers 12           faster scan\n"
            "  python polyweather_predict.py --json                 raw JSON\n"
            "  python polyweather_predict.py --list                 list all cities\n"
            "  python polyweather_predict.py --record-actual istanbul 2026-06-13 20"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cities",          nargs="*")
    parser.add_argument("--workers",       type=int,   default=6)
    parser.add_argument("--min-prob",      type=float, default=0.0)
    parser.add_argument("--detail",        nargs="+")
    parser.add_argument("--json",          action="store_true")
    parser.add_argument("--prices",        action="store_true",
                        help="Fetch live Polymarket prices and compute edge automatically")
    parser.add_argument("--list",          action="store_true")
    parser.add_argument("--record-actual", nargs=3, metavar=("CITY", "DATE", "TEMP"))
    parser.add_argument("--pmtest", nargs="+", metavar="CITY",
                        help="Debug Polymarket market lookup for a city (shows raw matching)")
    parser.add_argument("--positions", action="store_true",
                        help="Show YOUR live Polymarket weather positions + P&L (needs wallet)")
    parser.add_argument("--wallet", type=str, default=None,
                        help="Your public wallet address (0x...). Or set POLYMARKET_WALLET env var")
    parser.add_argument("--all-positions", action="store_true",
                        help="With --positions: show ALL positions, not just weather")
    args = parser.parse_args()

    if args.positions:
        wallet = get_wallet_address(args.wallet)
        if not wallet:
            print("\n  ❌ No wallet address provided.")
            print("     Either: python3 polyweather_predict.py --positions --wallet 0xYourAddress")
            print("     Or set once: export POLYMARKET_WALLET=0xYourAddress")
            print("\n  ℹ️  This only needs your PUBLIC address (0x...), never your private key.")
            print("     Find it on Polymarket → Profile → your address is shown there.\n")
            return
        weather_only = not args.all_positions
        positions = fetch_positions(wallet, weather_only=weather_only)
        print_positions(wallet, positions, weather_only=weather_only)
        return

    if args.pmtest:
        for city in args.pmtest:
            ck = resolve_city(city)
            if not ck:
                print(f"❌ Unknown city: {city}")
                continue
            t = resolve_target(ck)
            print(f"\n🔍 Polymarket lookup: {ck} → market date {t['target_date']}")
            pm = fetch_polymarket_market(ck, t["target_date"], debug=True)
            if pm:
                print(f"\n✅ FOUND: {pm['title']}")
                print(f"   URL: {pm['url']}")
                print(f"   Buckets + prices:")
                for temp in sorted(pm["buckets"].keys()):
                    b = pm["buckets"][temp]
                    yes = f"{b['yes']*100:.0f}¢" if b.get("yes") is not None else "—"
                    print(f"     {temp}°: YES={yes}  vol=${b.get('vol',0):,.0f}")
            else:
                print(f"\n❌ No market found. The bot will show model-only signal for this city.")
                print(f"   Possible reasons: market not listed yet, different title format,")
                print(f"   or Polymarket has no temperature market for this city/date.")
        return

    if args.list:
        print(f"\n  {'CITY':<20} {'ICAO':<6}  {'TZ':>7}s  UNIT  SETTLEMENT  PEAK_END")
        print(f"  {'─'*20} {'─'*6}  {'─'*8}  {'─'*4}  {'─'*10}  {'─'*8}")
        for c in sorted(CITIES.keys()):
            m = CITIES[c]
            print(f"  {c:<20} {m['icao']:<6}  {m['tz']:>8}  "
                  f"{'°F' if m.get('f') else '°C'}   {m['settlement'].upper():<10}  "
                  f"{m.get('peak_end',17):02d}:00")
        print()
        return

    if args.record_actual:
        city_a, date_a, temp_a = args.record_actual
        record_actual(city_a, date_a, float(temp_a))
        return

    if args.detail:
        for city in args.detail:
            p = predict(city, fetch_prices=args.prices)
            if args.json:
                print(json.dumps(p, indent=2))
            else:
                print_city_detail(p)
        return

    target  = args.cities if args.cities else list(CITIES.keys())
    unknown = [c for c in target if resolve_city(c) is None]
    if unknown:
        print(f"❌ Unknown: {unknown}  (run --list to see valid names)")
        return

    results = scan_all(target, workers=min(max(1, args.workers), 20), fetch_prices=args.prices)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary_table(results, min_prob=args.min_prob)


if __name__ == "__main__":
    main()
