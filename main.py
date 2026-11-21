#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Idea
----
A trophy's earn rate is a ratio: earners over players tracked. Both counts
grow, but on different clocks. Players tracked jumps whenever Exophase
refreshes the page (new players sync their profiles in bulk), which dilutes
every rate at once; earners grow as the players already tracked work through
the game, which lifts the rates of the trophies they reach. So a rate's path
is a tug of war: dilution pulls it down at each refresh, progression pulls
it back up between them, and which wins depends on the trophy. An early
story trophy is earned by most players within days, so its rate recovers
and climbs; a late or grindy trophy is earned by almost nobody in week one,
so dilution wins and its rate keeps sliding.

This program reads the case's history of reads, collapses it to the distinct
snapshots (the page sits flat between refreshes, so repeated reads carry no
new information), and fits, per trophy, a line of the logit's rate against
log time since release. The slope is that trophy's tug of war, shrunk toward
the average slope across trophies when the trophy has few distinct reads to
speak for itself. The line is read at the target, and the spread around it
is the fit's own prediction error plus a fixed allowance for the next
refresh landing where it lands. When the history is a single flat snapshot
(launch night, before any refresh), no slope can be fit, so the program
falls back to the one thing launch night teaches: the first refresh dilutes
the average rate by about a quarter, so the median sits below the last rate
and the spread is wide.

The forecast is deterministic: the same case always writes the same bytes.
Standard library only, no network, no clock beyond the case's own times.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZNORM = [-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722]

# A rate is clamped before its logit, as the scoring does.
LO, HI = 0.01, 99.99
# Spread floor and cap, in logit units: never claim more precision than the
# refresh noise allows, never spread wider than the history justifies.
SPREAD_MIN = 0.35
SPREAD_MAX = 1.20
# Fixed allowance for the next refresh arriving at an unknown moment,
# in logit units, added in quadrature to the fit's own error.
REFRESH_SD = 0.25
# When the history is a single flat snapshot, the median sits this far below
# the last rate on the logit scale, and the spread is this wide: launch night
# teaches the size of the first refresh. The house's own numbers say the
# first refresh took players tracked from 638 to 3,868 while trophies earned
# went from 943 to 4,276 -- and the trophy everyone holds (100%) accounts
# for 3,230 of the 3,333 new earners, so the other 54 trophies shared only
# about 103 new earners among 3,230 new players. Their average rate fell to
# about 0.22 of what it was, a logit shift near -1.5, and the spread covers
# how unevenly those few new earners land across trophies.
FLAT_SHIFT = -1.50
FLAT_SPREAD = 0.50
# How fast the weight of an old snapshot decays: half-life in hours.
HALF_LIFE_HOURS = 24.0
# How strongly a trophy's slope is pulled toward the average slope when it
# has few distinct reads: the average counts as this many extra reads.
SHRINK_READS = 3.0


def parse_time(text):
    """ISO 8601 UTC text to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(pct):
    return min(HI, max(LO, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l > 20.0:
        return HI
    if l < -20.0:
        return LO
    return clamp_rate(100.0 / (1.0 + math.exp(-l)))


def distinct_snapshots(history):
    """Collapse consecutive reads with identical rates to their first.

    The page sits flat between refreshes, so repeated reads carry no new
    information. Only the rates matter here: the players-tracked count can
    move on its own clock, and this program forecasts from the rates.
    """
    snaps = []
    for read in history:
        key = tuple(read.get("rates") or ())
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _, read in snaps]


def weighted_line(xs, ys, ws):
    """Weighted least squares of y on x.

    Returns (intercept, slope, prediction variance at a given x) as a
    function of x: the variance of the fitted value at x, from the weighted
    residuals, so it grows the further x is from the data's centre.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / sw
    ym = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 0:
        return (ym, 0.0, lambda x: 0.0)
    slope = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws)) / sxx
    intercept = ym - slope * xm
    dof = max(1, n - 2)
    resid2 = sum(
        w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)
    )
    # Weighted residual variance; with two points the line is exact, so the
    # caller adds its own allowance for thin data instead.
    sigma2 = resid2 / dof if n > 2 else 0.0

    def pred_var(x):
        base = sigma2 * (1.0 / sw + (x - xm) ** 2 / sxx)
        return max(0.0, base)

    return (intercept, slope, pred_var)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    target_ms = parse_time(case["target"])
    release = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        release_ms = parse_time(release + "T00:00:00Z")
    except ValueError:
        release_ms = parse_time(history[0]["at"]) if history else target_ms

    def hours_since_release(when_ms):
        return max(0.0, (when_ms - release_ms) / 3600.0)

    def x_of(when_ms):
        return math.log1p(hours_since_release(when_ms))

    snaps = distinct_snapshots(history)
    x_target = x_of(target_ms)
    last = snaps[-1] if snaps else None
    last_rates = (last or {}).get("rates") or [None] * len(trophies)

    # Per-trophy trend fits, and the average slope to shrink toward.
    fits = []
    slopes = []
    for idx in range(len(trophies)):
        xs, ys, ws = [], [], []
        for read in snaps:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            xs.append(x_of(parse_time(read["at"])))
            ys.append(logit(rates[idx]))
            age_h = max(0.0, (parse_time(snaps[-1]["at"]) - parse_time(read["at"])) / 3600.0)
            ws.append(0.5 ** (age_h / HALF_LIFE_HOURS))
        fit = weighted_line(xs, ys, ws) if len(xs) >= 2 else None
        fits.append((fit, len(xs)))
        if fit is not None:
            slopes.append(fit[1])
    mean_slope = sum(slopes) / len(slopes) if slopes else 0.0

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy["id"]
        last_rate = last_rates[idx]
        if last_rate is None:
            last_rate = 1.0
        # A trophy everyone already holds stays held: tight, at the top.
        if last_rate >= 99.99:
            rows.append({"id": tid, "q": [99.99, 99.99, 99.99, 99.99, 99.99]})
            continue
        fit, n_reads = fits[idx]
        if fit is None:
            # One flat snapshot: expect the first refresh to dilute.
            med_logit = logit(last_rate) + FLAT_SHIFT
            spread = FLAT_SPREAD
        else:
            intercept, slope, pred_var = fit
            # Shrink a thinly supported slope toward the average slope.
            w = n_reads / (n_reads + SHRINK_READS)
            slope = w * slope + (1.0 - w) * mean_slope
            med_logit = intercept + slope * x_target
            # Fit error, refresh noise, and a thin-data allowance: a slope
            # from two snapshots is exact on them and unproven beyond them.
            thin_sd = 0.45 / math.sqrt(max(1, n_reads))
            var = pred_var(x_target) + REFRESH_SD * REFRESH_SD + thin_sd * thin_sd
            spread = math.sqrt(max(0.0, var))
            spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        qs = []
        prev = 0.0
        for z in ZNORM:
            v = inv_logit(med_logit + z * spread)
            v = round(v, 4)
            if v < prev:
                v = prev
            if v > HI:
                v = HI
            qs.append(v)
            prev = v
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(
        name for name in os.listdir(INPUT_DIR) if name.endswith(".in")
    )
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
