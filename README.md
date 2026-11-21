# GTA VI Trophy Rarity Forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, the percent
of the players Exophase tracks who will have earned it at 2026-11-26 00:00
UTC, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## How it works

A trophy's earn rate is earners over players tracked, and the two counts
grow on different clocks. Players tracked jumps whenever Exophase refreshes
the page, which dilutes every rate at once; earners grow as the players
already tracked work through the game, which lifts the rates of the
trophies they reach. So a rate's path is a tug of war, and which side wins
depends on the trophy: an early story trophy is earned by most players
within days, so its rate recovers and climbs, while a late or grindy trophy
is earned by almost nobody in week one, so dilution wins and its rate keeps
sliding.

`main.py` reads the case's history of reads, collapses it to the distinct
snapshots (the page sits flat between refreshes), and fits, per trophy, a
line of the logit's rate against log time since release. The slope is that
trophy's tug of war, shrunk toward the average slope across trophies when
the trophy has few distinct reads to speak for itself. The line is read at
the target, and the spread around it is the fit's own prediction error plus
an allowance for the next refresh landing where it lands. When the history
is a single flat snapshot (launch night, before any refresh), no slope can
be fit, so the program falls back to the one thing launch night teaches:
the first refresh dilutes the average rate to about a fifth of what it was
(the house's own counts say the first refresh took players tracked from 638
to 3,868 while trophies earned went from 943 to 4,276, and the trophy
everyone holds accounts for 3,230 of the 3,333 new earners), so the median
sits well below the last rate and the spread is wide.

The forecast is deterministic: the same case always writes the same bytes.
Standard library only, no network.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per
case, `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## Validation (local)

- Practice cases (flat launch-night history): valid output, medians below
  the last rate, as the refresh demands. On a synthetic first refresh
  rebuilt from the house's own counts (3,868 players, 4,276 trophies
  earned), pinball loss 0.114 against the house baseline's 1.41 (the rules
  put the house practice loss at about 1.51).
- Simulated 4.5-day final (log-time trends with refresh steps, 14 distinct
  snapshots): pinball loss 0.128 against the house baseline's 0.267;
  with milder trends, 0.092 against 0.240.
- Runtime about 0.07 s per case, far inside the 8 s cap.

## License

MIT. See LICENSE.
