"""Aggregating EV-map records (points.jsonl + games.jsonl, possibly from several shards)."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np


def load(dirs):
    points, games = [], {}
    for d in dirs:
        with open(os.path.join(d, "games.jsonl")) as f:
            for ln in f:
                g = json.loads(ln)
                games[(g["seed"], g["rot"])] = g
        with open(os.path.join(d, "points.jsonl")) as f:
            points.extend(json.loads(ln) for ln in f)
    return points, games


def _cluster(y, n):
    y, n = np.asarray(y, float), np.asarray(n, float)
    keep = n > 0
    y, n = y[keep], n[keep]
    G = len(y)
    if G < 2 or n.sum() == 0:
        return float("nan"), float("nan")
    r = y.sum() / n.sum()
    return float(r), float(1.96 * np.sqrt(G / (G - 1) * ((y - r * n) ** 2).sum()) / n.sum())


def summarize(points, games, *, out=print):
    """Only complete seed groups (all four rotations present) are used; everything clusters by seed."""
    by_seed = defaultdict(dict)
    for (seed, rot), g in games.items():
        by_seed[seed][rot] = g
    seeds = sorted(s for s, d in by_seed.items() if len(d) == 4)
    sidx = {s: i for i, s in enumerate(seeds)}
    G, n_games = len(seeds), 4 * len(seeds)
    pts = [p for p in points if p["seed"] in sidx]
    res = {"n_games": n_games, "n_points": len(pts), "by_subtype": {}}
    if G < 2:
        out("fewer than two complete seed groups")
        return res
    lp = np.array([np.mean([by_seed[s][r]["logged_pt"] for r in "abcd"]) for s in seeds])
    out(f"{n_games:,} games ({G:,} complete seed groups); logged avg_pt of ours {lp.mean():+.3f} ± "
        f"{1.96 * lp.std(ddof=1) / np.sqrt(G):.3f}")
    ok = sum(p["ok"] for p in pts)
    uns = sum(not p["supported"] for p in pts)
    out(f"disagreement points {len(pts):,} ({len(pts) / n_games:.2f}/game); rolled out {ok:,}, unsupported {uns:,}, "
        f"failed {len(pts) - ok - uns:,}")
    g1 = [[by_seed[s][r].get("g1") for r in "abcd"] for s in seeds]
    full = [x for x in g1 if all(v is not None for v in x)]
    if full:
        sums = np.array([sum(x) for x in full])
        out(f"zero check (all-reference games sum to 0 per seed): exact in {int((np.abs(sums) < 1e-9).sum()):,}/"
            f"{len(full):,} groups; the rest diverged numerically (GPU fp32), mean {sums.mean() / 4:+.4f} pt/game")
    metrics = [m for m in ("d_game_pt", "d_kyoku_grp", "d_kyoku_raw") if any(m in p for p in pts)]
    subtypes = sorted({p["subtype"] for p in pts})
    for m in metrics:
        unit = "points" if m == "d_kyoku_raw" else "pt"
        out(f"\n=== {m} ({unit}; + = our choice better than the reference's) ===")
        out(f"  {'decision type':<42}{'pts/game':>9}{'per point':>12}{'± CI':>9}{'per game':>11}{'± CI':>9}")
        rows = []
        for st in subtypes + ["ALL"]:
            y, n = np.zeros(G), np.zeros(G)
            for p in pts:
                if (st == "ALL" or p["subtype"] == st) and m in p:
                    i = sidx[p["seed"]]
                    y[i] += p[m]
                    n[i] += 1
            if n.sum() == 0:
                continue
            pp, cp = _cluster(y, n)
            pg, cg = y.sum() / n_games, 1.96 * (y / 4).std(ddof=1) / np.sqrt(G)
            rows.append((st, n.sum() / n_games, pp, cp, pg, cg))
            res["by_subtype"].setdefault(st, {})[m] = dict(n=int(n.sum()), per_point=pp, ci_point=cp,
                                                           per_game=float(pg), ci_game=float(cg))
        rows.sort(key=lambda x: (x[0] == "ALL", x[4]))
        for st, rate, pp, cp, pg, cg in rows:
            out(f"  {st:<42}{rate:>9.2f}{pp:>+12.3f}{cp:>9.3f}{pg:>+11.3f}{cg:>9.3f}")
    return res
