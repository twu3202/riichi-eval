"""Game-horizon vs kyoku-horizon (GRP) valuation of the same EV-map rollouts.

An EV map run with --horizon game --grp gives every disagreement two values from the same rollouts:
d_game_pt (true final placement, reference playing on to the end) and d_kyoku_grp (the kyoku end valued by the
GRP, which is what the kyoku-horizon map and Mortal-style training use). Their per-point difference is the error
of truncating at the kyoku end with the GRP. Reported per decision type: both means and the paired difference,
per point and per game, seed-clustered 95% CIs (complete seed groups only).

    python tools/ev_horizon_cmp.py runs/evmap_x_game [more shard dirs]
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.rollout import ev_summary  # noqa: E402


def cluster(y, n):
    keep = n > 0
    y, n = y[keep], n[keep]
    G = len(y)
    if G < 2 or n.sum() == 0:
        return float("nan"), float("nan")
    r = y.sum() / n.sum()
    return r, 1.96 * np.sqrt(G / (G - 1) * ((y - r * n) ** 2).sum()) / n.sum()


def main():
    points, games = ev_summary.load(sys.argv[1:])
    by_seed = defaultdict(set)
    for seed, rot in games:
        by_seed[seed].add(rot)
    seeds = sorted(s for s, r in by_seed.items() if len(r) == 4)
    sidx = {s: i for i, s in enumerate(seeds)}
    G, n_games = len(seeds), 4 * len(seeds)
    pts = [p for p in points if p["seed"] in sidx and "d_game_pt" in p and "d_kyoku_grp" in p]
    print(f"{n_games:,} games ({G:,} complete seed groups), {len(pts):,} points with both valuations\n")
    print(f"{'decision type':<40}{'/game':>6} | {'game-horizon':>16} | {'kyoku+GRP':>16} | "
          f"{'game - GRP /point':>18} {'z':>6} | {'game - GRP /game':>17}")
    rows = []
    for st in sorted({p["subtype"] for p in pts}) + ["ALL"]:
        y = {m: np.zeros(G) for m in ("g", "k", "d")}
        n = np.zeros(G)
        for p in pts:
            if st == "ALL" or p["subtype"] == st:
                i = sidx[p["seed"]]
                y["g"][i] += p["d_game_pt"]
                y["k"][i] += p["d_kyoku_grp"]
                y["d"][i] += p["d_game_pt"] - p["d_kyoku_grp"]
                n[i] += 1
        if n.sum() < 200 and st != "ALL":
            continue
        g, gc = cluster(y["g"], n)
        k, kc = cluster(y["k"], n)
        d, dc = cluster(y["d"], n)
        pg = y["d"].sum() / n_games
        pgc = 1.96 * (y["d"] / 4).std(ddof=1) / np.sqrt(G)
        rows.append((st, n.sum() / n_games, g, gc, k, kc, d, dc, pg, pgc))
    rows.sort(key=lambda r: (r[0] == "ALL", -abs(r[8])))
    for st, rate, g, gc, k, kc, d, dc, pg, pgc in rows:
        z = d / (dc / 1.96) if dc > 0 else float("nan")
        print(f"{st:<40}{rate:>6.2f} | {g:>+7.3f} ± {gc:<6.3f} | {k:>+7.3f} ± {kc:<6.3f} | "
              f"{d:>+8.3f} ± {dc:<7.3f} {z:>+6.2f} | {pg:>+7.3f} ± {pgc:<7.3f}")


if __name__ == "__main__":
    main()
