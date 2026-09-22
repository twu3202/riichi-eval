"""EV map of a tested model against a reference, on the tested model's own 1v3 logs.

Every decision of ours where the reference would act differently is rolled out with the reference action and
the reference playing on (same wall). Telescoping (see riichi_eval/rollout/ev_map.py) splits the realized
difference exactly into per-decision contributions; this tool aggregates them by decision type.

    python tools/ev_map.py --log-dir DIR --ours ours.pth --ref mortal_v4.pth [--grp grp.pth]
        [--horizon kyoku|game] [--max-games 400] [--out runs/evmap_x]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.engines.mortal import cap_gpu_mem, load_mortal  # noqa: E402
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.rollout.ev_map import evaluate, find_points  # noqa: E402
from riichi_eval.rollout.value import GrpValuer  # noqa: E402


def complete_groups(dirs):
    groups = defaultdict(dict)
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            m = NAME_RE.search(os.path.basename(p))
            if m:
                groups[(int(m.group(1)), int(m.group(2)))][m.group(3)] = p
    return [[g[k] for k in "abcd"] for key, g in sorted(groups.items()) if len(g) == 4]


def cluster_mean(y_by_group, n_per_group):
    """Mean per unit and 95% half-width, clustering by seed group."""
    y, n = np.asarray(y_by_group, float), np.asarray(n_per_group, float)
    keep = n > 0
    y, n = y[keep], n[keep]
    G = len(y)
    if G < 2 or n.sum() == 0:
        return float("nan"), float("nan")
    r = y.sum() / n.sum()
    se = np.sqrt(G / (G - 1) * ((y - r * n) ** 2).sum()) / n.sum()
    return float(r), float(1.96 * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--grp")
    ap.add_argument("--horizon", default="kyoku", choices=("kyoku", "game"))
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=200, help="games per rollout batch (multiple of 4)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    groups = complete_groups(args.log_dir)[: max(1, args.max_games // 4)]
    paths = [p for g in groups for p in g]
    print(f"{len(groups)} complete seed groups = {len(paths)} games", flush=True)
    cap_gpu_mem(args.device)
    ours = load_mortal(args.ours, args.device, name="ours")
    ref = load_mortal(args.ref, args.device, name=args.ref_name)
    grp = GrpValuer(args.grp) if args.grp else None

    all_rec, g1_all, logged_pt, pc = [], [], [], [0, 0]
    t0 = time.time()
    fh = open(os.path.join(args.out, "points.jsonl"), "w")
    for s in range(0, len(paths), args.chunk):
        chunk = paths[s:s + args.chunk]
        games = [read_game(p) for p in chunk]
        seats = [next(i for i, n in enumerate(g.names) if n != args.ref_name) for g in games]
        points, c = find_points(games, seats, ours, ref)
        pc[0] += c[0]
        pc[1] += c[1]
        recs, g1 = evaluate(games, seats, points, ref, horizon=args.horizon, grp=grp, batch=args.batch)
        for rec in recs:
            gi = rec["game"]
            rec["game"] = s + gi
            rec["group"] = (s + gi) // 4
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_rec.append(rec)
        for gi, g in enumerate(games):
            from riichi_eval.protocol import placement_pt
            logged_pt.append(placement_pt(g.final_scores(), seats[gi]))
            g1_all.append(g1.get(gi))
        print(f"  [{s + len(chunk)}/{len(paths)}] points {len(all_rec):,}  {time.time() - t0:.0f}s", flush=True)
    fh.close()

    n_games = len(paths)
    n_groups = n_games // 4
    print(f"\n{n_games} games; positive control (ours argmax = logged) {pc[0] / max(pc[1], 1):.3%} of {pc[1]:,} decisions")
    ok = sum(r["ok"] for r in all_rec)
    uns = sum(not r["supported"] for r in all_rec)
    print(f"disagreement points {len(all_rec):,} ({len(all_rec) / n_games:.2f}/game); rolled out {ok:,}, "
          f"unsupported {uns:,}, failed {len(all_rec) - ok - uns:,}")
    metrics = (["d_game_pt"] if args.horizon == "game" else []) + (["d_kyoku_grp"] if grp else []) + ["d_kyoku_raw"]
    lp = np.array(logged_pt).reshape(n_groups, 4).mean(1)
    print(f"logged avg_pt of ours on these games {lp.mean():+.3f} ± {1.96 * lp.std(ddof=1) / np.sqrt(n_groups):.3f}")
    if args.horizon == "game":
        g1 = [x for x in g1_all]
        full = [i for i in range(n_groups) if all(g1[4 * i + k] is not None for k in range(4))]
        z = np.array([np.mean([g1[4 * i + k] for k in range(4)]) for i in full])
        print(f"zero check: all-reference games, mean pt over {len(full)} complete groups = {z.mean():+.4f} "
              f"(max |group mean| {np.abs(z).max():.3f}; must be 0 up to GPU fp32 divergence)")
    summary = {"n_games": n_games, "pos_ctrl": pc, "n_points": len(all_rec), "by_subtype": {}}
    subtypes = sorted({r["subtype"] for r in all_rec})
    for m in metrics:
        unit = "pt" if m != "d_kyoku_raw" else "points"
        print(f"\n=== {m} ({unit}; + means our choice was better than the reference's) ===")
        print(f"  {'decision type':<42}{'pts/game':>9}{'per point':>12}{'± CI':>9}{'per game':>11}{'± CI':>9}")
        rows = []
        for st in subtypes + ["ALL"]:
            y = np.zeros(n_groups)
            n = np.zeros(n_groups)
            cnt = 0
            for r in all_rec:
                if (st == "ALL" or r["subtype"] == st) and m in r:
                    y[r["group"]] += r[m]
                    n[r["group"]] += 1
                    cnt += 1
            if cnt == 0:
                continue
            per_pt, ci_pt = cluster_mean(y, n)
            per_game = y.sum() / n_games
            ci_game = 1.96 * (y / 4).std(ddof=1) / np.sqrt(n_groups) if n_groups > 1 else float("nan")
            rows.append((st, cnt / n_games, per_pt, ci_pt, per_game, ci_game))
            summary["by_subtype"].setdefault(st, {})[m] = dict(n=cnt, per_point=per_pt, ci_point=ci_pt,
                                                               per_game=per_game, ci_game=ci_game)
        rows.sort(key=lambda x: (x[0] == "ALL", x[4]))
        for st, rate, pp, cp, pg, cg in rows:
            print(f"  {st:<42}{rate:>9.2f}{pp:>+12.3f}{cp:>9.3f}{pg:>+11.3f}{cg:>9.3f}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}/points.jsonl, summary.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
