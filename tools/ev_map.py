"""EV map of a tested model against a reference, on the tested model's own 1v3 logs.

Every decision of ours where the reference would act differently is rolled out with the reference action and
the reference playing on (same wall). Telescoping (riichi_eval/rollout/ev_map.py) splits the realized difference
exactly into per-decision contributions; results are aggregated by decision type, clustered by seed.

Output dir gets points.jsonl, games.jsonl (appended chunk by chunk; --resume skips finished seeds) and
summary.json. Shards from several machines are combined with tools/ev_merge.py.

    python tools/ev_map.py --log-dir DIR [--log-dir ...] --ours ours.pth --ref mortal_v4.pth [--grp grp.pth]
        [--horizon kyoku|game] [--max-games N] --out runs/evmap_x [--resume]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.engines.mortal import cap_gpu_mem, load_mortal  # noqa: E402
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.protocol import placement_pt  # noqa: E402
from riichi_eval.rollout import ev_summary  # noqa: E402
from riichi_eval.rollout.ev_map import evaluate, find_points  # noqa: E402
from riichi_eval.rollout.value import GrpValuer  # noqa: E402


def complete_groups(dirs):
    groups = defaultdict(dict)
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            m = NAME_RE.search(os.path.basename(p))
            if m:
                groups[int(m.group(1))][m.group(3)] = p
    return [(seed, [g[k] for k in "abcd"]) for seed, g in sorted(groups.items()) if len(g) == 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--grp")
    ap.add_argument("--horizon", default="kyoku", choices=("kyoku", "game"))
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200, help="games per rollout batch (multiple of 4)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    groups = complete_groups(args.log_dir)
    if args.max_games:
        groups = groups[: args.max_games // 4]
    gpath = os.path.join(args.out, "games.jsonl")
    done = set()
    if args.resume and os.path.exists(gpath):
        with open(gpath) as f:
            done = {json.loads(ln)["seed"] for ln in f}
    elif os.path.exists(gpath):
        sys.exit(f"{gpath} exists; use --resume or a new --out")
    todo = [(s, ps) for s, ps in groups if s not in done]
    print(f"{len(groups):,} complete seed groups, {len(done):,} already done, {len(todo):,} to go", flush=True)
    cap_gpu_mem(args.device)
    ours = load_mortal(args.ours, args.device, name="ours")
    ref = load_mortal(args.ref, args.device, name=args.ref_name)
    grp = GrpValuer(args.grp) if args.grp else None
    per = max(1, args.chunk // 4)
    t0 = time.time()
    n_pts = 0
    pc = [0, 0]
    for c in range(0, len(todo), per):
        chunk = todo[c:c + per]
        paths = [p for _, ps in chunk for p in ps]
        games = [read_game(p) for p in paths]
        seats = [next(i for i, n in enumerate(g.names) if n != args.ref_name) for g in games]
        points, pcc = find_points(games, seats, ours, ref)
        pc[0] += pcc[0]
        pc[1] += pcc[1]
        recs, g1 = evaluate(games, seats, points, ref, horizon=args.horizon, grp=grp, batch=args.batch)
        with open(os.path.join(args.out, "points.jsonl"), "a") as fp:
            for rec in recs:
                gi = rec["game"]
                rec["seed"], rec["rot"] = chunk[gi // 4][0], "abcd"[gi % 4]
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(gpath, "a") as fg:
            for gi, g in enumerate(games):
                npts = sum(1 for r in recs if r["game"] == gi)
                fg.write(json.dumps({"seed": chunk[gi // 4][0], "rot": "abcd"[gi % 4], "path": paths[gi],
                                     "logged_pt": placement_pt(g.final_scores(), seats[gi]), "g1": g1.get(gi),
                                     "n_points": npts}) + "\n")
        n_pts += len(recs)
        el = time.time() - t0
        eta = el / (c + len(chunk)) * (len(todo) - c - len(chunk))
        print(f"  [{4 * (c + len(chunk)):,}/{4 * len(todo):,} games] points {n_pts:,}  pos-ctrl "
              f"{pc[0] / max(pc[1], 1):.3%}  {el:.0f}s, eta {eta / 60:.0f} min", flush=True)
    print()
    points, games = ev_summary.load([args.out])
    res = ev_summary.summarize(points, games)
    res["pos_ctrl_this_run"] = pc
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
