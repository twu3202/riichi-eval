"""Calibration probe (A1): price flipping our own pushes / riichi declarations, and see whether our own Q
margin can gate the flip.

    python tools/policy_probe.py --log-dir DIR --ours ours.pth --ref mortal_v4.pth --grp grp.pth
        --max-games 20000 --out runs/probe_x [--resume]

Writes candidates.jsonl (one row per priced flip) and prints, for a grid of thresholds tau, the first-order
gain of the rule "flip when our Q margin < tau", with an honest split-half estimate (tau chosen on half the
seeds, scored on the other half).
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
from riichi_eval.rollout.probe import find_candidates, price  # noqa: E402
from riichi_eval.rollout.value import GrpValuer  # noqa: E402

TAUS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]


def complete_groups(dirs):
    groups = defaultdict(dict)
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            m = NAME_RE.search(os.path.basename(p))
            if m:
                groups[int(m.group(1))][m.group(3)] = p
    return [(s, [g[k] for k in "abcd"]) for s, g in sorted(groups.items()) if len(g) == 4]


def gain_curve(rows, metric, seeds_keep=None):
    """Per-game gain of 'flip when margin < tau' for each tau, plus the count of flips per game."""
    out = []
    seeds = sorted({r["seed"] for r in rows} if seeds_keep is None else seeds_keep)
    idx = {s: i for i, s in enumerate(seeds)}
    n_games = 4 * len(seeds)
    for tau in TAUS:
        y = np.zeros(len(seeds))
        n = 0
        for r in rows:
            if r["seed"] in idx and r["margin"] < tau and r.get(metric) is not None:
                y[idx[r["seed"]]] += r[metric]
                n += 1
        g = y.sum() / n_games
        ci = 1.96 * (y / 4).std(ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else float("nan")
        out.append((tau, n / n_games, g, ci))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--grp")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--commit", action="store_true",
                    help="price the alternative as a macro-action: keep folding / keep dama for the rest of the kyoku")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    groups = complete_groups(args.log_dir)
    if args.max_games:
        groups = groups[: args.max_games // 4]
    cpath = os.path.join(args.out, "candidates.jsonl")
    done = set()
    if os.path.exists(cpath):
        if not args.resume:
            sys.exit(f"{cpath} exists; use --resume or a new --out")
        with open(cpath) as f:
            done = {json.loads(ln)["seed"] for ln in f}
    todo = [(s, ps) for s, ps in groups if s not in done]
    print(f"{len(groups):,} seed groups, {len(done):,} done, {len(todo):,} to go", flush=True)
    cap_gpu_mem(args.device)
    ours = load_mortal(args.ours, args.device, name="ours")
    ref = load_mortal(args.ref, args.device, name=args.ref_name)
    grp = GrpValuer(args.grp) if args.grp else None
    per, t0, pc = max(1, args.chunk // 4), time.time(), [0, 0]
    for c in range(0, len(todo), per):
        chunk = todo[c:c + per]
        paths = [p for _, ps in chunk for p in ps]
        games = [read_game(p) for p in paths]
        seats = [next(i for i, n in enumerate(g.names) if n != args.ref_name) for g in games]
        cands, pcc = find_candidates(games, seats, ours, ref_engine=ref)
        pc[0] += pcc[0]
        pc[1] += pcc[1]
        priced = price(games, seats, cands, ours, ref, grp=grp, batch=args.batch, commit=args.commit)
        with open(cpath, "a") as f:
            for cd in priced:
                row = {k: getattr(cd, k) for k in ("r", "pos", "actor", "kind", "ours", "alt", "margin",
                                                    "exposed", "shanten", "opened", "turn", "n_riichi", "dora",
                                                    "ref_flip", "ref_margin")}
                row["seed"], row["rot"] = chunk[cd.game // 4][0], "abcd"[cd.game % 4]
                if cd.alt_ev:
                    row.update({"d_grp": cd.alt_ev.get("grp"), "d_raw": cd.alt_ev.get("raw")})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        el = time.time() - t0
        print(f"  [{4 * (c + len(chunk)):,}/{4 * len(todo):,}] pos-ctrl {pc[0] / max(pc[1], 1):.3%}  {el:.0f}s, "
              f"eta {el / (c + len(chunk)) * (len(todo) - c - len(chunk)) / 60:.0f} min", flush=True)

    rows = [json.loads(ln) for ln in open(cpath)]
    seeds = sorted({r["seed"] for r in rows})
    n_games = 4 * len(seeds)
    print(f"\n{n_games:,} games, {len(rows):,} priced flips "
          f"({sum(1 for r in rows if r['kind'] == 'push2fold') / n_games:.2f} push→fold, "
          f"{sum(1 for r in rows if r['kind'] == 'riichi2dama') / n_games:.2f} riichi→dama per game)")
    for kind in ("push2fold", "riichi2dama", "both"):
        rk = rows if kind == "both" else [r for r in rows if r["kind"] == kind]
        if not rk:
            continue
        print(f"\n=== {kind}: gain of 'flip when our Q margin < tau' (pt/game, GRP kyoku value) ===")
        print(f"  {'tau':>8}{'flips/game':>12}{'gain':>10}{'± CI':>9}")
        for tau, rate, g, ci in gain_curve(rk, "d_grp"):
            print(f"  {tau:>8.2f}{rate:>12.2f}{g:>+10.3f}{ci:>9.3f}")
        half = set(seeds[::2])
        fit = gain_curve(rk, "d_grp", seeds_keep=[s for s in seeds if s in half])
        best = max(fit, key=lambda x: x[2])
        test = gain_curve(rk, "d_grp", seeds_keep=[s for s in seeds if s not in half])
        t_at = [x for x in test if x[0] == best[0]][0]
        print(f"  split-half: tau*={best[0]:g} fitted on half the seeds -> other half gain "
              f"{t_at[2]:+.3f} ± {t_at[3]:.3f} pt/game ({t_at[1]:.2f} flips/game)")
        if any(r.get("ref_flip") is not None for r in rk):
            # Diagnostic only: where would the reference engine also have folded / stayed dama?
            print(f"  by what {args.ref_name} would do at the same state (all margins):")
            for lab, sel in (("ref would also flip", True), ("ref would do as we did", False)):
                sub = [r for r in rk if r.get("ref_flip") is sel]
                tau, rate, g, ci = gain_curve(sub, "d_grp", seeds_keep=seeds)[-1]
                print(f"    {lab:<24}{rate:>8.2f} flips/game  gain {g:+.3f} ± {ci:.3f} pt/game  "
                      f"({g / rate if rate else float('nan'):+.3f} pt/flip)")
            sub = [r for r in rk if r.get("ref_flip")]
            print(f"    ...and within 'ref would also flip', gated by our own margin:")
            for tau, rate, g, ci in gain_curve(sub, "d_grp", seeds_keep=seeds):
                if tau in (0.1, 0.3, 0.5, 1.0, 2.0):
                    print(f"      tau {tau:>4.1f}: {rate:.2f} flips/game, gain {g:+.3f} ± {ci:.3f}")
    print(f"\n-> {cpath}")


if __name__ == "__main__":
    main()
