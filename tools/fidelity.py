"""Rollout fidelity controls. Must pass before any rollout number is trusted.

A. Forced replay: every kyoku of every game replayed entirely from the log (no engine decides) must reproduce
   the logged kyoku event by event, and its end scores must equal the next kyoku's start scores.
   Tests wall seeding, the init state and the forced-reaction rule. Expect 100%.
B. Free continuation: from a random position of a random kyoku, every seat plays its engine (original seating)
   to the end of the game. Must reproduce the rest of the logged game. Expect ~100% at CPU fp32; at GPU fp32
   batch-dependent kernels can flip a near-tie, so a small divergence rate is reported, not hidden.

    python tools/fidelity.py --log-dir DIR --challenger ck.pth --champion mortal_v4.pth [--n-games 200]
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.engines.mortal import cap_gpu_mem, load_mortal  # noqa: E402
from riichi_eval.logs import NAME_RE, read_game, strip_meta  # noqa: E402
from riichi_eval.rollout import driver  # noqa: E402
from riichi_eval.rollout.spec import kyoku_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--champion-name", default="mortal-v4")
    ap.add_argument("--n-games", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in glob.glob(os.path.join(args.log_dir, "*.json.gz")) if NAME_RE.search(os.path.basename(p)))
    games = [read_game(p) for p in files[:args.n_games]]
    print(f"{len(games)} games, {sum(len(g.rounds) for g in games)} kyoku, from {args.log_dir}")
    cap_gpu_mem(args.device)
    engines = [load_mortal(args.challenger, args.device, name="challenger"),
               load_mortal(args.champion, args.device, name=args.champion_name)]

    def seating(g):
        return [1 if n == args.champion_name else 0 for n in g.names]

    if not args.skip_a:
        specs, refs = [], []
        for g in games:
            for r, rd in enumerate(g.rounds):
                specs.append(kyoku_spec(g, r, seat_engines=seating(g)))
                nxt = g.rounds[r + 1].scores if r + 1 < len(g.rounds) else None
                refs.append(([strip_meta(ln) for ln in rd.lines], nxt))
        t0 = time.time()
        res = driver.run(engines, specs, batch=args.batch)
        ok_ev = ok_sc = n_sc = err = 0
        first_bad = None
        for (ref, nxt), x in zip(refs, res):
            if x.error:
                err += 1
                first_bad = first_bad or x.error[:400]
                continue
            got = x.kyoku_events(0)
            if got == ref:
                ok_ev += 1
            elif first_bad is None:
                k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
                first_bad = f"kyoku mismatch at event {k}: got {got[k:k+2]} ref {ref[k:k+2]}"
            if nxt is not None:
                n_sc += 1
                ok_sc += x.kyoku_scores == nxt
        n = len(specs)
        print(f"A forced replay: {ok_ev}/{n} kyoku identical ({ok_ev / n:.4%}), end scores {ok_sc}/{n_sc}, "
              f"errors {err}  ({time.time() - t0:.1f}s)")
        if first_bad:
            print(f"  first problem: {first_bad}")

    rng = random.Random(args.seed)
    specs, refs = [], []
    for g in games:
        r = rng.randrange(len(g.rounds))
        pos = rng.randrange(1, len(g.rounds[r].lines) - 1)
        specs.append(kyoku_spec(g, r, seat_engines=seating(g), branch_pos=pos, horizon="game"))
        rest = [strip_meta(ln) for rd in g.rounds[r:] for ln in rd.lines]
        refs.append((rest, g.final_scores()))
    t0 = time.time()
    res = driver.run(engines, specs, batch=args.batch)
    ok_ev = ok_fin = err = 0
    first_bad = None
    for (rest, fin), x in zip(refs, res):
        if x.error:
            err += 1
            first_bad = first_bad or x.error[:400]
            continue
        got = [e for e in (strip_meta(ln) for ln in x.log.splitlines()) if e["type"] not in ("start_game", "end_game")]
        ok_ev += got == rest
        ok_fin += x.final_scores == fin
        if got != rest and first_bad is None:
            k = next((i for i, (a, b) in enumerate(zip(got, rest)) if a != b), min(len(got), len(rest)))
            first_bad = f"diverged at event {k} of the remainder: got {got[k:k+1]} ref {rest[k:k+1]}"
    n = len(specs)
    print(f"B free continuation to game end: {ok_ev}/{n} identical remainder ({ok_ev / n:.2%}), "
          f"final scores {ok_fin}/{n} ({ok_fin / n:.2%}), errors {err}  ({time.time() - t0:.1f}s)")
    if first_bad:
        print(f"  first problem: {first_bad}")


if __name__ == "__main__":
    main()
