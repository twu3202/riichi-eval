"""Positive control for `commit` rollouts: after the branch the branch actor must keep to the macro-action.

  fold: at every later decision of that kyoku, a legal genbutsu is discarded whenever one exists (agari is
        allowed), offered calls are declined, and there is no riichi or kan;
  dama: no riichi.

Checks the live rollout logs against the Python genbutsu rule (riichi_eval.replay), which is written
independently of the Rust one.

    python tools/commit_check.py --log-dir DIR --ours ck.pth --ref mortal_v4.pth [--n-games 100]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.engines.mortal import cap_gpu_mem, load_mortal  # noqa: E402
from riichi_eval.logs import parse_game, read_game, strip_meta  # noqa: E402
from riichi_eval.replay import iter_decisions, kind_of_action  # noqa: E402
from riichi_eval.rollout import driver  # noqa: E402
from riichi_eval.rollout.probe import find_candidates  # noqa: E402
from riichi_eval.rollout.spec import kyoku_spec  # noqa: E402


def violation(kind: str, d) -> str | None:
    a = d.logged
    if kind == "riichi2dama":
        return "riichi" if a == 37 else None
    if a == 43:
        return None
    if d.bucket == "call" or (d.mask[45] and not d.mask[:37].any()):
        return None if a == 45 else f"call {a}"
    safe = [x for x in range(37) if d.mask[x] and d.safe is not None and kind_of_action(x) in d.safe]
    if safe:
        return None if a in safe else f"non-genbutsu {a} with genbutsu {safe}"
    return None if 0 <= a <= 36 else f"action {a} with no genbutsu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--n-games", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cap_gpu_mem(args.device)
    ours = load_mortal(args.ours, args.device, name="ours")
    ref = load_mortal(args.ref, args.device, name=args.ref_name)
    paths = sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz")))[: args.n_games]
    games = [read_game(p) for p in paths]
    seats = [next(i for i, n in enumerate(g.names) if n != args.ref_name) for g in games]
    cands, _ = find_candidates(games, seats, ours)
    specs = [kyoku_spec(games[c.game], c.r, seat_engines=[0 if s == seats[c.game] else 1 for s in range(4)],
                        branch_pos=c.pos, branch_actor=c.actor, replacement=json.dumps(c.alt_event),
                        commit="fold" if c.kind == "push2fold" else "dama") for c in cands]
    res = driver.run([ours, ref], specs)
    checked, bad, errors = Counter(), Counter(), 0
    shown = 0
    for c, x in zip(cands, res):
        if x.error:
            errors += 1
            continue
        g = parse_game([json.dumps(strip_meta(ln)) for ln in x.log.splitlines()])
        for d in iter_decisions(g, [c.actor]):
            if d.r != 0 or d.pos <= c.pos or d.bucket == "forced":
                continue
            checked[c.kind] += 1
            v = violation(c.kind, d)
            if v:
                bad[c.kind] += 1
                if shown < 5:
                    shown += 1
                    print(f"  VIOLATION {c.kind} {os.path.basename(games[c.game].path)} r{c.r} pos {d.pos}: {v}")
    print(f"{len(cands)} candidates, {errors} rollout errors")
    for k in ("push2fold", "riichi2dama"):
        print(f"  {k}: {checked[k]} later decisions checked, {bad[k]} violations")
    sys.exit(1 if errors or sum(bad.values()) else 0)


if __name__ == "__main__":
    main()
