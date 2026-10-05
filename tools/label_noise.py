"""Label-noise decomposition for rollout-priced action pairs (A3-style labels): how much of a single true-wall
rollout label's variance is the order of the unseen wall, and how much is the opponents' hidden hands.

Why it matters. Rollouts are deterministic given the full deal (greedy engines on the logged wall), so a single
true-wall rollout already *is* the full-information value of the state. An oracle / privileged value model that
also sees the wall would reproduce that label's noise, not remove it. What can be removed is the spread over
hidden information given what the player sees. The logs carry the true hands, and given all four hands the rest of
the wall is a uniform shuffle, so averaging rollouts over re-dealt walls (libriichi `reshuffle`) removes the wall
part exactly, with no belief model. Removing the hands part would need one.

For each candidate (probe.find_candidates kinds), both the logged action and the alternative are rolled out at the
same branch, on the logged wall (k=0) and on K re-dealt walls (k=1..K; one seed per k, shared by both branches =
common random numbers); our model in our seat, the reference engine in the other three; kyoku horizon, GRP value.
d_k = V(alt, k) − V(ours, k).

    total  = Var_c(d_0)               a single true-wall label (what A3 v1 trained on)
    wall   = E_c[Var_k(d_k)]          removable by averaging over re-dealt walls
    rest   = total − wall             hidden hands + true signal: what a wall-averaged label keeps

Checks printed with the result: mean(d_0) − mean(d_1..K) must be ~0 and Var_c(d_k) must not depend on k (the logged
wall is one draw from the same conditional distribution); every re-dealt rollout must replay the logged prefix.

    python tools/label_noise.py --log-dir DIR --ours ours.pth --ref mortal_v4.pth --grp grp.pth
        --walls 7 --max-games 400 --out runs/noise_x [--resume]
    python tools/label_noise.py --report runs/noise_x

`--only pairs.jsonl` restricts the rollouts to listed pairs (e.g. the ones a tuned model would flip), so a
low-noise price of a model's flips costs rollouts on those pairs only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.engines.mortal import cap_gpu_mem, load_mortal  # noqa: E402
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.protocol import SACRED, placement_pt  # noqa: E402
from riichi_eval.rollout import driver  # noqa: E402
from riichi_eval.rollout.ev_map import _logged_round_end  # noqa: E402
from riichi_eval.rollout.probe import ALL_KINDS, find_candidates  # noqa: E402
from riichi_eval.rollout.spec import kyoku_spec  # noqa: E402
from riichi_eval.rollout.value import GrpValuer  # noqa: E402

LABEL_KINDS = ("push2fold", "fold2push", "riichi2dama", "dama2riichi")


def complete_groups(dirs):
    groups = defaultdict(dict)
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            m = NAME_RE.search(os.path.basename(p))
            if m:
                groups[int(m.group(1))][m.group(3)] = p
    return [(s, [g[k] for k in "abcd"]) for s, g in sorted(groups.items()) if len(g) == 4]


def wall_seed(*key) -> int:
    """Deterministic re-deal seed per (game, kyoku, branch, k); not per branch, so both branches share it."""
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")


def strip(line: str) -> dict:
    ev = json.loads(line)
    ev.pop("meta", None)
    return ev


def run_chunk(games, seats, seeds_rot, ours, ref, grp, kinds, walls, per_game, rng, batch, pts=SACRED.pts, only=None):
    cands, pc = find_candidates(games, seats, ours, ref_engine=ref, kinds=kinds)
    if only is not None:
        cands = [c for c in cands if (*seeds_rot[c.game], c.r, c.pos, c.kind) in only]
    if per_game:
        by_game = defaultdict(list)
        for c in cands:
            by_game[c.game].append(c)
        cands = [c for gi in sorted(by_game) for c in
                 (by_game[gi] if len(by_game[gi]) <= per_game
                  else [by_game[gi][i] for i in sorted(rng.choice(len(by_game[gi]), per_game, replace=False))])]
    specs, meta = [], []
    for ci, c in enumerate(cands):
        g, seat = games[c.game], seats[c.game]
        rd = g.rounds[c.r]
        logged = json.dumps(strip(rd.lines[c.pos]))
        for k in range(walls + 1):
            rs = None if k == 0 else wall_seed(*seeds_rot[c.game], c.r, c.pos, k)
            for b, rep in ((0, logged), (1, json.dumps(c.alt_event))):
                specs.append(kyoku_spec(g, c.r, seat_engines=[0 if s == seat else 1 for s in range(4)],
                                        branch_pos=c.pos, branch_actor=c.actor, replacement=rep,
                                        horizon="kyoku", reshuffle=rs))
                meta.append((ci, k, b))
    res = driver.run([ours, ref], specs, batch=batch) if specs else []

    n = len(cands)
    V = np.full((n, walls + 1, 2), np.nan)
    RAW = np.full((n, walls + 1, 2), np.nan)
    grp_q, grp_seat, grp_idx = [], [], []
    errors, prefix_bad, pos_ok = 0, 0, [0, 0]
    for (ci, k, b), spec, x in zip(meta, specs, res):
        c = cands[ci]
        g, seat = games[c.game], seats[c.game]
        rd = g.rounds[c.r]
        if x.error:
            errors += 1
            continue
        if k > 0 and b == 0:     # a re-dealt wall must still replay the logged prefix exactly
            got = x.kyoku_events(0)[:c.pos]
            if got != [strip(ln) for ln in rd.lines[:c.pos]]:
                prefix_bad += 1
                continue
        if k == 0 and b == 0:    # positive control: the logged action on the logged wall reproduces the log
            nxt = g.rounds[c.r + 1] if c.r + 1 < len(g.rounds) else None
            pos_ok[0] += int(list(x.kyoku_scores) == _logged_round_end(rd, nxt))
            pos_ok[1] += 1
        RAW[ci, k, b] = x.kyoku_scores[seat] - rd.scores[seat]
        if x.next_kyoku is None:
            V[ci, k, b] = placement_pt(x.final_scores, seat, pts)
        else:
            pre = [grp.feature(q.index, q.honba, q.kyotaku, q.scores) for q in g.rounds[:c.r + 1]]
            grp_q.append(pre + [grp.feature(x.next_kyoku[0], x.next_kyoku[1], x.kyoku_kyotaku, x.kyoku_scores)])
            grp_seat.append(seat)
            grp_idx.append((ci, k, b))
    for s in range(0, len(grp_q), 8192):
        for (ci, k, b), v in zip(grp_idx[s:s + 8192], grp.expected_pt(grp_q[s:s + 8192], grp_seat[s:s + 8192])):
            V[ci, k, b] = v
    rows = []
    for ci, c in enumerate(cands):
        if np.isnan(V[ci]).any():
            continue
        rows.append(dict(seed=seeds_rot[c.game][0], rot=seeds_rot[c.game][1], r=c.r, pos=c.pos, kind=c.kind,
                         margin=c.margin, turn=c.turn, exposed=c.exposed, ref_flip=c.ref_flip,
                         v=np.round(V[ci], 4).tolist(), raw=RAW[ci].astype(int).tolist()))
    return rows, dict(cands=n, kept=len(rows), errors=errors, prefix_bad=prefix_bad, pos_ok=pos_ok, pc=pc)


def decompose(rows, n_boot=300, seed=0):
    """Variance split of the per-pair label (see module doc), with a seed-cluster bootstrap on the wall share."""
    V = np.array([r["v"] for r in rows], np.float64)            # (n, K+1, 2): [ours, alt]
    D = V[:, :, 1] - V[:, :, 0]
    K1 = D.shape[1]
    clusters = np.array([r["seed"] for r in rows])

    def stats(Dx, Vx):
        total = Dx[:, 0].var(ddof=1)
        wall = Dx.var(axis=1, ddof=1).mean()
        indep = (Vx[:, :, 0].var(axis=1, ddof=1) + Vx[:, :, 1].var(axis=1, ddof=1)).mean()
        return total, wall, indep

    total, wall, indep = stats(D, V)
    useq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    pos = {s: np.flatnonzero(clusters == s) for s in useq}
    shares = []
    for _ in range(n_boot):
        idx = np.concatenate([pos[s] for s in rng.choice(useq, len(useq))])
        t, w, _ = stats(D[idx], V[idx])
        shares.append(w / t)
    lo, hi = np.percentile(shares, [2.5, 97.5])
    d0, dk = D[:, 0], D[:, 1:].mean(axis=1)
    diff = d0 - dk
    dc = diff - diff.mean()
    se = np.sqrt(sum(dc[pos[s]].sum() ** 2 for s in useq)) / len(diff)     # seed-cluster robust SE
    return dict(n=len(rows), clusters=len(useq), walls=K1 - 1, total=total, wall=wall, rest=total - wall,
                share=wall / total, share_ci=(lo, hi), crn_gain=indep / wall,
                var_by_k=[float(D[:, k].var(ddof=1)) for k in range(K1)],
                mean_d0=float(d0.mean()), mean_dk=float(dk.mean()), bias=float(diff.mean()), bias_se=float(se),
                sd_single=float(np.sqrt(total)), sd_avg=float(np.sqrt(total - wall + wall / K1)))


def report(rows):
    groups = [("all", rows)] + [(k, [r for r in rows if r["kind"] == k]) for k in LABEL_KINDS]
    print(f"{'kind':<12}{'pairs':>7}{'SD single':>11}{'wall share':>12}{'95% CI':>16}"
          f"{'SD avg':>8}{'CRN x':>7}   check: mean d0 − d1..K, Var_c(d_k) over k")
    for name, rk in groups:
        if len(rk) < 30:
            continue
        s = decompose(rk)
        vk = s["var_by_k"]
        lo, hi = s["share_ci"]
        print(f"{name:<12}{s['n']:>7,}{s['sd_single']:>11.2f}{s['share']:>12.1%}"
              f"   [{lo:.1%}, {hi:.1%}]{s['sd_avg']:>8.2f}{s['crn_gain']:>7.1f}"
              f"   {s['bias']:+.3f} ± {1.96 * s['bias_se']:.3f}   {min(vk):.0f}..{max(vk):.0f}")
    s = decompose(rows)
    print(f"\nK = {s['walls']} re-dealt walls per pair (+ the logged wall), {s['clusters']:,} seed groups.")
    print("SD single = one logged-wall label (A3 v1); SD avg = label averaged over the K+1 walls, i.e.")
    print("sqrt(rest + wall/(K+1)); wall share = the part of a single label's variance that wall-averaging removes;")
    print("CRN x = how much larger the wall variance would be without sharing the wall between the two branches.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append")
    ap.add_argument("--ours")
    ap.add_argument("--ref")
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--grp")
    ap.add_argument("--walls", type=int, default=7, help="re-dealt walls per pair, besides the logged one")
    ap.add_argument("--kinds", default=",".join(LABEL_KINDS))
    ap.add_argument("--per-game", type=int, default=0, help="subsample at most this many pairs per game (0 = all)")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40, help="games per chunk")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", metavar="OUT_DIR")
    ap.add_argument("--only", help="jsonl of {seed, rot, r, pos, kind}: roll out only these pairs (and only their seed groups)")
    args = ap.parse_args()

    if args.report:
        report([json.loads(ln) for ln in open(os.path.join(args.report, "pairs.jsonl"))])
        return
    kinds = tuple(k for k in args.kinds.split(",") if k)
    if not set(kinds) <= set(ALL_KINDS):
        sys.exit(f"unknown kinds: {set(kinds) - set(ALL_KINDS)}")
    os.makedirs(args.out, exist_ok=True)
    groups = complete_groups(args.log_dir)
    only = None
    if args.only:
        only = {(x["seed"], x["rot"], x["r"], x["pos"], x["kind"]) for x in map(json.loads, open(args.only))}
        want = {k[0] for k in only}
        groups = [(s, ps) for s, ps in groups if s in want]
    if args.max_games:
        groups = groups[: args.max_games // 4]
    path = os.path.join(args.out, "pairs.jsonl")
    done = set()
    if os.path.exists(path):
        if not args.resume:
            sys.exit(f"{path} exists; use --resume or a new --out")
        done = {json.loads(ln)["seed"] for ln in open(path)}
    todo = [(s, ps) for s, ps in groups if s not in done]
    print(f"{len(groups):,} seed groups, {len(done):,} done, {len(todo):,} to go; K={args.walls}", flush=True)
    cap_gpu_mem(args.device)
    ours = load_mortal(args.ours, args.device, name="ours")
    ref = load_mortal(args.ref, args.device, name=args.ref_name)
    grp = GrpValuer(args.grp)
    per, t0 = max(1, args.chunk // 4), time.time()
    tot = defaultdict(int)
    pos = [0, 0]
    for c in range(0, len(todo), per):
        chunk = todo[c:c + per]
        paths = [p for _, ps in chunk for p in ps]
        games = [read_game(p) for p in paths]
        seats = [next(i for i, n in enumerate(g.names) if n != args.ref_name) for g in games]
        seeds_rot = [(s, rot) for s, _ in chunk for rot in "abcd"]
        rng = np.random.default_rng(chunk[0][0])
        rows, st = run_chunk(games, seats, seeds_rot, ours, ref, grp, kinds, args.walls, args.per_game, rng,
                             args.batch, only=only)
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        for k in ("cands", "kept", "errors", "prefix_bad"):
            tot[k] += st[k]
        pos[0] += st["pos_ok"][0]
        pos[1] += st["pos_ok"][1]
        el = time.time() - t0
        print(f"  [{4 * (c + len(chunk)):,}/{4 * len(todo):,}] pairs {tot['kept']:,}/{tot['cands']:,}, "
              f"errors {tot['errors']}, prefix mismatches {tot['prefix_bad']}, "
              f"logged-wall reproduces log {pos[0] / max(pos[1], 1):.2%}  {el:.0f}s, "
              f"eta {el / (c + len(chunk)) * (len(todo) - c - len(chunk)) / 60:.0f} min", flush=True)
    print()
    report([json.loads(ln) for ln in open(path)])
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
