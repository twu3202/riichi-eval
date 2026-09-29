"""GRP calibration audit on our own 1v3 logs: does the GRP's expected placement at a kyoku start match what
actually happens at our tables?

Mortal-style training (and the kyoku-horizon EV map) values the end of every kyoku by the GRP: the expected
placement pt from the next kyoku start, as predicted for Tenhou Houou humans playing on. Nothing after that kyoku
is looked at. If the GRP is off by a constant, no decision changes; if it is off *differently* for states one
decision can move between (after a deal-in vs after folding, say), the target is biased there. This tool pairs,
for every kyoku start of every seat, the GRP-expected pt with the realized final placement pt, and reports the
residual (realized - GRP) by game stage x own score and by rank, separately for the tested model's seat and the
reference seats. Residuals are clustered by seed (all rotations and kyoku of one seed are one cluster).

    python tools/grp_audit.py --log-dir DIR [--log-dir ...] --grp grp.pth --out audit.npz [--procs 8]
    python tools/grp_audit.py --report audit.npz [audit2.npz ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.protocol import SACRED, placement_pt, ranks  # noqa: E402

REF_NAME = "mortal-v4"
COLS = ("seed", "rot", "r", "kyoku", "honba", "kyotaku", "seat", "role", "score", "rank", "gap_up", "gap_down",
        "oya", "pred", "real")
SBIN_LABELS = ("<8k", "8-16k", "16-24k", "24-32k", "32-40k", ">=40k")
_grp = None


def _init(grp_path):
    global _grp
    import torch
    torch.set_num_threads(1)
    from riichi_eval.rollout.value import GrpValuer
    _grp = GrpValuer(grp_path)


def game_rows(path):
    try:
        g = read_game(path)
        m = NAME_RE.search(os.path.basename(path))
        seed, rot = int(m.group(1)), "abcd".index(m.group(3))
        fin = g.final_scores()
        real = [placement_pt(fin, s) for s in range(4)]
        seqs = [[_grp.feature(q.index, q.honba, q.kyotaku, q.scores) for q in g.rounds[:r + 1]]
                for r in range(len(g.rounds))]
        torch = _grp.torch
        with torch.inference_mode():
            mat = _grp.grp.calc_matrix(_grp.grp([torch.tensor(q, dtype=torch.float64) for q in seqs]))
            exp = (mat @ _grp.pts).numpy()                                  # (rounds, player)
        rows = []
        for r, q in enumerate(g.rounds):
            rk = ranks(q.scores)
            srt = sorted(q.scores, reverse=True)
            for s in range(4):
                up = srt[rk[s] - 1] - q.scores[s] if rk[s] > 0 else -1
                down = q.scores[s] - srt[rk[s] + 1] if rk[s] < 3 else -1
                rows.append((seed, rot, r, q.index, q.honba, q.kyotaku, s, int(g.names[s] == REF_NAME), q.scores[s],
                             rk[s], up, down, int(q.oya == s), float(exp[r, s]), real[s]))
        return rows
    except Exception as e:  # noqa: BLE001
        return path, repr(e)


def extract(args):
    paths = sorted(p for d in args.log_dir for p in glob.glob(os.path.join(d, "*.json.gz")))
    if args.max_games:
        paths = paths[: args.max_games]
    print(f"{len(paths):,} games", flush=True)
    t0 = time.time()
    rows, errors = [], []
    with Pool(args.procs, initializer=_init, initargs=(args.grp,)) as pool:
        for i, r in enumerate(pool.imap_unordered(game_rows, paths, chunksize=32), 1):
            (errors.append(r) if isinstance(r, tuple) else rows.extend(r))
            if i % 10000 == 0:
                print(f"  {i:,} ({time.time() - t0:.0f}s)", flush=True)
    a = np.array(rows, dtype=np.float64)
    np.savez_compressed(args.out, cols=np.array(COLS), data=a)
    print(f"-> {args.out}: {len(paths) - len(errors):,} games, {len(a):,} kyoku-seat rows, {len(errors)} errors, "
          f"{time.time() - t0:.0f}s")
    for p, e in errors[:5]:
        print("  error", p, e)


def _cluster(res, seed_idx, n_seeds):
    """Ratio mean of res with a seed-clustered 95% CI."""
    y = np.bincount(seed_idx, weights=res, minlength=n_seeds)
    n = np.bincount(seed_idx, minlength=n_seeds).astype(float)
    k = n > 0
    y, n = y[k], n[k]
    G = len(y)
    if G < 2:
        return float("nan"), float("nan"), int(n.sum())
    m = y.sum() / n.sum()
    return m, 1.96 * np.sqrt(G / (G - 1) * ((y - m * n) ** 2).sum()) / n.sum(), int(n.sum())


def report(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    cols = [str(c) for c in parts[0]["cols"]]
    a = np.concatenate([p["data"] for p in parts])
    c = {k: a[:, i] for i, k in enumerate(cols)}
    _, seed_idx = np.unique(c["seed"], return_inverse=True)
    G = seed_idx.max() + 1
    res = c["real"] - c["pred"]
    roles = {"tested model": c["role"] == 0, "reference (v4)": c["role"] == 1}
    print(f"{len(a):,} kyoku-seat rows from {len(np.unique(c['seed'] * 4 + c['rot'])):,} games, {G:,} seeds; "
          f"pt = {list(SACRED.pts)}; residual = realized - GRP expected (+ = GRP too pessimistic)\n")

    print("1. overall (a constant offset per role is expected: the GRP does not know who is sitting there)")
    for name, mk in roles.items():
        m, ci, n = _cluster(res[mk], seed_idx[mk], G)
        print(f"   {name:<16} rows {n:>9,}  GRP {c['pred'][mk].mean():+7.2f}  realized {c['real'][mk].mean():+7.2f}"
              f"  residual {m:+6.2f} ± {ci:.2f}")

    print("\n2. reliability (binned by GRP expected pt)")
    edges = [-200, -60, -35, -20, -10, 0, 10, 20, 35, 60, 200]
    print(f"   {'GRP bin':<14}" + "".join(f"{nm:>34}" for nm in roles))
    for lo, hi in zip(edges[:-1], edges[1:]):
        line = f"   [{lo:>4},{hi:>4})   "
        for mk in roles.values():
            b = mk & (c["pred"] >= lo) & (c["pred"] < hi)
            m, ci, n = _cluster(res[b], seed_idx[b], G)
            line += f"   n {n:>8,} GRP {c['pred'][b].mean() if n else float('nan'):+6.1f} res {m:+6.2f}±{ci:4.2f}"
        print(line)

    stages = [("East", (c["kyoku"] <= 3)), ("South 1-3", (c["kyoku"] >= 4) & (c["kyoku"] <= 6)),
              ("South 4", c["kyoku"] == 7), ("West+", c["kyoku"] >= 8)]
    sbins = [(-99999, 8000), (8000, 16000), (16000, 24000), (24000, 32000), (32000, 40000), (40000, 999999)]
    for name, mk in roles.items():
        print(f"\n3. residual by stage x own score at kyoku start — {name} (pt, ± 95% seed-clustered)")
        print(f"   {'stage':<10}" + "".join(f"{lab:>16}" for lab in SBIN_LABELS))
        for sn, smk in stages:
            line = f"   {sn:<10}"
            for lo, hi in sbins:
                b = mk & smk & (c["score"] >= lo) & (c["score"] < hi)
                m, ci, n = _cluster(res[b], seed_idx[b], G)
                line += f"{m:>+8.2f}±{ci:<5.2f}  " if n >= 200 else f"{'(n<200)':>16}"
            print(line)

    for name, mk in roles.items():
        print(f"\n4. residual by stage x rank at kyoku start — {name}")
        print(f"   {'stage':<10}" + "".join(f"{f'rank {k + 1}':>16}" for k in range(4)))
        for sn, smk in stages:
            line = f"   {sn:<10}"
            for k in range(4):
                b = mk & smk & (c["rank"] == k)
                m, ci, n = _cluster(res[b], seed_idx[b], G)
                line += f"{m:>+8.2f}±{ci:<5.2f}  " if n >= 200 else f"{'(n<200)':>16}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append")
    ap.add_argument("--grp")
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    else:
        if not (args.log_dir and args.grp and args.out):
            sys.exit("--log-dir, --grp and --out are required for extraction")
        extract(args)


if __name__ == "__main__":
    main()
