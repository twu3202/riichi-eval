"""Extract per-game luck control variates from full-information 1v3 logs (variance reduction, stage 1).

Every feature is a sum of terms  h(outcome) − E[h(outcome) | history]  over chance events, so its expectation is
exactly zero whatever h is — the estimator  mean(Y − β·C)  stays unbiased for any β (β is fitted by cross-fitted
OLS in vr_fit.py). Chance events and why the conditional expectation is known exactly:

  - **deal** (per kyoku): the four dealt hands are exchangeable, so for any hand function f,
    f(challenger's hand) − mean over the four hands of f  has mean zero. f = shanten, dora count, red fives.
  - **draws** (every tsumo, rinshan included): the drawn tile is uniform over the tiles not yet revealed to an
    omniscient observer (136 minus all dealt hands, all earlier draws, revealed dora indicators).
      win: 1[tile completes the drawer's tenpai hand]  (waits from the drawer's PlayerState)
      dora: number of dora kinds hit + red five
      adv: 1[the tile lowers the drawer's shanten] (tenpai: completes the hand shape), from the Rust helper
           `arena.shanten_after_draw` on the drawer's closed hand; split by drawer role, drawer shanten before the
           draw (0/1/2/3+) and whether another player is in riichi
      advval: the same term times the dora count already in the drawer's hand, for shanten <= 1 (hand value)
  - **ura-dora** (riichi wins): each ura indicator is uniform over the unrevealed tiles, so
    hits − E[hits]  over the winner's 14+ tiles.

Terms are split by role relative to the challenger (its own draws / opponents'; for ura: it won, it dealt in,
an opponent's tsumo, an opponent's ron off another opponent), because the sign and size of the effect on the
challenger's result differ.

    python tools/vr_extract.py --log-dir DIR [--log-dir ...] --out feats.npz [--procs 12] [--max-games N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval import env  # noqa: E402
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.protocol import placement_pt  # noqa: E402
from riichi_eval.tiles import dora_of, is_aka, kind  # noqa: E402

ADV = tuple(f"adv_{r}_s{b}_{'r' if rt else 'n'}" for r in ("ch", "opp") for b in range(4) for rt in (0, 1))
FEATURES = (
    "deal_shanten", "deal_dora", "deal_aka",
    "win_ch", "win_opp", "dora_ch", "dora_opp",
    "ura_ch_win", "ura_ch_dealin", "ura_opp_tsumo", "ura_opp_ron_other",
    *ADV, "advval_ch", "advval_opp",
)
FI = {f: i for i, f in enumerate(FEATURES)}
AKA_KIND = {4: 0, 13: 1, 22: 2}
NEXT_KIND = np.array([dora_of(t) for t in
                      [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]])

REF_NAME = "mortal-v4"


def game_features(path: str):
    lib = env.setup()
    PS = lib.state.PlayerState
    g = read_game(path)
    seat = next(i for i, n in enumerate(g.names) if n != REF_NAME)
    y = placement_pt(g.final_scores(), seat)
    x = np.zeros(len(FEATURES))
    bad = 0
    ps = [PS(s) for s in range(4)]
    start = json.dumps({"type": "start_game", "names": g.names, "seed": list(g.seed)})
    for p in ps:
        p.update(start)
    for rd in g.rounds:
        unrev = np.full(34, 4)
        unrev_aka = np.ones(3, int)
        owned = np.zeros((4, 34), int)
        markers = []
        riichi = [False] * 4
        last_discard = last_kakan = None
        prev_type = None

        def take(tile):
            nonlocal bad
            k = kind(tile)
            unrev[k] -= 1
            if is_aka(tile):
                unrev_aka[AKA_KIND[k]] -= 1
            if unrev[k] < 0:
                bad += 1

        for ln, ev in zip(rd.lines, rd.events):
            t = ev["type"]
            if t == "start_kyoku":
                for p, hand in enumerate(ev["tehais"]):
                    for tile in hand:
                        take(tile)
                        owned[p, kind(tile)] += 1
                markers = [ev["dora_marker"]]
                take(ev["dora_marker"])
            elif t == "tsumo":
                p, tile = int(ev["actor"]), ev["pai"]
                role = "ch" if p == seat else "opp"
                tot = unrev.sum()
                waits = np.asarray(ps[p].waits, bool)
                if waits.any():
                    x[FI["win_" + role]] += float(waits[kind(tile)]) - unrev[waits].sum() / tot
                dk = [dora_of(m) for m in markers]
                e_dora = (sum(unrev[k] for k in dk) + unrev_aka.sum()) / tot
                h = sum(1 for k in dk if k == kind(tile)) + int(is_aka(tile))
                x[FI["dora_" + role]] += h - e_dora
                closed = np.frombuffer(bytes(ps[p].tehai), np.uint8).astype(np.int64)   # 34 closed-hand counts
                n_closed = int(closed.sum())
                if n_closed % 3 != 1:
                    bad += 1
                else:
                    before, after = lib.arena.shanten_after_draw([int(c) for c in closed], n_closed // 3)
                    adv = np.asarray(after) < before
                    term = float(adv[kind(tile)]) - unrev[adv].sum() / tot
                    rt = any(riichi[o] for o in range(4) if o != p)
                    x[FI[f"adv_{role}_s{min(int(before), 3)}_{'r' if rt else 'n'}"]] += term
                    if before <= 1:
                        x[FI["advval_" + role]] += term * min(int(sum(owned[p, k] for k in dk)), 4)
                take(tile)
                owned[p, kind(tile)] += 1
            elif t == "dahai":
                owned[int(ev["actor"]), kind(ev["pai"])] -= 1
                last_discard = ev["pai"]
            elif t in ("chi", "pon", "daiminkan"):
                owned[int(ev["actor"]), kind(ev["pai"])] += 1
            elif t == "kakan":
                last_kakan = ev["pai"]
            elif t == "reach_accepted":
                riichi[int(ev["actor"])] = True
            elif t == "dora":
                markers.append(ev["dora_marker"])
                take(ev["dora_marker"])
            elif t == "hora" and ev.get("ura_markers"):
                w, tg = int(ev["actor"]), int(ev["target"])
                hand = owned[w].copy()
                if tg != w:
                    hand[kind(last_kakan if prev_type == "kakan" else last_discard)] += 1
                tot = unrev.sum()
                n_ura = len(markers)
                e_hits = n_ura * float((unrev * hand[NEXT_KIND]).sum()) / tot
                hits = sum(hand[dora_of(m)] for m in ev["ura_markers"])
                if w == seat:
                    role = "ura_ch_win"
                elif tg == seat:
                    role = "ura_ch_dealin"
                elif tg == w:
                    role = "ura_opp_tsumo"
                else:
                    role = "ura_opp_ron_other"
                x[FI[role]] += hits - e_hits
            if t != "hora":
                prev_type = t
            for p in ps:
                p.update(ln)
            if t == "start_kyoku":
                sh = np.array([ps[i].shanten for i in range(4)], float)
                d = np.array([owned[i, dora_of(markers[0])] for i in range(4)], float)
                a = np.array([sum(is_aka(tl) for tl in ev["tehais"][i]) for i in range(4)], float)
                x[FI["deal_shanten"]] += sh[seat] - sh.mean()
                x[FI["deal_dora"]] += d[seat] - d.mean()
                x[FI["deal_aka"]] += a[seat] - a.mean()
    m = NAME_RE.search(os.path.basename(path))
    return int(m.group(1)), "abcd".index(m.group(3)), seat, y, x, bad


def _safe(path):
    try:
        return game_features(path)
    except Exception as e:  # noqa: BLE001
        return path, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()
    paths = sorted(p for d in args.log_dir for p in glob.glob(os.path.join(d, "*.json.gz")))
    if args.max_games:
        paths = paths[: args.max_games]
    print(f"{len(paths):,} games", flush=True)
    t0 = time.time()
    rows, errors = [], []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_safe, paths, chunksize=64), 1):
            (errors if len(r) == 2 else rows).append(r)
            if i % 20000 == 0:
                print(f"  {i:,} ({time.time() - t0:.0f}s)", flush=True)
    seed = np.array([r[0] for r in rows])
    rot = np.array([r[1] for r in rows], np.int8)
    ch = np.array([r[2] for r in rows], np.int8)
    y = np.array([r[3] for r in rows])
    x = np.stack([r[4] for r in rows])
    bad = np.array([r[5] for r in rows])
    np.savez_compressed(args.out, seed=seed, rot=rot, ch_seat=ch, y=y, x=x, bad=bad, features=np.array(FEATURES))
    print(f"-> {args.out}: {len(rows):,} games, {len(errors)} errors, {int((bad > 0).sum())} games with tile-count "
          f"inconsistencies, {time.time() - t0:.0f}s")
    for p, e in errors[:5]:
        print("  error", p, e)


if __name__ == "__main__":
    main()
