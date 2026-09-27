"""Fit and validate the luck control variates from vr_extract.py (variance reduction, stage 1).

    python tools/vr_fit.py feats1.npz [feats2.npz ...] [--folds 10]

Works at the level of duplicate seed groups (the four rotations of one seed), like the protocol CI:
  Y_g = mean challenger pt over the group, X_g = mean feature vector over the group.
Reports
  1. unbiasedness: every feature must have mean 0 (its terms are  h − E[h | history]); a mean clearly away from 0
     means the chance model or the tile accounting is wrong, and the feature must not be used;
  2. the raw estimate  mean(Y_g) ± CI;
  3. the corrected estimate  mean(Y_g − X_g·β) ± CI with β fitted by K-fold cross-fitting over seeds (each group's
     β comes from the other folds), the variance ratio and the equivalent number of games, for feature subsets.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def load(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    feats = [str(f) for f in parts[0]["features"]]
    seed = np.concatenate([p["seed"] for p in parts])
    rot = np.concatenate([p["rot"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    x = np.concatenate([p["x"] for p in parts])
    bad = np.concatenate([p["bad"] for p in parts])
    return feats, seed, rot, y, x, bad


def groups(seed, rot, y, x):
    order = np.lexsort((rot, seed))
    seed, rot, y, x = seed[order], rot[order], y[order], x[order]
    us, start, cnt = np.unique(seed, return_index=True, return_counts=True)
    keep = cnt == 4
    idx = start[keep][:, None] + np.arange(4)[None, :]
    if not (rot[idx] == np.arange(4)[None, :]).all():
        sys.exit("duplicate or missing rotations inside a seed group")
    return us[keep], y[idx].mean(1), x[idx].mean(1)


def ci(v):
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def crossfit(Y, X, seeds, folds):
    fold = (seeds * 2654435761 % 2 ** 32) % folds
    Yc = np.empty_like(Y)
    betas = []
    for f in range(folds):
        tr, te = fold != f, fold == f
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        coef, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
        betas.append(coef[1:])
        Yc[te] = Y[te] - X[te] @ coef[1:]
    return Yc, np.mean(betas, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--save-beta", help="fit the chosen subset on ALL groups and save the weights (reference fit)")
    ap.add_argument("--beta", help="apply saved weights instead of fitting (for evals too small to fit their own)")
    ap.add_argument("--subset", default="all", help="feature subset for --save-beta / --beta")
    ap.add_argument("--pair", help="with --beta: a second model's npz on the same seeds; report B - A raw and corrected "
                                   "(A = this run's npz, B = --pair)")
    args = ap.parse_args()
    feats, seed, rot, y, x, bad = load(args.npz)
    n_bad = int((bad > 0).sum())
    print(f"{len(y):,} games; {n_bad} with tile-count inconsistencies" + ("  <-- investigate" if n_bad else ""))
    seeds, Y, X = groups(seed, rot, y, x)
    G = len(Y)
    print(f"{G:,} complete seed groups ({4 * G:,} games)\n")

    print("1. unbiasedness (group means of each feature; must be ~0)")
    ok = []
    zs = []
    for j, f in enumerate(feats):
        m, c = X[:, j].mean(), ci(X[:, j])
        z = m / (c / 1.96) if c > 0 else 0.0
        flag = "" if abs(z) < 3.5 else "  <-- NOT zero-mean, excluded"
        if abs(z) < 3.5:
            ok.append(j)
        zs.append(z)
        if not f.startswith(("cell_", "doracell_")) or flag:
            print(f"   {f:<20} {m:+.5f} ± {c:.5f}  z={z:+.2f}  sd={X[:, j].std():.4f}{flag}")
    zc = np.array([z for z, f in zip(zs, feats) if f.startswith(("cell_", "doracell_"))])
    if len(zc):
        print(f"   ({len(zc)} cell features: |z| max {np.abs(zc).max():.2f}, share |z|>2 = {np.mean(np.abs(zc) > 2):.3f}"
              f" vs 0.046 expected under zero means)")

    raw_ci = ci(Y)
    print(f"\n2. raw: avg_pt {Y.mean():+.4f} ± {raw_ci:.4f}")
    if args.beta:
        z = np.load(args.beta, allow_pickle=False)
        names = [str(f) for f in z["features"]]
        cols = [feats.index(f) for f in names]
        Yc = Y - X[:, cols] @ z["beta"]
        r = Yc.var(ddof=1) / Y.var(ddof=1)
        print(f"\n3. fixed weights from {args.beta} ({len(cols)} features, fitted on {int(z['n_groups']):,} groups)")
        print(f"   avg_pt {Yc.mean():+.4f} ± {ci(Yc):.4f}   var ratio {r:.3f}  (= {1 / r:.2f}x games)   "
              f"shift vs raw {Yc.mean() - Y.mean():+.4f}")
        if args.pair:
            f2, seed2, rot2, y2, x2, _ = load([args.pair])
            s2, Y2, X2 = groups(seed2, rot2, y2, x2)
            c = np.intersect1d(seeds, s2)
            ia, ib = np.searchsorted(seeds, c), np.searchsorted(s2, c)
            cols2 = [f2.index(f) for f in names]
            Y2c = Y2 - X2[:, cols2] @ z["beta"]
            d, dc = Y2[ib] - Y[ia], Y2c[ib] - Yc[ia]
            print(f"\n4. paired B - A on {len(c):,} common seed groups (B = {args.pair})")
            print(f"   raw       {d.mean():+.4f} ± {ci(d):.4f}   r = {np.corrcoef(Y[ia], Y2[ib])[0, 1]:+.3f}")
            print(f"   corrected {dc.mean():+.4f} ± {ci(dc):.4f}   var ratio {dc.var(ddof=1) / d.var(ddof=1):.3f}")
        return

    print(f"\n3. control variates ({args.folds}-fold cross-fitted by seed)")
    subsets = {
        "deal": [j for j in ok if feats[j].startswith("deal_")],
        "draw win": [j for j in ok if feats[j].startswith("win_")],
        "draw dora": [j for j in ok if feats[j].startswith("dora_")],
        "draw adv": [j for j in ok if feats[j].startswith("adv")],
        "draws s2": [j for j in ok if feats[j].startswith(("win_", "dora_", "adv"))],
        "draws cells": [j for j in ok if feats[j].startswith(("win_", "cell_", "doracell_"))],
        "ura": [j for j in ok if feats[j].startswith("ura_")],
        "s2 all": [j for j in ok if not feats[j].startswith(("cell_", "doracell_"))],
        "all": ok,
    }
    for name, cols in subsets.items():
        if not cols:
            continue
        Yc, beta = crossfit(Y, X[:, cols], seeds, args.folds)
        r = Yc.var(ddof=1) / Y.var(ddof=1)
        print(f"   {name:<10} avg_pt {Yc.mean():+.4f} ± {ci(Yc):.4f}   var ratio {r:.3f}  "
              f"(= {1 / r:.2f}x games)   shift vs raw {Yc.mean() - Y.mean():+.4f}")
        if name == "s2 all":
            for j, b in zip(cols, beta):
                print(f"      beta {feats[j]:<20} {b:+9.3f}")
    if args.save_beta:
        cols = subsets[args.subset]
        A = np.column_stack([np.ones(G), X[:, cols]])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        np.savez(args.save_beta, features=np.array([feats[j] for j in cols]), beta=coef[1:], n_groups=G)
        print(f"\nsaved {len(cols)} weights ({args.subset}, fitted on all {G:,} groups) -> {args.save_beta}")


if __name__ == "__main__":
    main()
