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
    args = ap.parse_args()
    feats, seed, rot, y, x, bad = load(args.npz)
    n_bad = int((bad > 0).sum())
    print(f"{len(y):,} games; {n_bad} with tile-count inconsistencies" + ("  <-- investigate" if n_bad else ""))
    seeds, Y, X = groups(seed, rot, y, x)
    G = len(Y)
    print(f"{G:,} complete seed groups ({4 * G:,} games)\n")

    print("1. unbiasedness (group means of each feature; must be ~0)")
    ok = []
    for j, f in enumerate(feats):
        m, c = X[:, j].mean(), ci(X[:, j])
        z = m / (c / 1.96) if c > 0 else 0.0
        flag = "" if abs(z) < 3 else "  <-- NOT zero-mean, excluded"
        if abs(z) < 3:
            ok.append(j)
        print(f"   {f:<20} {m:+.5f} ± {c:.5f}  z={z:+.2f}  sd={X[:, j].std():.4f}{flag}")

    raw_ci = ci(Y)
    print(f"\n2. raw: avg_pt {Y.mean():+.4f} ± {raw_ci:.4f}")

    print(f"\n3. control variates ({args.folds}-fold cross-fitted by seed)")
    subsets = {
        "deal": [j for j in ok if feats[j].startswith("deal_")],
        "draw win": [j for j in ok if feats[j].startswith("win_")],
        "draw dora": [j for j in ok if feats[j].startswith("dora_")],
        "draw adv": [j for j in ok if feats[j].startswith("adv")],
        "draws all": [j for j in ok if feats[j].startswith(("win_", "dora_", "adv"))],
        "ura": [j for j in ok if feats[j].startswith("ura_")],
        "all": ok,
    }
    for name, cols in subsets.items():
        if not cols:
            continue
        Yc, beta = crossfit(Y, X[:, cols], seeds, args.folds)
        r = Yc.var(ddof=1) / Y.var(ddof=1)
        print(f"   {name:<10} avg_pt {Yc.mean():+.4f} ± {ci(Yc):.4f}   var ratio {r:.3f}  "
              f"(= {1 / r:.2f}x games)   shift vs raw {Yc.mean() - Y.mean():+.4f}")
        if name == "all":
            for j, b in zip(cols, beta):
                print(f"      beta {feats[j]:<20} {b:+9.3f}")


if __name__ == "__main__":
    main()
