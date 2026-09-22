"""EV map: exact attribution of a model's realized result, relative to a reference policy, to its decisions.

Take a game the tested model ("ours") played in one seat. Let t_1 < ... < t_K be our decisions at which the
reference would have acted differently. G_k = outcome for our seat when the game is replayed from the log up to
t_k, the *reference* action is played at t_k, and the reference plays every seat afterwards (same seed, same
wall). Between disagreements the reference reproduces our logged actions, so the rollout that plays *our* action
at t_k and continues with the reference is exactly G_{k+1}; and G_{K+1} is the logged outcome. Hence

    logged outcome − G_1 = Σ_k (G_{k+1} − G_k),      G_1 = the reference playing our seat from the start,

an exact telescoping split of the realized difference into per-decision contributions Δ_k = G_{k+1} − G_k,
one rollout per disagreement. Averaged over games this is E[ours] − E[reference in our seat], i.e. the
duplicate gap, and summed over the four rotations of a seed the G_1 terms cancel (all-reference games are
identical, placement points sum to zero) — a built-in check.

Horizons: "game" values G by final placement pt; "kyoku" telescopes within each kyoku, valued by raw points
(deposits included) and, with a GRP, by expected placement pt at the start of the next kyoku.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..actions import action_to_event
from ..logs import GameLog
from ..protocol import SACRED, placement_pt
from ..replay import Decision, iter_decisions
from . import driver
from .spec import kyoku_spec
from .value import GrpValuer

LABEL = {**{a: "discard" for a in range(37)}, 37: "riichi", 38: "chi", 39: "chi", 40: "chi", 41: "pon",
         42: "kan", 43: "agari", 44: "ryukyoku", 45: "pass"}


def subtype(d: Decision, ours: int, ref: int) -> str:
    lo, lr = LABEL[ours], LABEL[ref]
    if d.exposed and lo in ("discard", "riichi") and lr in ("discard", "riichi"):
        po, pr = d.push_fold(ours), d.push_fold(ref)
        if lo == lr == "discard":
            return {("push", "fold"): "exposed: we push, ref folds",
                    ("fold", "push"): "exposed: we fold, ref pushes",
                    ("push", "push"): "exposed: different push tile",
                    ("fold", "fold"): "exposed: different genbutsu"}[(po, pr)]
        return f"exposed: {lo} vs ref {lr}"
    if lo == lr == "discard":
        return f"{d.bucket}: tile choice"
    return f"{d.bucket}: {lo} vs ref {lr}"


@dataclass
class Point:
    game: int
    r: int
    pos: int
    actor: int
    ours: int
    ref: int
    subtype: str
    bucket: str
    exposed: bool
    shanten: int
    opened: bool
    turn: int
    alt: dict | None


def _forward(engine, obs, masks, batch=4096) -> np.ndarray:
    out = []
    for s in range(0, len(obs), batch):
        acts = engine.react_batch(obs[s:s + batch], masks[s:s + batch], None)[0]
        out.extend(acts)
    return np.asarray(out, int)


def find_points(games: list[GameLog], seats: list[int], ours_engine, ref_engine, version: int = 4):
    """Returns (points, positive-control [agree, total])."""
    decs, owner = [], []
    for gi, (g, seat) in enumerate(zip(games, seats)):
        for d in iter_decisions(g, [seat], version):
            if d.bucket != "forced" and d.logged >= 0:
                decs.append(d)
                owner.append(gi)
    if not decs:
        return [], [0, 0]
    obs = [d.obs for d in decs]
    masks = [d.mask for d in decs]
    a_ours = _forward(ours_engine, obs, masks)
    a_ref = _forward(ref_engine, obs, masks)
    logged = np.array([d.logged for d in decs])
    pc = [int((a_ours == logged).sum()), len(decs)]
    points = []
    for d, gi, ar in zip(decs, owner, a_ref):
        if int(ar) == d.logged:
            continue
        points.append(Point(game=gi, r=d.r, pos=d.pos, actor=d.actor, ours=d.logged, ref=int(ar),
                            subtype=subtype(d, d.logged, int(ar)), bucket=d.bucket, exposed=d.exposed,
                            shanten=d.shanten, opened=d.opened, turn=d.turn,
                            alt=action_to_event(int(ar), d.actor, d.ctx)))
    return points, pc


def _logged_round_end(rd, nxt) -> list[int]:
    if nxt is not None:
        return list(nxt.scores)
    sc, reaches, deltas = list(rd.scores), [0] * 4, [0] * 4
    for ev in rd.events:
        if ev["type"] == "reach_accepted":
            reaches[int(ev["actor"])] += 1
        elif ev["type"] in ("hora", "ryukyoku"):
            for i, v in enumerate(ev.get("deltas") or [0] * 4):
                deltas[i] += int(v)
    return [sc[i] - 1000 * reaches[i] + deltas[i] for i in range(4)]


def evaluate(games: list[GameLog], seats: list[int], points: list[Point], ref_engine, *, horizon: str = "kyoku",
             grp: GrpValuer | None = None, pts=SACRED.pts, batch: int = 512):
    """Rolls out every point (reference action, reference continuation) and returns per-point records with
    telescoped deltas, plus per-game G_1 values for the zero check."""
    ok = [p for p in points if p.alt is not None]
    specs = [kyoku_spec(games[p.game], p.r, seat_engines=[0, 0, 0, 0], branch_pos=p.pos, branch_actor=p.actor,
                        replacement=json.dumps(p.alt), horizon=horizon) for p in ok]
    res = driver.run([ref_engine], specs, batch=batch) if specs else []
    val = {}
    grp_q, grp_seat, grp_key = [], [], []
    for p, x in zip(ok, res):
        if x.error:
            continue
        g, seat, rd = games[p.game], seats[p.game], games[p.game].rounds[p.r]
        v = {"kyoku_raw": x.kyoku_scores[seat] - rd.scores[seat]}
        if horizon == "game":
            v["game_pt"] = placement_pt(x.final_scores, seat, pts)
        if grp is not None:
            if x.next_kyoku is None:
                v["kyoku_grp"] = placement_pt(x.final_scores, seat, pts)
            else:
                pre = [grp.feature(q.index, q.honba, q.kyotaku, q.scores) for q in g.rounds[:p.r + 1]]
                grp_q.append(pre + [grp.feature(x.next_kyoku[0], x.next_kyoku[1], x.kyoku_kyotaku, x.kyoku_scores)])
                grp_seat.append(seat)
                grp_key.append(id(p))
        val[id(p)] = v
    # logged values per game / kyoku
    logged = {}
    for gi, (g, seat) in enumerate(zip(games, seats)):
        fin = g.final_scores()
        lg = {"game_pt": placement_pt(fin, seat, pts), "kyoku_raw": {}, "kyoku_grp": {}}
        for r, rd in enumerate(g.rounds):
            nxt = g.rounds[r + 1] if r + 1 < len(g.rounds) else None
            lg["kyoku_raw"][r] = _logged_round_end(rd, nxt)[seat] - rd.scores[seat]
            if grp is not None:
                if nxt is None:
                    lg["kyoku_grp"][r] = placement_pt(fin, seat, pts)
                else:
                    pre = [grp.feature(q.index, q.honba, q.kyotaku, q.scores) for q in g.rounds[:r + 1]]
                    grp_q.append(pre + [grp.feature(nxt.index, nxt.honba, nxt.kyotaku, nxt.scores)])
                    grp_seat.append(seat)
                    grp_key.append(("logged", gi, r))
        logged[gi] = lg
    if grp is not None and grp_q:
        for k, e in zip(grp_key, grp.expected_pt(grp_q, grp_seat)):
            if isinstance(k, tuple):
                logged[k[1]]["kyoku_grp"][k[2]] = e
            else:
                val[k]["kyoku_grp"] = e

    by_game = defaultdict(list)
    for p in points:
        by_game[p.game].append(p)
    records, g1 = [], {}
    metrics_k = ["kyoku_raw"] + (["kyoku_grp"] if grp is not None else [])
    for gi in range(len(games)):
        pl = sorted(by_game.get(gi, []), key=lambda p: (p.r, p.pos))
        complete = all(id(p) in val for p in pl)
        # game horizon
        if horizon == "game":
            if complete:
                G = [val[id(p)]["game_pt"] for p in pl] + [logged[gi]["game_pt"]]
                g1[gi] = G[0]
                for k, p in enumerate(pl):
                    val[id(p)]["d_game_pt"] = G[k + 1] - G[k]
            else:
                g1[gi] = None
        # kyoku horizon: telescope within each kyoku
        by_r = defaultdict(list)
        for p in pl:
            by_r[p.r].append(p)
        for r, rp in by_r.items():
            if not all(id(p) in val for p in rp):
                continue
            for m in metrics_k:
                G = [val[id(p)][m] for p in rp] + [logged[gi][m][r]]
                for k, p in enumerate(rp):
                    val[id(p)]["d_" + m] = G[k + 1] - G[k]
        if horizon != "game":
            g1[gi] = logged[gi]["game_pt"] if not pl else None
    for p in points:
        rec = {k: getattr(p, k) for k in ("game", "r", "pos", "actor", "ours", "ref", "subtype", "bucket",
                                           "exposed", "shanten", "opened", "turn")}
        rec["supported"] = p.alt is not None
        rec.update({k: v for k, v in val.get(id(p), {}).items() if k.startswith("d_")})
        rec["ok"] = id(p) in val
        records.append(rec)
    return records, g1
