"""Calibration probe: what does flipping our own risky choices buy, and can our own Q values gate the flip?

No reference policy is involved. At each of our decisions of interest we roll out the *alternative* action with
our own model continuing in our seat (opponents unchanged), and compare against the logged kyoku:

    gain = value(alternative) − value(logged)

Two candidate sets:
  - **push→fold**: an exposed decision (an opponent's riichi is accepted, we are not in riichi) where we pushed
    (non-genbutsu discard, or riichi) and a genbutsu discard was legal; the alternative is our own best genbutsu.
  - **riichi→dama**: we declared riichi; the alternative is our own best discard.

Two reverse sets (for training labels, so both sides of each decision get priced):
  - **fold→push**: an exposed decision where we discarded a genbutsu and a non-genbutsu discard was legal;
    the alternative is our own best non-genbutsu discard.
  - **dama→riichi**: riichi was legal and we discarded without it; the alternative is riichi (our model then
    picks the riichi tile).

Each candidate also carries `margin` = Q_ours[what we did] − Q_ours[alternative] ≥ 0, so a rule "flip when
margin < τ" can be scored offline for any τ. Flips are priced one at a time, so the total for a τ is a
first-order estimate of applying the rule everywhere; a real engine-level eval is the confirmation step.

With `commit=True` the alternative is a macro-action rather than one discard: after the flip our model keeps
folding (push→fold: best genbutsu whenever one is legal, no riichi / calls / kan) or keeps dama (riichi→dama)
for the rest of the kyoku. Without it, a one-tile fold is often followed by our model pushing again next turn,
which prices "fold once" rather than "fold".

If a reference engine is given, each candidate also records what that engine would do at the same state
(`ref_flip`: it would not push / not riichi either; `ref_margin` = Q_ref[ours] − Q_ref[alt]). This is a
diagnostic filter only — the alternative is always our own model's choice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..actions import action_to_event
from ..logs import GameLog
from ..protocol import SACRED, placement_pt
from ..replay import iter_decisions
from . import driver
from .ev_map import _logged_round_end
from .spec import kyoku_spec
from .value import GrpValuer


@dataclass
class Candidate:
    game: int
    r: int
    pos: int
    actor: int
    kind: str          # "push2fold" | "riichi2dama" | "fold2push" | "dama2riichi" | "eff"
    ours: int
    alt: int
    margin: float
    exposed: bool
    shanten: int
    opened: bool
    turn: int
    n_riichi: int
    dora: int
    alt_event: dict          # mjai event for the alternative action
    alt_ev: dict | None = None
    ref_flip: bool | None = None     # the reference engine would not push / not riichi here either
    ref_margin: float | None = None  # Q_ref[ours] − Q_ref[alt]
    eff_sd: bool | None = None       # "eff": our discard goes back in shanten while the efficiency discard does not
    eff_dk: int | None = None        # "eff": required-tile kinds of the efficiency discard minus ours (same shanten)


KINDS = ("push2fold", "riichi2dama")
ALL_KINDS = ("push2fold", "riichi2dama", "fold2push", "dama2riichi", "eff")

# "eff": a routine discard (no opponent riichi, riichi not legal) where we did not play the pure tile-efficiency
# discard; the alternative is that discard. libriichi writes it into v4 observations as a one-hot row: among the
# discards that keep shanten, the one with the most required tiles (ties by discard priority), i.e. classic
# ukeire-max efficiency with no model involved. Present only when the single-player tables are computed.
EFF_ROW = 959
REQ_ROW = 891          # rows 891..924: tiles required after discarding kind k (shanten kept); 925..958: shanten down
_KIND_OF = list(range(34)) + [4, 13, 22]
_AKA = {4: 34, 13: 35, 22: 36}


def eff_action(obs, mask):
    """The legal discard action of the pure-efficiency choice, or None."""
    row = obs[EFF_ROW]
    k = np.flatnonzero(row > 0.5)
    if len(k) != 1:
        return None
    k = int(k[0])
    if mask[k]:
        return k
    a = _AKA.get(k)
    return a if a is not None and mask[a] else None


def find_candidates(games: list[GameLog], seats: list[int], engine, version: int = 4, batch: int = 4096,
                    ref_engine=None, kinds=KINDS):
    decs, owner = [], []
    for gi, (g, seat) in enumerate(zip(games, seats)):
        for d in iter_decisions(g, [seat], version):
            if d.bucket != "forced" and d.logged >= 0:
                decs.append(d)
                owner.append(gi)
    if not decs:
        return [], [0, 0]
    q = []
    for s in range(0, len(decs), batch):
        q.extend(engine.react_batch([d.obs for d in decs[s:s + batch]], [d.mask for d in decs[s:s + batch]], None)[1])
    pc = [0, 0]
    out, cdec = [], []
    for d, gi, qv in zip(decs, owner, q):
        qv = np.asarray(qv, np.float64)
        qv[~d.mask] = -np.inf
        pc[0] += int(qv.argmax() == d.logged)
        pc[1] += 1
        def add(kind, alt):
            ev = action_to_event(int(alt), d.actor, d.ctx)
            if ev is not None:
                out.append(Candidate(gi, d.r, d.pos, d.actor, kind, d.logged, int(alt),
                                     float(qv[d.logged] - qv[alt]), d.exposed, d.shanten, d.opened, d.turn,
                                     d.n_riichi, d.dora, ev))
                cdec.append(d)

        if d.logged == 37 and "riichi2dama" in kinds:
            disc = [a for a in range(37) if d.mask[a]]
            if disc:
                add("riichi2dama", max(disc, key=lambda a: qv[a]))
        if d.exposed and d.push_fold(d.logged) == "push" and "push2fold" in kinds:
            safe = [a for a in range(37) if d.mask[a] and d.push_fold(a) == "fold"]
            if safe:
                add("push2fold", max(safe, key=lambda a: qv[a]))
        if d.exposed and d.push_fold(d.logged) == "fold" and "fold2push" in kinds:
            push = [a for a in range(37) if d.mask[a] and d.push_fold(a) == "push"]
            if push:
                add("fold2push", max(push, key=lambda a: qv[a]))
        if d.mask[37] and 0 <= d.logged <= 36 and "dama2riichi" in kinds:
            add("dama2riichi", 37)
        if "eff" in kinds and 0 <= d.logged <= 36 and not d.exposed and not d.mask[37]:
            a = eff_action(d.obs, d.mask)
            if a is not None and _KIND_OF[a] != _KIND_OF[d.logged]:
                add("eff", a)
                if out and out[-1].kind == "eff" and out[-1].alt == a:
                    ko, ke = _KIND_OF[d.logged], _KIND_OF[a]
                    keep_o = bool(d.obs[REQ_ROW + ko].any())
                    out[-1].eff_sd = bool(d.obs[REQ_ROW + 34 + ko].any())
                    out[-1].eff_dk = int(d.obs[REQ_ROW + ke].sum() - d.obs[REQ_ROW + ko].sum()) if keep_o else None
    if ref_engine is not None and cdec:
        rq = []
        for s in range(0, len(cdec), batch):
            rq.extend(ref_engine.react_batch([d.obs for d in cdec[s:s + batch]],
                                             [d.mask for d in cdec[s:s + batch]], None)[1])
        for c, d, qv in zip(out, cdec, rq):
            qv = np.asarray(qv, np.float64)
            qv[~d.mask] = -np.inf
            best = int(qv.argmax())
            c.ref_flip = {"riichi2dama": best != 37, "dama2riichi": best == 37,
                          "push2fold": d.push_fold(best) == "fold",
                          "fold2push": d.push_fold(best) == "push",
                          "eff": best <= 36 and _KIND_OF[best] == _KIND_OF[c.alt]}[c.kind]
            c.ref_margin = float(qv[c.ours] - qv[c.alt])
    return out, pc


COMMIT = {"push2fold": "fold", "riichi2dama": "dama"}   # the reverse kinds have no macro-action


def price(games, seats, cands: list[Candidate], ours_engine, ref_engine, *, grp: GrpValuer | None = None,
          pts=SACRED.pts, batch: int = 1024, commit: bool = False):
    """Roll out each candidate's alternative with our own model continuing in our seat (see module doc
    for `commit`)."""
    specs, keep = [], []
    for c in cands:
        g, seat = games[c.game], seats[c.game]
        specs.append(kyoku_spec(g, c.r, seat_engines=[0 if s == seat else 1 for s in range(4)],
                                branch_pos=c.pos, branch_actor=c.actor,
                                replacement=json.dumps(c.alt_event), horizon="kyoku",
                                commit=COMMIT.get(c.kind, "") if commit else ""))
        keep.append(c)
    res = driver.run([ours_engine, ref_engine], specs, batch=batch) if specs else []
    grp_q, grp_seat, grp_key = [], [], []
    logged_cache = {}
    for c, x in zip(keep, res):
        if x.error:
            continue
        g, seat = games[c.game], seats[c.game]
        rd = g.rounds[c.r]
        key = (c.game, c.r)
        if key not in logged_cache:
            nxt = g.rounds[c.r + 1] if c.r + 1 < len(g.rounds) else None
            raw = _logged_round_end(rd, nxt)[seat] - rd.scores[seat]
            logged_cache[key] = {"raw": raw, "nxt": nxt}
        c.alt_ev = {"raw": (x.kyoku_scores[seat] - rd.scores[seat]) - logged_cache[key]["raw"]}
        if grp is not None:
            pre = [grp.feature(q.index, q.honba, q.kyotaku, q.scores) for q in g.rounds[:c.r + 1]]
            if x.next_kyoku is None:
                c.alt_ev["grp_alt"] = placement_pt(x.final_scores, seat, pts)
            else:
                grp_q.append(pre + [grp.feature(x.next_kyoku[0], x.next_kyoku[1], x.kyoku_kyotaku, x.kyoku_scores)])
                grp_seat.append(seat)
                grp_key.append(("alt", id(c)))
            nxt = logged_cache[key]["nxt"]
            if nxt is None:
                logged_cache[key]["grp"] = placement_pt(g.final_scores(), seat, pts)
            elif "grp" not in logged_cache[key]:
                grp_q.append(pre + [grp.feature(nxt.index, nxt.honba, nxt.kyotaku, nxt.scores)])
                grp_seat.append(seat)
                grp_key.append(("logged", key))
    if grp is not None and grp_q:
        vals = grp.expected_pt(grp_q, grp_seat)
        by_id = {}
        for k, v in zip(grp_key, vals):
            if k[0] == "alt":
                by_id[k[1]] = v
            else:
                logged_cache[k[1]]["grp"] = v
        for c in keep:
            if c.alt_ev is None:
                continue
            base = logged_cache[(c.game, c.r)]["grp"]
            alt = c.alt_ev.pop("grp_alt", None)
            if alt is None:
                alt = by_id.get(id(c))
            c.alt_ev["grp"] = None if alt is None else alt - base
    return keep
