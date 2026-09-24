"""Replaying a logged game with four libriichi PlayerStates and extracting decision points."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import env
from .actions import capture_ctx, event_to_action
from .logs import GameLog
from .tiles import dora_of, kind

BUCKETS = ("forced", "agari", "riichi", "call", "self_kan", "ryukyoku", "discard")


@dataclass
class Decision:
    r: int                  # index into game.rounds
    pos: int                # live log length at the decision (= rollout branch_pos)
    actor: int
    obs: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    logged: int             # action actually taken (-1 if it cannot be recovered)
    ctx: dict = field(repr=False)
    bucket: str
    exposed: bool           # an opponent's riichi is accepted and we are not in riichi
    safe: frozenset | None  # genbutsu kinds against every riichi opponent (exposed decisions only)
    shanten: int
    opened: bool
    turn: int               # own discards so far this kyoku
    n_riichi: int
    dora: int               # dora + red fives in hand

    def push_fold(self, a: int) -> str:
        """For exposed decisions: fold = discard a genbutsu; push = any other discard or riichi."""
        if a == 37:
            return "push"
        if 0 <= a <= 36:
            return "fold" if self.safe is not None and kind_of_action(a) in self.safe else "push"
        return "other"


def kind_of_action(a: int) -> int:
    return {34: 4, 35: 13, 36: 22}.get(a, a)


def bucket_of(mask: np.ndarray, cans) -> str:
    if mask.sum() <= 1:
        return "forced"
    if mask[43]:
        return "agari"
    if mask[37]:
        return "riichi"
    if cans.can_pass and mask[38:43].any():
        return "call"
    if mask[42]:
        return "self_kan"
    if mask[44]:
        return "ryukyoku"
    return "discard"


def _logged_action(events: list[dict], j: int, me: int, cans) -> int:
    """Mirror of libriichi ForcedPrefixAgent's reference rule: what `me` did at the decision after event j."""
    for ev in events[j + 1:]:
        t = ev["type"]
        if t in ("reach_accepted", "dora"):
            continue
        if t == "hora" and int(ev["actor"]) != me:
            continue
        if t == "ryukyoku":
            return 44 if cans.can_ryukyoku else 45
        if t in ("dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora") and int(ev["actor"]) == me:
            return event_to_action(ev)
        return 45 if cans.can_pass else -1
    return 45 if cans.can_pass else -1


def iter_decisions(game: GameLog, seats, version: int = 4, select=None):
    """Every can-act point of the given seats, in log order, skipping forced ones only via `bucket`.
    `select(r, pos, seat) -> bool`, if given, limits which points are encoded and yielded (the replay itself
    still runs over every event); training loaders use it to encode only the states they need."""
    lib = env.setup()
    PlayerState = lib.state.PlayerState
    seats = set(seats)
    ps = [PlayerState(s) for s in range(4)]
    start = json.dumps({"type": "start_game", "names": game.names, "seed": list(game.seed)})
    for p in ps:
        p.update(start)
    for r, rd in enumerate(game.rounds):
        riichi = [False] * 4
        own = [set() for _ in range(4)]
        after = [set() for _ in range(4)]
        dora, ndisc = [], [0] * 4
        for j, (ln, ev) in enumerate(zip(rd.lines, rd.events)):
            t = ev["type"]
            if t == "start_kyoku":
                dora = [dora_of(ev["dora_marker"])]
            elif t == "dora":
                dora.append(dora_of(ev["dora_marker"]))
            elif t == "dahai":
                a, k = int(ev["actor"]), kind(ev["pai"])
                own[a].add(k)
                ndisc[a] += 1
                for o in range(4):
                    if riichi[o] and o != a:
                        after[o].add(k)
            elif t == "reach_accepted":
                riichi[int(ev["actor"])] = True
            cands = [ps[s].update(ln) for s in range(4)]
            for s in seats:
                cans = cands[s]
                if not cans.can_act or (select is not None and not select(r, j + 1, s)):
                    continue
                p = ps[s]
                obs, mask = p.encode_obs(version, False)
                mask = np.asarray(mask, bool)
                rs = [o for o in range(4) if o != s and riichi[o]]
                exposed = bool(rs) and not (p.self_riichi_accepted or p.self_riichi_declared)
                safe = frozenset(set.intersection(*[own[o] | after[o] for o in rs])) if exposed else None
                tehai = p.tehai
                yield Decision(
                    r=r, pos=j + 1, actor=s, obs=np.asarray(obs, np.float32), mask=mask,
                    logged=_logged_action(rd.events, j, s, cans), ctx=capture_ctx(p, cans),
                    bucket=bucket_of(mask, cans), exposed=exposed, safe=safe, shanten=int(p.shanten),
                    opened=(len(p.chis) + len(p.pons) + len(p.minkans)) > 0, turn=ndisc[s], n_riichi=len(rs),
                    dora=sum(int(tehai[d]) for d in dora) + sum(bool(x) for x in p.akas_in_hand))
