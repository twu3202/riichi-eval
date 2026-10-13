"""Rollout specs: restart a logged game at the start of one kyoku, replay up to a branch, then play on."""
from __future__ import annotations

import random

from ..logs import GameLog, Round

NEVER = 10 ** 9     # branch position that is never reached: the whole kyoku is replayed from the log


def kyoku_spec(game: GameLog, r: int, *, seat_engines, branch_pos: int = NEVER, branch_actor: int = 0,
               replacement: str | None = None, horizon: str = "kyoku", commit: str = "",
               reshuffle: int | None = None, wall: tuple | None = None) -> dict:
    """branch_pos = live log length at the branch decision, i.e. the index (within the kyoku's events,
    start_kyoku = 0) of the event that resolves that decision in the reference log.
    Before it every seat replays the logged reaction; at it `branch_actor` plays `replacement`
    (mjai JSON) or, if None, its engine; after it every seat plays its engine.
    horizon: "kyoku" stops after this kyoku, "game" plays the game out.
    commit: "" / "fold" / "dama" — after the replacement `branch_actor` keeps folding (best genbutsu whenever one
    is legal, no riichi / calls / kan) or keeps dama (no riichi) for the rest of the branch kyoku.
    reshuffle: None plays the logged wall; an integer re-deals the tiles still hidden at the branch (rest of the
    live wall, unused rinshan, unrevealed dora indicators, all ura indicators) with that seed. Hands and the
    replayed prefix are unchanged, so given the logged hands this samples the wall exactly (no belief model).
    wall: None takes the wall from the game's libriichi seed; a log without one (a human Tenhou game) passes
    `log_wall(round)` and should also reshuffle, since its unseen tiles are only placeholders."""
    rd = game.rounds[r]
    if horizon not in ("kyoku", "game"):
        raise ValueError(horizon)
    if commit not in ("", "fold", "dama"):
        raise ValueError(commit)
    return dict(seed=tuple(game.seed), kyoku=rd.index, honba=rd.honba, kyotaku=rd.kyotaku,
                scores=list(rd.scores),
                # only gates whether the first kyoku may start; it did start in the logged game
                in_renchan=True,
                seat_engines=list(seat_engines), reference=rd.lines, branch_pos=int(branch_pos),
                branch_actor=int(branch_actor), replacement=replacement, horizon=horizon, commit=commit,
                reshuffle=None if reshuffle is None else int(reshuffle), wall=wall)


TILE_SET = [t for k in [f"{n}{s}" for s in "mps" for n in range(1, 10)] + list("ESWNPFC")
            for t in ([k] * 3 + [k + "r"] if k in ("5m", "5p", "5s") else [k] * 4)]   # libriichi's 136, aka 1 each


def log_wall(rd: Round, seed: int = 0) -> tuple:
    """The wall of a logged kyoku in libriichi's Board layout (see rollout.rs `wall`): the dealt hands, every
    logged draw from the live wall and from the rinshan, the revealed dora indicators and any ura indicators shown
    at a win, each in the slot it was taken from; the tiles that never appear are placed at random (seeded) in
    the remaining slots. Raises ValueError if the log does not fit one wall."""
    sk = rd.events[0]
    hands = [list(h) for h in sk["tehais"]]
    yama, rinshan, dora, ura = [], [], [sk["dora_marker"]], []
    after_kan = False
    for ev in rd.events[1:]:
        t = ev["type"]
        if t in ("ankan", "kakan", "daiminkan"):
            after_kan = True
        elif t == "tsumo":
            (rinshan if after_kan else yama).append(ev["pai"])
            after_kan = False
        elif t == "dora":
            dora.append(ev["dora_marker"])
        elif t == "hora" and len(ev.get("ura_markers") or []) > len(ura):
            ura = list(ev["ura_markers"])
    if any(len(h) != 13 for h in hands) or len(yama) > 70 or len(rinshan) > 4 or len(dora) > 5 or len(ura) > 5:
        raise ValueError("log does not fit one wall (hand / draw / indicator counts)")
    unseen = list(TILE_SET)
    for t in [x for h in hands for x in h] + yama + rinshan + dora + ura:
        try:
            unseen.remove(t)
        except ValueError:
            raise ValueError(f"tile {t} appears more often than the set has it") from None
    random.Random(seed).shuffle(unseen)
    take = lambda n: [unseen.pop() for _ in range(n)]
    out = (hands, take(70 - len(yama)) + yama[::-1], take(4 - len(rinshan)) + rinshan[::-1],
           take(5 - len(dora)) + dora[::-1], ura + take(5 - len(ura)))
    assert not unseen
    return out
