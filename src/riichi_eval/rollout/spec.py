"""Rollout specs: restart a logged game at the start of one kyoku, replay up to a branch, then play on."""
from __future__ import annotations

from ..logs import GameLog

NEVER = 10 ** 9     # branch position that is never reached: the whole kyoku is replayed from the log


def kyoku_spec(game: GameLog, r: int, *, seat_engines, branch_pos: int = NEVER, branch_actor: int = 0,
               replacement: str | None = None, horizon: str = "kyoku", commit: str = "",
               reshuffle: int | None = None) -> dict:
    """branch_pos = live log length at the branch decision, i.e. the index (within the kyoku's events,
    start_kyoku = 0) of the event that resolves that decision in the reference log.
    Before it every seat replays the logged reaction; at it `branch_actor` plays `replacement`
    (mjai JSON) or, if None, its engine; after it every seat plays its engine.
    horizon: "kyoku" stops after this kyoku, "game" plays the game out.
    commit: "" / "fold" / "dama" — after the replacement `branch_actor` keeps folding (best genbutsu whenever one
    is legal, no riichi / calls / kan) or keeps dama (no riichi) for the rest of the branch kyoku.
    reshuffle: None plays the logged wall; an integer re-deals the tiles still hidden at the branch (rest of the
    live wall, unused rinshan, unrevealed dora indicators, all ura indicators) with that seed. Hands and the
    replayed prefix are unchanged, so given the logged hands this samples the wall exactly (no belief model)."""
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
                reshuffle=None if reshuffle is None else int(reshuffle))
