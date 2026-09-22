"""Mortal's 46-way action space <-> mjai events.

`action_to_event` mirrors libriichi's MortalBatchAgent::get_reaction (agent/mortal.rs) so that a replacement
built here is exactly what a Mortal engine choosing that action would have played.
"""
from __future__ import annotations

from .tiles import TILES, akaize, deaka, is_aka, kind, nxt, prv

ACT_EVENTS = {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku"}
TURN_EVENTS = {"tsumo", "end_kyoku", "end_game", "start_kyoku"}
N_ACTIONS = 46


def event_to_action(ev: dict) -> int:
    t = ev["type"]
    if t == "dahai":
        return TILES.index(ev["pai"])
    if t == "reach":
        return 37
    if t == "chi":
        p = kind(ev["pai"])
        c = sorted(kind(x) for x in ev["consumed"])
        if c[0] == p + 1:
            return 38
        if c[0] == p - 1:
            return 39
        return 40
    if t == "pon":
        return 41
    if t in ("daiminkan", "ankan", "kakan"):
        return 42
    if t == "hora":
        return 43
    if t == "ryukyoku":
        return 44
    return 45


def _call(x):
    return x() if callable(x) else x


def capture_ctx(ps, cans) -> dict:
    """What action_to_event needs, taken from the live PlayerState at the decision."""
    return dict(last_kawa=_call(ps.last_kawa_tile), last_tsumo=_call(ps.last_self_tsumo),
                akas=list(_call(ps.akas_in_hand)), ankan=list(_call(ps.ankan_candidates) or []),
                kakan=list(_call(ps.kakan_candidates) or []), target=int(cans.target_actor),
                can_daiminkan=bool(cans.can_daiminkan), can_ankan=bool(cans.can_ankan),
                can_kakan=bool(cans.can_kakan))


def action_to_event(a: int, actor: int, ctx: dict) -> dict | None:
    """None when the action cannot be built without Mortal's second kan-select pass (several kan choices)."""
    akas = ctx["akas"]
    if 0 <= a <= 36:
        pai = TILES[a]
        return {"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": ctx["last_tsumo"] == pai}
    if a == 37:
        return {"type": "reach", "actor": actor}
    if a in (38, 39, 40):
        pai = ctx["last_kawa"]
        d = deaka(pai)
        suit = {"m": 0, "p": 1, "s": 2}[d[1]]
        n = int(d[0])
        if a == 38:
            first = nxt(pai)
            ak = n in (3, 4) and akas[suit]
            consumed = [first, nxt(first)]
        elif a == 39:
            ak = n in (4, 6) and akas[suit]
            consumed = [prv(pai), nxt(pai)]
        else:
            last = prv(pai)
            ak = n in (6, 7) and akas[suit]
            consumed = [prv(last), last]
        if ak:
            consumed = [akaize(x) for x in consumed]
        return {"type": "chi", "actor": actor, "target": ctx["target"], "pai": pai, "consumed": consumed}
    if a == 41:
        pai = ctx["last_kawa"]
        ak = pai in ("5m", "5p", "5s") and akas["mps".index(pai[1])]   # a discarded red five never akaizes
        consumed = [akaize(pai), deaka(pai)] if ak else [deaka(pai), deaka(pai)]
        return {"type": "pon", "actor": actor, "target": ctx["target"], "pai": pai, "consumed": consumed}
    if a == 42:
        if ctx["can_daiminkan"]:
            tile = ctx["last_kawa"]
            consumed = [deaka(tile)] * 3 if is_aka(tile) else [akaize(tile), tile, tile]
            return {"type": "daiminkan", "actor": actor, "target": ctx["target"], "pai": tile, "consumed": consumed}
        cands = [("ankan", t) for t in ctx["ankan"]] + [("kakan", t) for t in ctx["kakan"]]
        if len(cands) != 1:
            return None
        k, tile = cands[0]
        if k == "ankan":
            return {"type": "ankan", "actor": actor, "consumed": [akaize(tile), tile, tile, tile]}
        d = deaka(tile)
        ak = tile in ("5m", "5p", "5s") and akas["mps".index(tile[1])]
        pai, consumed = (akaize(tile), [d] * 3) if ak else (d, [akaize(tile), d, d])
        return {"type": "kakan", "actor": actor, "pai": pai, "consumed": consumed}
    if a == 43:
        return {"type": "hora", "actor": actor, "target": ctx["target"]}
    if a == 44:
        return {"type": "ryukyoku"}
    return {"type": "none"}
