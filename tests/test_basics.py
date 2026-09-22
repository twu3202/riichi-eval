"""Pure unit tests (no libriichi, no GPU): tile arithmetic, action <-> event, placement."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from riichi_eval.actions import action_to_event, event_to_action  # noqa: E402
from riichi_eval.protocol import SACRED, placement_pt, ranks  # noqa: E402
from riichi_eval.tiles import INDEX, dora_of, nxt, prv  # noqa: E402


def ctx(**kw):
    base = dict(last_kawa=None, last_tsumo=None, akas=[False, False, False], ankan=[], kakan=[], target=0,
                can_daiminkan=False, can_ankan=False, can_kakan=False)
    base.update(kw)
    return base


def test_tile_cycles():
    assert nxt("9m") == "1m" and prv("1s") == "9s"
    assert nxt("N") == "E" and prv("E") == "N"
    assert nxt("C") == "P" and prv("P") == "C"
    assert nxt("5pr") == "6p"
    assert dora_of("9p") == INDEX["1p"] and dora_of("W") == INDEX["N"] and dora_of("F") == INDEX["C"]


def test_roundtrip_chi_pon():
    for a, kawa in ((38, "3m"), (39, "4s"), (40, "7p")):
        ev = action_to_event(a, 1, ctx(last_kawa=kawa, target=0))
        assert event_to_action(ev) == a
    ev = action_to_event(38, 1, ctx(last_kawa="3m", akas=[True, False, False]))
    assert ev["consumed"] == ["4m", "5mr"]
    ev = action_to_event(41, 2, ctx(last_kawa="5s", akas=[False, False, True], target=1))
    assert ev["consumed"] == ["5sr", "5s"] and event_to_action(ev) == 41
    ev = action_to_event(41, 2, ctx(last_kawa="5sr", akas=[False, False, False], target=1))
    assert ev["consumed"] == ["5s", "5s"]


def test_discard_tsumogiri_and_misc():
    ev = action_to_event(INDEX["7p"], 0, ctx(last_tsumo="7p"))
    assert ev == {"type": "dahai", "actor": 0, "pai": "7p", "tsumogiri": True}
    assert action_to_event(37, 3, ctx())["type"] == "reach"
    assert action_to_event(45, 3, ctx()) == {"type": "none"}
    assert action_to_event(42, 0, ctx(ankan=["1m", "9m"], can_ankan=True)) is None     # needs kan select


def test_ranks_tiebreak():
    assert ranks([25000, 25000, 30000, 20000]) == [1, 2, 0, 3]
    assert ranks([25000] * 4) == [0, 1, 2, 3]
    assert placement_pt([30000, 20000, 25000, 25000], 0) == SACRED.pts[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
