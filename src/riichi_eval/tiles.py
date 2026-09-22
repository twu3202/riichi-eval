"""mjai tile strings, Mortal action indices (0..36 = 34 kinds + three red fives) and tile arithmetic."""
from __future__ import annotations

TILES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"]
INDEX = {t: i for i, t in enumerate(TILES)}
_DEAKA = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}
_AKAIZE = {v: k for k, v in _DEAKA.items()}


def deaka(t: str) -> str:
    return _DEAKA.get(t, t)


def akaize(t: str) -> str:
    return _AKAIZE.get(t, t)


def is_aka(t: str) -> bool:
    return t in _DEAKA


def kind(t: str) -> int:
    """0..33, red fives folded onto their kind."""
    return INDEX[deaka(t)]


def _step(t: str, d: int) -> str:
    i = kind(t)
    k, n = divmod(i, 9)
    if k < 3:
        return TILES[k * 9 + (n + d) % 9]
    if n < 4:
        return TILES[27 + (n + d) % 4]
    return TILES[31 + (n - 4 + d) % 3]


def nxt(t: str) -> str:
    return _step(t, 1)


def prv(t: str) -> str:
    return _step(t, -1)


def dora_of(indicator: str) -> int:
    """Dora kind from an indicator."""
    return kind(nxt(indicator))
