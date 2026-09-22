"""Evaluation protocol presets and placement arithmetic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    seed_key: int
    seed_start: int
    pts: tuple[float, float, float, float]
    champion_name: str


# TanyaoDojo's fixed protocol: duplicate 1v3, every seed played four times with the challenger rotating seats.
SACRED = Protocol(seed_key=20260711, seed_start=10000, pts=(90.0, 45.0, 0.0, -135.0), champion_name="mortal-v4")


def ranks(scores) -> list[int]:
    """Rank per seat (0 = top). Ties go to the lower seat index, as in libriichi's Rankings."""
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    out = [0] * 4
    for r, s in enumerate(order):
        out[s] = r
    return out


def placement_pt(scores, seat: int, pts=SACRED.pts) -> float:
    return float(pts[ranks(scores)[seat]])
