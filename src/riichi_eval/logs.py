"""Reading libriichi arena logs (gzipped mjai JSON lines, one file per game)."""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field

NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")
ROTATION_SEAT = {"a": 0, "b": 1, "c": 2, "d": 3}   # 1v3 arena: rotation suffix = challenger's seat
BAKAZE = {"E": 0, "S": 1, "W": 2, "N": 3}


@dataclass
class Round:
    index: int              # 0-based kyoku index of the game (E1 = 0, S1 = 4, W1 = 8)
    honba: int
    kyotaku: int
    scores: list[int]       # at start_kyoku
    oya: int
    lines: list[str]        # raw JSON lines, start_kyoku .. end_kyoku inclusive
    events: list[dict] = field(repr=False)

    @property
    def n_tsumo(self) -> int:
        return sum(1 for e in self.events if e["type"] == "tsumo")


@dataclass
class GameLog:
    path: str
    seed: tuple[int, int]
    names: list[str]
    rounds: list[Round]

    def seat_of(self, name: str) -> list[int]:
        return [i for i, n in enumerate(self.names) if n == name]

    def final_scores(self) -> list[int]:
        return final_scores(self.rounds)


def read_game(path: str) -> GameLog:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return parse_game(lines, path)


def parse_game(lines: list[str], path: str = "") -> GameLog:
    """A GameLog from mjai JSON lines (start_game first), e.g. a rollout's log with meta stripped."""
    head = json.loads(lines[0])
    if head.get("type") != "start_game":
        raise ValueError(f"{path}: first event is not start_game")
    seed = tuple(head.get("seed") or ())
    rounds, cur, cur_ev = [], None, None
    for ln in lines[1:]:
        ev = json.loads(ln)
        t = ev["type"]
        if t == "start_kyoku":
            cur, cur_ev = [ln], [ev]
            continue
        if cur is None:
            continue
        cur.append(ln)
        cur_ev.append(ev)
        if t == "end_kyoku":
            sk = cur_ev[0]
            rounds.append(Round(index=BAKAZE[sk["bakaze"]] * 4 + int(sk["kyoku"]) - 1, honba=int(sk["honba"]),
                                kyotaku=int(sk["kyotaku"]), scores=list(sk["scores"]), oya=int(sk["oya"]),
                                lines=cur, events=cur_ev))
            cur = cur_ev = None
    return GameLog(path=path, seed=seed, names=list(head.get("names", [])), rounds=rounds)


def final_scores(rounds: list[Round]) -> list[int]:
    """Final scores. mjai's reach_accepted carries no deltas, so the 1000-point deposit is implicit; the last
    kyoku is settled by hand: deposits + deltas, and sticks left after a draw go to the top (Tenhou rule)."""
    if not rounds:
        return [25000] * 4
    last = rounds[-1]
    sc = list(last.scores)
    reaches, deltas, hora = [0] * 4, [0] * 4, False
    for ev in last.events:
        t = ev["type"]
        if t == "reach_accepted":
            reaches[int(ev["actor"])] += 1
        elif t in ("hora", "ryukyoku"):
            hora |= t == "hora"
            for i, v in enumerate(ev.get("deltas") or [0] * 4):
                deltas[i] += int(v)
    fin = [sc[i] - 1000 * reaches[i] + deltas[i] for i in range(4)]
    if not hora:
        left = last.kyotaku + sum(reaches)
        if left:
            top = max(range(4), key=lambda i: (fin[i], -i))
            fin[top] += 1000 * left
    return fin


def strip_meta(line: str) -> dict:
    ev = json.loads(line)
    ev.pop("meta", None)
    return ev
