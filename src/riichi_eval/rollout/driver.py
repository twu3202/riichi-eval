"""Run rollout specs through the patched libriichi, in batches, isolating failing specs."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .. import env


@dataclass
class RolloutResult:
    log: str | None                 # whole game log (start_game .. end_game) as mjai JSON lines
    final_scores: list[int] | None  # after game-end settlement (horizon=kyoku: after the single kyoku + settlement)
    kyoku_scores: list[int] | None  # right after the first (branch) kyoku, deposits included
    kyoku_kyotaku: int | None       # sticks left on the table after the first kyoku
    next_kyoku: tuple[int, int] | None = None   # (kyoku, honba) that follows; None if the game ended there
    error: str | None = None

    def kyoku_events(self, i: int = 0) -> list[dict]:
        """Events of the i-th kyoku of the rollout (meta stripped)."""
        out, k = [], -1
        for ln in self.log.splitlines():
            ev = json.loads(ln)
            ev.pop("meta", None)
            if ev["type"] == "start_kyoku":
                k += 1
            if k == i:
                out.append(ev)
                if ev["type"] == "end_kyoku":
                    break
        return out


def run(engines: list, specs: list[dict], *, batch: int = 512, progress: bool = False) -> list[RolloutResult]:
    Rollout = env.rollout_class()
    if any(s.get("commit") for s in specs) and not getattr(Rollout, "SUPPORTS_COMMIT", False):
        raise RuntimeError("this libriichi build predates `commit` and would silently ignore it; rebuild")
    if any(s.get("reshuffle") is not None for s in specs) and not getattr(Rollout, "SUPPORTS_RESHUFFLE", False):
        raise RuntimeError("this libriichi build predates `reshuffle` and would silently ignore it; rebuild")
    if any(s.get("wall") is not None for s in specs) and not getattr(Rollout, "SUPPORTS_WALL", False):
        raise RuntimeError("this libriichi build predates `wall`; rebuild")
    ro = Rollout(disable_progress_bar=not progress)
    out: list[RolloutResult | None] = [None] * len(specs)

    def go(idxs):
        try:
            res = ro.run(engines, [specs[i] for i in idxs])
        except Exception as e:  # noqa: BLE001 — isolate the failing spec by bisection
            if len(idxs) == 1:
                out[idxs[0]] = RolloutResult(None, None, None, None, None, error=f"{type(e).__name__}: {e}")
                return
            h = len(idxs) // 2
            go(idxs[:h])
            go(idxs[h:])
            return
        for i, (log, fin, ks, kk, nk) in zip(idxs, res):
            out[i] = RolloutResult(log, list(fin), list(ks), int(kk), tuple(nk) if nk is not None else None)

    for s in range(0, len(specs), batch):
        go(list(range(s, min(s + batch, len(specs)))))
    return out
