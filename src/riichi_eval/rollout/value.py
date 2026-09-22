"""Valuing a rollout for one seat."""
from __future__ import annotations

from ..protocol import SACRED, placement_pt


def kyoku_delta(spec: dict, res, seat: int) -> int:
    """Points gained in the branch kyoku, riichi deposits included (sticks left on the table are not credited)."""
    return res.kyoku_scores[seat] - spec["scores"][seat]


def final_pt(res, seat: int, pts=SACRED.pts) -> float:
    return placement_pt(res.final_scores, seat, pts)


class GrpValuer:
    """Expected placement pt from Mortal's GRP (rank predictor over kyoku-start states).
    Bring your own checkpoint; GRPs trained on Tenhou logs are derived data and must not be distributed."""

    def __init__(self, path: str, pts=SACRED.pts):
        from .. import env
        env.setup()
        import torch
        from model import GRP  # Mortal/mortal/model.py
        st = torch.load(path, map_location="cpu", weights_only=True)["model"]
        hidden = st["rnn.weight_hh_l0"].shape[1]
        layers = sum(1 for k in st if k.startswith("rnn.weight_ih_l"))
        self.grp = GRP(hidden_size=hidden, num_layers=layers).eval()
        self.grp.load_state_dict(st)
        self.pts = torch.tensor(pts, dtype=torch.float64)
        self.torch = torch

    @staticmethod
    def feature(kyoku: int, honba: int, kyotaku: int, scores) -> list[float]:
        return [float(kyoku), float(honba), float(kyotaku)] + [s / 10000.0 for s in scores]

    def expected_pt(self, seqs: list[list[list[float]]], seats: list[int]) -> list[float]:
        torch = self.torch
        if not seqs:
            return []
        with torch.inference_mode():
            logits = self.grp([torch.tensor(q, dtype=torch.float64) for q in seqs])
            m = self.grp.calc_matrix(logits)                      # (N, player, rank)
            probs = m[torch.arange(len(seats)), torch.tensor(seats)]
            return (probs @ self.pts).tolist()
