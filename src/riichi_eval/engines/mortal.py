"""Mortal-format checkpoints (.pth with `config`, `mortal`, `current_dqn`) as libriichi engines."""
from __future__ import annotations

import os

from .. import env


def cap_gpu_mem(device: str) -> None:
    """Honour GPU_MEM_FRAC: evaluation shares cards with training and other users' jobs."""
    import torch
    frac = float(os.environ.get("GPU_MEM_FRAC", "0") or 0)
    if frac > 0 and str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(frac, torch.device(device).index or 0)


def load_mortal(path: str, device: str = "cuda:0", amp: str = "off", name: str | None = None,
                rule_based_agari_guard: bool = True):
    """amp: "off" = fp32 everywhere (the protocol setting; GPU fp32 matches CPU fp32 on ~97% of games),
    "on" = autocast on GPU, "auto" = on for GPU, off for CPU."""
    env.setup()
    import torch
    from engine import MortalEngine  # Mortal/mortal/engine.py
    from model import Brain, DQN     # Mortal/mortal/model.py

    state = torch.load(path, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    brain = Brain(version=version, conv_channels=cfg["resnet"]["conv_channels"],
                  num_blocks=cfg["resnet"]["num_blocks"]).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    enable_amp = (device != "cpu") if amp == "auto" else (amp == "on")
    return MortalEngine(brain, dqn, is_oracle=False, version=version, device=torch.device(device),
                        enable_amp=enable_amp, enable_rule_based_agari_guard=rule_based_agari_guard,
                        name=name or os.path.splitext(os.path.basename(path))[0])
