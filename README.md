# riichi-eval

Evaluation toolkit for riichi mahjong AIs, built on Mortal's `libriichi` arena.

**Status: in development (Phase 1).** Working today:

- **Counterfactual rollouts.** A patched libriichi (`rust/rollout.patch`) can restart any logged arena game at the
  start of a kyoku. The wall is the same one: it is derived from the game seed and (kyoku, honba). The rollout
  replays the logged reactions up to a branch point, plays a replacement action there, and continues with any
  engines to the end of the kyoku or of the game. Fidelity controls are in `tools/fidelity.py`: forced replay
  reproduces 100% of kyoku; free continuation reproduces ~99% of games at GPU fp32 and 100% at CPU fp32.
- **EV map** (`tools/ev_map.py`). Every decision where a reference policy would have acted differently is rolled
  out with the reference action. Telescoping splits the realized result into exact per-decision contributions,
  aggregated by decision type. Built-in check: all-reference games cancel to 0 over each seed's four rotations.

Planned: a unified runner with chunk-level resume and multi-machine sharding; single-pass mechanism readouts;
paired and sequential statistics; a variance-reduced estimator; HTML reports; external mjai bots and akochan.

## Setup

1. Build the patched libriichi in a *separate* copy of a Mortal tree:
   `rust/build.sh <Mortal dir> <build dir>` (see `rust/BASELINE.md` for the tree the patch targets).
2. Environment:
   - `RIICHI_EVAL_LIBRIICHI=<build dir>/pylib` — the patched `libriichi.so`, which shadows the stock one;
   - `MORTAL_DIR=<Mortal>/mortal` — `model.py` / `engine.py`, used to load Mortal-format checkpoints;
   - `GPU_MEM_FRAC=0.15` on shared GPUs.

## License

AGPL-3.0, because it links against libriichi (AGPL-3.0). No model weights, game logs or Tenhou-derived data
(including GRP checkpoints trained on Tenhou logs) are part of this repository.
