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

## Variance reduction (luck control variates)

Full-information 1v3 logs let every chance event be priced exactly: the four dealt hands are exchangeable across
seats, every draw is uniform over the tiles not yet revealed, and so is every ura-dora indicator. Each correction
term is `h(outcome) − E[h | history]`, so the corrected mean `mean(Y − β·C)` is unbiased for any weights β.

    python tools/vr_extract.py --log-dir DIR [--log-dir ...] --out run.npz      # ~1,000 games/s on 8 cores
    python tools/vr_fit.py run.npz --beta data/vr_beta_a2.npz [--pair other.npz]  # fixed weights, optional pairing

`data/vr_beta_a2.npz` holds 27 weights fitted once on 1.07M duplicate games (A², v4b vs v4): dealt-hand terms,
dora/red-five draws, shanten-advance draws by role × shanten × riichi-on-table, ura-dora by role.
Measured: variance ratio 0.856 in-sample (10-fold cross-fitted), 0.846 when transferred unchanged to another
model (tpt 408k, 72k games), 0.856 on a paired snapshot comparison — about 1.17× the games for free.
All terms are checked for zero mean on every run (`vr_fit.py` excludes any that fail).
Finer lookup tables (288 cells) add nothing; larger gains need a real value function over hypothetical draws,
which costs more GPU time than simply playing more games.
