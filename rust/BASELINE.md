# Patch baseline

`rollout.patch` applies to the Mortal source tree used for the `libriichi.so` built on 2026-07-14
(trx50 `~/Projects/better_mortal/Mortal`; that tree is not a git checkout and already carries the local
obs-v5 changes).

Baseline fingerprint (sha256 over the sorted per-file sha256 of the tree, excluding `target/`, `*.so`,
`log-viewer/`, `docs/`): `68bcbd7454649c9d`.

The patch is additive: `BatchGame::run` behaves exactly as before; `run_with_init`, `ForcedPrefixAgent`
and `libriichi.arena.Rollout` are new.
