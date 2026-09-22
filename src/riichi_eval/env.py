"""Where libriichi and Mortal's Python modules come from.

    RIICHI_EVAL_LIBRIICHI  directory holding the patched libriichi.so (rust/build.sh output); it must
                           shadow the unpatched copy that ships inside Mortal/mortal
    MORTAL_DIR             Mortal/mortal (model.py, engine.py) — needed to load Mortal-format checkpoints
"""
from __future__ import annotations

import os
import sys

_done = False


def setup():
    """Put Mortal's python dir and the patched libriichi on sys.path (patched first). Returns libriichi."""
    global _done
    if not _done:
        mortal = os.environ.get("MORTAL_DIR")
        if mortal and mortal not in sys.path:
            sys.path.insert(0, mortal)
        lib = os.environ.get("RIICHI_EVAL_LIBRIICHI")
        if lib:
            if lib in sys.path:
                sys.path.remove(lib)
            sys.path.insert(0, lib)
        _done = True
    import libriichi
    return libriichi


def rollout_class():
    lib = setup()
    try:
        return lib.arena.Rollout
    except AttributeError as e:
        raise RuntimeError(
            f"{lib.__file__} has no arena.Rollout: build the patched libriichi (rust/build.sh) "
            "and set RIICHI_EVAL_LIBRIICHI to its pylib dir") from e
