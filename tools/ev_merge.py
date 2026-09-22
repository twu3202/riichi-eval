"""Combine EV-map shards (e.g. one per machine) and print the summary.

    python tools/ev_merge.py runs/evmap_srv runs/evmap_wsl [--json merged.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from riichi_eval.rollout import ev_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--json")
a = ap.parse_args()
points, games = ev_summary.load(a.dirs)
res = ev_summary.summarize(points, games)
if a.json:
    with open(a.json, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
