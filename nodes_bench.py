"""Node ladder vs FULL-STRENGTH Stockfish: one match per rung, in one go.

  python nodes_bench.py --ckpt release/av55m.pt --nodes 100 400 1600 3200 6400 --games 30 --widths 8

Same as `evaluate.py match --sf_nodes N` for each N; prints one JSON line per rung.
Use 30+ games per rung: the July 2026 runs used 10 and the top rungs were noisy.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", default=[100, 400, 1600, 3200, 6400])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--widths", default="8")
    ap.add_argument("--sf", default=None)
    a = ap.parse_args()
    for n in a.nodes:
        cmd = [sys.executable, str(HERE / "evaluate.py"), "match", "--ckpt", a.ckpt,
               "--games", str(a.games), "--widths", a.widths, "--sf_nodes", str(n)]
        if a.sf:
            cmd += ["--sf", a.sf]
        print(f"== Stockfish @ {n} nodes/move", flush=True)
        subprocess.run(cmd, check=True)
