"""Puzzle accuracy and skill-capped matches for a checkpoint.

  python evaluate.py puzzles --ckpt release/av55m.pt --n 1000 --widths 8
  python evaluate.py match   --ckpt release/av55m.pt --games 10 --sf_skill 5 --widths 8

--widths "" = depth-1 argmax (no search); "8" = depth-2 beam-8; "8,6" = depth-3.
Stockfish binary: --sf, or $STOCKFISH, or `stockfish` on PATH. Puzzles come from
data/puzzles.csv (prep_data.py downloads it; or fetch it from the ChessBench bucket).
"""

import argparse
import io
import os
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chesscore import build_model
from search import AVSearcher


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(ckpt, dev):
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    model = build_model(**sd["config"]).to(dev)
    model.load_state_dict(sd["model"])
    model.eval()
    return model, sd["step"]


def parse_widths(s):
    return tuple(int(x) for x in s.split(",") if x.strip()) if s else ()


def stockfish_path(arg):
    p = arg or os.environ.get("STOCKFISH") or shutil.which("stockfish")
    if not p:
        sys.exit("no Stockfish found: pass --sf, set $STOCKFISH, or put `stockfish` on PATH")
    return p


def puzzles(a):
    import chess
    import chess.pgn
    import pandas as pd

    dev = device()
    model, step = load(a.ckpt, dev)
    s = AVSearcher(model, dev)
    widths = parse_widths(a.widths)
    df = pd.read_csv(a.puzzles, nrows=a.n)
    ok = 0
    for _, row in df.iterrows():
        game = chess.pgn.read_game(io.StringIO(row["PGN"]))
        board = game.end().board()
        solved = True
        for i, mv in enumerate(row["Moves"].split(" ")):
            if i % 2 == 1:
                pred = s.play(board, widths=widths).uci()
                if mv != pred:
                    board.push(chess.Move.from_uci(pred))
                    solved = board.is_checkmate()  # an alternative mate counts (DeepMind convention)
                    break
            board.push(chess.Move.from_uci(mv))
        ok += solved
    print({"ckpt": a.ckpt, "step": step, "widths": list(widths), "n": a.n,
           "accuracy": round(ok / a.n, 4), "net_evals": s.eval_calls})


def match(a):
    import chess
    import chess.engine

    dev = device()
    model, step = load(a.ckpt, dev)
    s = AVSearcher(model, dev)
    widths = parse_widths(a.widths)
    sf = chess.engine.SimpleEngine.popen_uci(stockfish_path(a.sf))
    if a.sf_skill is not None:
        sf.configure({"Skill Level": a.sf_skill})
    limit = (chess.engine.Limit(nodes=a.sf_nodes) if a.sf_nodes
             else chess.engine.Limit(time=a.sf_movetime))
    score = 0.0
    results = []
    for g in range(a.games):
        board = chess.Board()
        mw = g % 2 == 0
        while not board.is_game_over(claim_draw=True):
            if (board.turn == chess.WHITE) == mw:
                board.push(s.play(board, widths=widths))
            else:
                board.push(sf.play(board, limit).move)
        res = board.result(claim_draw=True)
        pts = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
        score += pts if mw else 1 - pts
        results.append(res if mw else {"1-0": "0-1", "0-1": "1-0", "1/2-1/2": "1/2-1/2"}[res])
        print(f"game {g + 1}/{a.games}: model score so far {score}", flush=True)
    sf.quit()
    print({"ckpt": a.ckpt, "step": step, "widths": list(widths), "games": a.games,
           "sf_skill": a.sf_skill, "sf_nodes": a.sf_nodes, "model_score": score,
           "results_model_pov": results})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("puzzles")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--widths", default="")
    p.add_argument("--puzzles", default=str(HERE / "data" / "puzzles.csv"))
    p.set_defaults(fn=puzzles)
    m = sub.add_parser("match")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--games", type=int, default=10)
    m.add_argument("--widths", default="")
    m.add_argument("--sf", default=None)
    m.add_argument("--sf_skill", type=int, default=None, help="UCI Skill Level 0-20; omit for full strength")
    m.add_argument("--sf_nodes", type=int, default=None, help="node limit per move (full-strength ladder)")
    m.add_argument("--sf_movetime", type=float, default=0.05, help="seconds per move if no node limit")
    m.set_defaults(fn=match)
    a = ap.parse_args()
    a.fn(a)
