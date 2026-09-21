"""Play the model from the terminal, or ask it for a move in a position.

  python play.py --ckpt release/av55m.pt                 # you are White, model is Black
  python play.py --ckpt release/av55m.pt --black          # you are Black
  python play.py --ckpt release/av55m.pt --fen "<fen>"    # one move + its win-prob table
  --widths "" for no search, "8" (default) for depth-2 beam-8, "8,6" for depth-3.
"""

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--widths", default="8")
    ap.add_argument("--fen", default=None)
    ap.add_argument("--black", action="store_true", help="you play Black")
    a = ap.parse_args()
    widths = tuple(int(x) for x in a.widths.split(",") if x.strip()) if a.widths else ()

    dev = device()
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = build_model(**sd["config"]).to(dev).eval()
    model.load_state_dict(sd["model"])
    s = AVSearcher(model, dev)
    print(f"loaded {a.ckpt}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"step {sd['step']}, device {dev}, widths {list(widths)}")

    if a.fen:
        board = chess.Board(a.fen)
        moves = list(board.legal_moves)
        scores = s._score_moves(board, moves)
        order = np.argsort(scores)[::-1]
        print("\nnet's win probability (side to move) after each move, no search:")
        for i in order[:10]:
            print(f"  {board.san(moves[i]):8s} {scores[i]:.3f}")
        best = s.play(board, widths=widths)
        print(f"\nmove with search widths {list(widths)}: {board.san(best)}   ({s.eval_calls} net evals)")
        return

    board = chess.Board()
    human_white = not a.black
    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board) + "\n")
        if board.turn == chess.WHITE and human_white or board.turn == chess.BLACK and not human_white:
            while True:
                txt = input("your move (SAN or UCI, q to quit): ").strip()
                if txt == "q":
                    return
                try:
                    mv = board.parse_san(txt)
                except ValueError:
                    try:
                        mv = chess.Move.from_uci(txt)
                        if mv not in board.legal_moves:
                            raise ValueError
                    except ValueError:
                        print("illegal move")
                        continue
                break
            board.push(mv)
        else:
            s.eval_calls = 0
            mv = s.play(board, widths=widths)
            print(f"model plays {board.san(mv)}  ({s.eval_calls} net evals)")
            board.push(mv)
    print("\n" + str(board))
    print("result:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
