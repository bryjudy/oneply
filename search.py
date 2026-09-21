"""Shallow search on top of the action-value net — the project's core experiment.

DeepMind showed a big transformer plays ~grandmaster chess with NO search;
Stockfish is a tiny eval with ENORMOUS search. The unexplored frontier is the
middle: how much does a little search amplify a learned eval? This module
implements a GPU-friendly batched pruned minimax:

  depth 1: score all legal moves in one batch (what beam_eval does today).
  depth 2: keep top-K of my moves, batch-score ALL opponent replies to each,
           my value for a move = worst-case (opponent's best reply).
  depth 3+: recurse with per-ply beam widths, e.g. widths=(8, 6) for depth 3.

Values are win probs for the side to move at that node; propagate with v = 1 - v
across plies. Terminal nodes: checkmate = 0 for the mated side, stalemate /
insufficient material / repetition = 0.5.

Eval-call budget per move ≈ 1 + K1 + K1*K2 ... — with batching this is a
handful of GPU forwards, so even depth 3 is fast on the 4090.
"""

import numpy as np


class AVSearcher:
    def __init__(self, model, device, batch_cap=4096):
        import torch

        from chesscore import NUM_BUCKETS, compute_action_tables, tokenize_fen

        self.torch = torch
        self.model = model
        self.dev = device
        self.batch_cap = batch_cap
        self.move_to_action, _ = compute_action_tables()
        self.tokenize_fen = tokenize_fen
        self.bv = torch.linspace(1 / (2 * NUM_BUCKETS), 1 - 1 / (2 * NUM_BUCKETS),
                                 NUM_BUCKETS, device=device)
        self.eval_calls = 0

    def _score_moves(self, board, moves):
        """Win prob (side to move) after each move, one batched forward."""
        t = self.torch
        fen_toks = self.tokenize_fen(board.fen())
        ids = np.empty((len(moves), 78), dtype=np.int64)
        for k, mv in enumerate(moves):
            ids[k, :77] = fen_toks
            ids[k, 77] = self.move_to_action[mv.uci()] + 31
        out = np.empty(len(moves), dtype=np.float64)
        with t.no_grad():
            for b in range(0, len(moves), self.batch_cap):
                chunk = t.from_numpy(ids[b:b + self.batch_cap]).to(self.dev)
                if str(self.dev).startswith("cuda"):
                    with t.autocast("cuda", t.bfloat16):
                        logits = self.model(chunk)
                else:
                    logits = self.model(chunk)
                wp = (logits.float().softmax(-1) * self.bv).sum(-1)
                out[b:b + self.batch_cap] = wp.cpu().numpy()
        self.eval_calls += len(moves)
        return out

    def _terminal_value(self, board):
        """Value for the side to move, or None if not terminal."""
        if board.is_checkmate():
            return 0.0
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.can_claim_draw()):
            return 0.5
        return None

    def _value(self, board, widths):
        """Win prob for side to move, searching len(widths) more plies."""
        tv = self._terminal_value(board)
        if tv is not None:
            return tv
        moves = list(board.legal_moves)
        scores = self._score_moves(board, moves)
        if not widths:
            return float(scores.max())
        k = min(widths[0], len(moves))
        top = np.argsort(scores)[::-1][:k]
        best = 0.0
        for i in top:
            board.push(moves[i])
            # opponent's value from their perspective -> ours = 1 - theirs
            v = 1.0 - self._value(board, widths[1:])
            board.pop()
            best = max(best, v)
        return best

    def play(self, board, widths=(8,)):
        """Best move with pruned minimax. widths=() -> plain depth-1 argmax."""
        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]
        # exhaustive mate-in-1 scan (no evals needed): beam pruning below must
        # never lose an immediate mate to bad move-ordering
        for mv in moves:
            board.push(mv)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return mv
        scores = self._score_moves(board, moves)
        if not widths:
            return moves[int(scores.argmax())]
        k = min(widths[0], len(moves))
        top = np.argsort(scores)[::-1][:k]
        best_v, best_m = -1.0, moves[int(scores.argmax())]
        for i in top:
            board.push(moves[i])
            tv = self._terminal_value(board)
            if tv is not None:
                v = 1.0 - tv
            else:
                v = 1.0 - self._value(board, widths[1:])
            board.pop()
            if v > best_v:
                best_v, best_m = v, moves[i]
        return best_m
