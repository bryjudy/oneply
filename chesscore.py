"""Shared core for the chess project: data decoding, tokenization, action space, model.

Data format facts (verified against the real ChessBench bytes on 2026-07-17):
- .bag file = concatenated records, then little-endian int64 end-offsets, last 8 bytes
  = index start (DeepMind "bagz" container, uncompressed for .bag).
- record = Apache-Beam TupleCoder(StrUtf8, StrUtf8, Float):
  varint-len fen | varint-len uci_move | 8-byte big-endian double win_prob.
- FEN tokenization → fixed 77 tokens over a 31-char vocab (DeepMind scheme, vendored).
- Action space: 1968 pseudo-legal UCI moves (queen+knight moves from every square
  + promotions), same construction as searchless_chess utils.
"""

import math
import struct

import numpy as np

# ---------------------------------------------------------------------------
# FEN tokenizer (DeepMind scheme, Apache-2.0, vendored from searchless_chess)
# ---------------------------------------------------------------------------

_CHARACTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'p', 'n', 'r', 'k', 'q', 'P', 'B', 'N', 'R', 'Q', 'K', 'w', '.',
]
_CHAR_INDEX = {c: i for i, c in enumerate(_CHARACTERS)}
_SPACES = frozenset('12345678')
SEQ_LEN_FEN = 77
FEN_VOCAB = len(_CHARACTERS)  # 31


def tokenize_fen(fen: str) -> np.ndarray:
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
    board = side + board.replace('/', '')
    idx = []
    for ch in board:
        if ch in _SPACES:
            idx.extend(int(ch) * [_CHAR_INDEX['.']])
        else:
            idx.append(_CHAR_INDEX[ch])
    if castling == '-':
        idx.extend(4 * [_CHAR_INDEX['.']])
    else:
        idx.extend(_CHAR_INDEX[c] for c in castling)
        idx.extend((4 - len(castling)) * [_CHAR_INDEX['.']])
    if en_passant == '-':
        idx.extend(2 * [_CHAR_INDEX['.']])
    else:
        idx.extend(_CHAR_INDEX[c] for c in en_passant)
    halfmoves += '.' * (3 - len(halfmoves))
    idx.extend(_CHAR_INDEX[c] for c in halfmoves)
    fullmoves += '.' * (3 - len(fullmoves))
    idx.extend(_CHAR_INDEX[c] for c in fullmoves)
    assert len(idx) == SEQ_LEN_FEN
    return np.asarray(idx, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Action space (1968 moves) — needs python-chess
# ---------------------------------------------------------------------------

def compute_action_tables():
    import chess

    all_moves = []
    board = chess.BaseBoard.empty()
    for square in range(64):
        next_squares = []
        board.set_piece_at(square, chess.Piece.from_symbol('Q'))
        next_squares += board.attacks(square)
        board.set_piece_at(square, chess.Piece.from_symbol('N'))
        next_squares += board.attacks(square)
        board.remove_piece_at(square)
        for nxt in next_squares:
            all_moves.append(chess.square_name(square) + chess.square_name(nxt))
    # Promotions — EXACT deterministic order from searchless_chess (no sets:
    # action ids must be identical across processes).
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for rank, next_rank in [('2', '1'), ('7', '8')]:
        for i, f in enumerate(files):
            all_moves += [f'{f}{rank}{f}{next_rank}{p}' for p in ['q', 'r', 'b', 'n']]
            if f > 'a':
                all_moves += [f'{f}{rank}{files[i - 1]}{next_rank}{p}'
                              for p in ['q', 'r', 'b', 'n']]
            if f < 'h':
                all_moves += [f'{f}{rank}{files[i + 1]}{next_rank}{p}'
                              for p in ['q', 'r', 'b', 'n']]
    move_to_action, action_to_move = {}, {}
    for action, move in enumerate(all_moves):
        assert move not in move_to_action
        move_to_action[move] = action
        action_to_move[action] = move
    assert len(move_to_action) == NUM_ACTIONS
    return move_to_action, action_to_move


NUM_ACTIONS = 1968  # asserted wherever compute_action_tables() is called


# ---------------------------------------------------------------------------
# Bag reading + record decoding (no external deps)
# ---------------------------------------------------------------------------

def _read_varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def iter_bag_records(path):
    """Yields raw record bytes from an uncompressed .bag file."""
    import mmap
    import os

    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    (index_start,) = struct.unpack('<Q', mm[-8:])
    num = (mm.size() - index_start) // 8
    prev = 0
    for k in range(num):
        (end,) = struct.unpack('<q', mm[index_start + 8 * k: index_start + 8 * k + 8])
        yield mm[prev:end]
        prev = end


def decode_action_value(rec: bytes):
    """(fen, uci_move, win_prob) from a ChessBench action_value record."""
    n, i = _read_varint(rec, 0)
    fen = rec[i:i + n].decode()
    i += n
    n, i = _read_varint(rec, i)
    move = rec[i:i + n].decode()
    i += n
    (wp,) = struct.unpack('>d', rec[i:i + 8])
    return fen, move, wp


# ---------------------------------------------------------------------------
# Win-prob buckets
# ---------------------------------------------------------------------------

NUM_BUCKETS = 128
BUCKET_EDGES = np.linspace(0.0, 1.0, NUM_BUCKETS + 1)[1:-1]
BUCKET_VALUES = (np.linspace(0, 1, NUM_BUCKETS + 1)[:-1]
                 + np.linspace(0, 1, NUM_BUCKETS + 1)[1:]) / 2


def win_prob_to_bucket(wp: np.ndarray) -> np.ndarray:
    return np.searchsorted(BUCKET_EDGES, wp, side='left').astype(np.uint8)


def centipawns_to_win_probability(cp: int) -> float:
    return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


# ---------------------------------------------------------------------------
# Model (PyTorch) — action-value transformer, searchless-chess recipe
# Input: 77 FEN tokens + 1 action token (ids offset by FEN_VOCAB).
# Output: distribution over NUM_BUCKETS win-prob buckets at the last position.
# ---------------------------------------------------------------------------

SEQ_LEN = SEQ_LEN_FEN + 1  # 78
VOCAB = FEN_VOCAB + NUM_ACTIONS  # 31 + 1968 = 1999


def build_model(d_model=256, n_layers=8, n_heads=8, d_ff=1024, dropout=0.0):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
            )

        def forward(self, x):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x

    class ChessAV(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, d_model)
            self.pos = nn.Embedding(SEQ_LEN, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, NUM_BUCKETS)

        def forward(self, ids):  # ids: (B, 78) long
            import torch as t
            x = self.tok(ids) + self.pos(t.arange(ids.shape[1], device=ids.device))
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x[:, -1]))  # (B, NUM_BUCKETS)

    return ChessAV()
