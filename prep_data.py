"""Download + tokenize N ChessBench action-value shards in parallel into ./data.

Each shard is ~8.6M (fen, move, win_prob) records, ~600 MB of .bag. Do this on a
machine with datacenter bandwidth: 48 shards = 413M examples, 130 shards = ~1.1B.
  python prep_data.py 48
"""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chesscore import (compute_action_tables, decode_action_value,
                       iter_bag_records, tokenize_fen)

GCS = "https://storage.googleapis.com/searchless_chess/data"
OUT = HERE / "data"


def prep(args):
    url, prefix = args
    if (OUT / f"{prefix}_tokens.npy").exists():
        return f"{prefix}: cached"
    tmp = OUT / f"{prefix}.bag"
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 23):
                f.write(chunk)
    move_to_action, _ = compute_action_tables()
    toks, acts, wps = [], [], []
    for rec in iter_bag_records(str(tmp)):
        fen, move, wp = decode_action_value(rec)
        toks.append(tokenize_fen(fen))
        acts.append(move_to_action[move])
        wps.append(wp)
    np.save(OUT / f"{prefix}_tokens.npy", np.stack(toks))
    np.save(OUT / f"{prefix}_actions.npy", np.asarray(acts, dtype=np.int16))
    np.save(OUT / f"{prefix}_winprob.npy", np.asarray(wps, dtype=np.float32))
    tmp.unlink()
    return f"{prefix}: {len(toks)} examples"


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    OUT.mkdir(exist_ok=True)
    (OUT / "puzzles.csv").write_bytes(
        requests.get(f"{GCS}/puzzles.csv", timeout=300).content)
    jobs = [(f"{GCS}/test/action_value_data.bag", "eval")]
    jobs += [(f"{GCS}/train/action_value-{i:05d}-of-02148_data.bag",
              f"train_{i:05d}") for i in range(n_shards)]
    with mp.Pool(min(12, mp.cpu_count())) as pool:
        for msg in pool.imap_unordered(prep, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
