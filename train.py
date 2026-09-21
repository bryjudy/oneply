"""Train the action-value transformer on pre-tokenized ChessBench shards in ./data.

Single GPU, bf16, resumable from ckpt/<run>/latest.pt, JSONL log, time-capped.
  python train.py --run av9m  --d_model 256 --n_layers 8  --batch 4096 --hours 24
  python train.py --run av55m --d_model 512 --n_layers 16 --batch 2048 --hours 120 --cosine_steps 1080000
"""

import argparse
import glob
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chesscore import NUM_BUCKETS, build_model, win_prob_to_bucket

DATA = HERE / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="av9m")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--no_fused", action="store_true")
    ap.add_argument("--sync_loader", action="store_true",
                    help="bypass prefetch thread (original known-good path)")
    ap.add_argument("--cosine_steps", type=int, default=0,
                    help="if >0, cosine-decay LR to 10% over this many steps")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = HERE / "ckpt" / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    shard_files = sorted(glob.glob(str(DATA / "train_*_tokens.npy")))
    shards = []
    for tf in shard_files:
        base = tf.replace("_tokens.npy", "")
        shards.append((np.load(tf, mmap_mode="r"),
                       np.load(base + "_actions.npy", mmap_mode="r"),
                       np.load(base + "_winprob.npy", mmap_mode="r")))
    total = sum(t.shape[0] for t, _, _ in shards)

    ev_t = np.load(DATA / "eval_tokens.npy", mmap_mode="r")[:100000]
    ev_a = np.load(DATA / "eval_actions.npy", mmap_mode="r")[:100000]
    ev_w = np.load(DATA / "eval_winprob.npy", mmap_mode="r")[:100000]
    ev_ids = torch.from_numpy(np.concatenate(
        [np.asarray(ev_t), (np.asarray(ev_a).astype(np.int64) + 31)[:, None]],
        axis=1)).long().to(dev)
    ev_tgt = torch.from_numpy(
        win_prob_to_bucket(np.asarray(ev_w)).astype(np.int64)).to(dev)

    cfg = {"d_model": args.d_model, "n_layers": args.n_layers,
           "n_heads": max(4, args.d_model // 32), "d_ff": 4 * args.d_model}
    model = build_model(**cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if args.no_fused:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4, fused=True)
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
    step = 0
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        sd = torch.load(latest, map_location=dev)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
    if args.no_compile:
        model_c = model
    else:
        try:
            model_c = torch.compile(model)
        except Exception:
            model_c = model

    def save(tag="latest"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "config": cfg}, ckpt_dir / f"{tag}.pt")

    def evaluate():
        model.eval()
        losses, maes = [], []
        bv = torch.linspace(1 / (2 * NUM_BUCKETS), 1 - 1 / (2 * NUM_BUCKETS),
                            NUM_BUCKETS, device=dev)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for b in range(0, ev_ids.shape[0], 16384):
                logits = model_c(ev_ids[b:b + 16384])
                losses.append(F.cross_entropy(
                    logits.float(), ev_tgt[b:b + 16384]).item())
                pred = (logits.float().softmax(-1) * bv).sum(-1)
                maes.append((pred - bv[ev_tgt[b:b + 16384]]).abs().mean().item())
        model.train()
        return float(np.mean(losses)), float(np.mean(maes))

    CHUNK = 1 << 21
    rng = np.random.default_rng(step + 1)

    def batches():
        while True:
            for si in rng.permutation(len(shards)):
                t, a, w = shards[si]
                n = t.shape[0]
                for start in range(0, max(n - CHUNK + 1, 1), CHUNK):
                    m = min(CHUNK, n - start)
                    if m < args.batch:
                        continue
                    tt = np.asarray(t[start:start + m])
                    aa = np.asarray(a[start:start + m]).astype(np.int64)
                    ww = np.asarray(w[start:start + m])
                    perm = rng.permutation(m)
                    ids = np.concatenate([tt, (aa + 31)[:, None]], axis=1)[perm]
                    tgt = win_prob_to_bucket(ww).astype(np.int64)[perm]
                    for b in range(0, m - args.batch + 1, args.batch):
                        yield ids[b:b + args.batch], tgt[b:b + args.batch]

    q = queue.Queue(maxsize=8)

    def producer():
        try:
            for item in batches():
                q.put(item)
        except BaseException as e:
            with open(log_path, "a") as f:
                f.write(json.dumps({"event": "producer_died",
                                    "error": repr(e)}) + "\n")
            raise

    if not args.sync_loader:
        threading.Thread(target=producer, daemon=True).start()

    def prefetched():
        if args.sync_loader:
            yield from batches()
            return
        while True:
            yield q.get(timeout=600)  # hang -> loud crash, not silent stall

    t0 = time.time()
    t_log = time.time()
    seen0 = step * args.batch
    running = []
    logf = open(log_path, "a")
    logf.write(json.dumps({"event": "start", "params": n_params,
                           "train_examples": total, "resume_step": step,
                           "batch": args.batch}) + "\n")
    logf.flush()
    for ids_np, tgt_np in prefetched():
        if time.time() - t0 > args.hours * 3600:
            break
        ids = torch.from_numpy(ids_np).long().to(dev, non_blocking=True)
        tgt = torch.from_numpy(tgt_np).to(dev, non_blocking=True)
        warm = min(1.0, (step + 1) / 1000)
        if args.cosine_steps > 0:
            import math
            prog = min(1.0, step / args.cosine_steps)
            sched = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        else:
            sched = 1.0
        for g in opt.param_groups:
            g["lr"] = args.lr * warm * sched
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model_c(ids), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        step += 1
        if step % 500 == 0:
            dt = time.time() - t_log
            seen = step * args.batch
            rec = {"step": step, "loss": round(float(np.mean(running)), 4),
                   "examples_seen": seen,
                   "examples_per_sec": int((seen - seen0) / dt),
                   "hours": round((time.time() - t0) / 3600, 3)}
            running = []
            t_log = time.time()
            seen0 = seen
            if step % 5000 == 0:
                ev_loss, ev_mae = evaluate()
                rec["eval_loss"] = round(ev_loss, 4)
                rec["eval_wp_mae"] = round(ev_mae, 4)
                save()
                if step % 100000 == 0:
                    save(f"step{step}")
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
    save()
    ev_loss, ev_mae = evaluate()
    logf.write(json.dumps({"event": "end", "step": step,
                           "eval_loss": round(ev_loss, 4),
                           "eval_wp_mae": round(ev_mae, 4)}) + "\n")


if __name__ == "__main__":
    main()
