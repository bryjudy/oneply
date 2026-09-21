# Build log

Condensed from the working notes, July 17-25, 2026. Infrastructure details (hosts, keys, credit programs) removed.

## Framing

The idea started as "build a model that outperforms Stockfish". The honest lay of the land: Stockfish is alpha-beta search plus a small, fast NNUE evaluation, and the only thing that has beaten it is AlphaZero/Leela-style MCTS with a large network and enormous compute. DeepMind's 2024 ChessBench paper showed the opposite extreme: a 270M transformer that plays ~2895 Lichess blitz Elo with no search at all. The underexplored middle is a strong learned evaluation with a small amount of search, and the question I could actually afford to ask on a single GPU was: on the (evaluation quality x nodes searched) curve, where does a learned action-value model with ~180 evaluations per move cross full-strength Stockfish?

## Timeline

- **07-17** Pipeline stood up on serverless GPU. Bagz reader and Apache-Beam record decoder verified byte-for-byte against real ChessBench shards; FEN tokenizer verified equal to DeepMind's; 1968-move action table built in DeepMind's deterministic order; PyTorch action-value transformer (6.9M default). Stockfish 17.1 match harness (~1.1M NPS single thread on the worker). 48 shards (413M examples) pre-tokenized to memmap .npy. The serverless provider's auth token died minutes after the first training run was accepted and stayed dead for four days.
- **07-18** `search.py`: batched pruned minimax on top of the action-value net. Root does an exhaustive mate-in-1 scan so beam pruning can never drop an immediate mate (bug found and fixed via a random-net test).
- **07-19/20** Offline fallback: 6 shards (36M examples) prepared locally. Training on Apple MPS was broken for this model (AdamW foreach deadlock, then 10-40x slower than CPU even with SDPA), so the **nano** model (1.08M params) trained on CPU for 10 hours: 41.5M examples, eval loss 3.34. First end-to-end numbers: puzzles 21.5% -> 25.25% with depth-2 (n=400); vs Stockfish 18 skill 0: 0.5/10 -> 1.5/10. Search helps even at nano scale.
- **07-20** Moved to a rented A100-80GB VM. 48 shards prepared on-instance in 15 minutes. **av9m** (6.9M) training at 15.2k examples/s, batch 4096, bf16. Second VM prepared 130 shards (906M unique examples) for **av55m** (d512 L16, 51.6M params). av55m OOM'd at batch 4096 (78-token sequences x 55M params), relaunched at batch 2048 with a 1.08M-step cosine schedule: 3.5k examples/s, then 4.2k with torch.compile. A restart of av9m with optimizations produced repeated silent GPU stalls at ~step 11k with no error; reverting the code did not help, so the VM was recreated (checkpoint rescued first). Lesson: do not SIGKILL CUDA processes on a virtualized GPU; the trainer grew `--sync_loader`, `--no_compile`, `--no_fused` escape hatches, producer-death logging and a queue timeout.
- **07-21** av9m @ step 55k (225M examples): puzzles 53.4% d1 / 67.2% d2 (n=1000). vs Stockfish 17.1 skill 0 no search 9.0/10; skill 2 no search 5.0/10; skill 3 with depth-2 8.5/10. One ply of search was worth more than one full skill level. Search gain grew from +3.75 pts at nano to +13.8 pts here.
- **07-21 pm** av9m complete: 1.154B examples (2.8 epochs), 24 h, ~$20. Eval loss 2.353, win-prob MAE 0.024. Puzzles 76.9% d1 / 81.5% d2. vs skill 3 no search 9.0/10; vs skill 5 with depth-2 8.5/10 undefeated. Search gain fell to +4.6: the first hint the gain is an inverted U in evaluation quality.
- **07-22** av55m mid-run @ 491M examples: puzzles 79.8 / 87.8, vs skill 5 with depth-2 9.5/10. Beats fully-trained av9m on every metric with less than half its data. Search gain +8.0.
- **07-22 pm** First **node ladder** vs full-strength Stockfish 17.1 (av55m @ 696M examples, depth-2 beam-8): 9.5/10 @ 100 nodes, 9.0 @ 400, 5.5 @ 1,600, 2.5 @ 3,200, 1.0 @ 6,400. A clean sigmoid with the crossover near 1,700-1,800 nodes. This became the benchmark axis.
- **07-23** @ 907M examples: 6.0/10 @ 3,200. Crossover moved past 3,200.
- **07-24** @ 1.54B: 3.5/10 @ 6,400. Crossover ~4,000-5,000 and climbing.
- **07-25** av55m complete: 2.07B examples (2.3 epochs), 120 h, ~$100. Eval loss 2.109, MAE 0.018. Puzzles 86.9% d1 / 92.0% d2. Final ladder: 6.0 @ 3,200, 1.0 @ 6,400, 3.0 @ 12,800. The non-monotone rungs are 10-game noise, so the final crossover is quoted as 3,500-4,500 nodes. Lesson for the next tier: 30+ games per rung. All instances torn down; total spend about $170.

## What I would do next

1. 136-270M model on the full 15B-example ChessBench (projected crossover 10-20k nodes at this scaling). Needs real GPU hours.
2. 30+ games per ladder rung, opening book, and both colors, so the crossover is a number instead of a range.
3. Depth-3 and wider beams: the search is batched and cheap on GPU, and the mid-quality checkpoints suggest the eval is still leaving a lot on the table.
4. Distill the search back into the net (search-improved targets), the AlphaZero loop but starting from Stockfish's knowledge.
