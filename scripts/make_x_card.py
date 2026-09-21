"""Dark 16:9 card for social posts. uv run --with matplotlib --with numpy python scripts/make_x_card.py"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "results.json").read_text())

# dark-mode palette steps
SURF, INK, INK2, INK3, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#8a8983", "#2e2e2c"
BLUE, ORANGE = "#3987e5", "#d95926"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

fig = plt.figure(figsize=(16, 9))
fig.text(0.05, 0.90, "oneply", fontsize=20, color=INK3, va="top", fontweight="bold")
fig.text(0.05, 0.83, "$100 of GPU time. 52M parameters. One ply of search.", fontsize=34, color=INK, va="top", fontweight="bold")
fig.text(0.05, 0.735, "A transformer trained on Stockfish's own evaluations, given ~180 network evaluations per move,\n"
         "plays even with full-strength Stockfish 17.1 searching ~4,000 nodes per move.",
         fontsize=15, color=INK2, va="top", linespacing=1.5)

tiles = [("parameters", "51.6M"), ("Stockfish-labeled positions", "2.07B"), ("A100 hours", "120"),
         ("Lichess puzzles, no search", "86.9%"), ("with one ply", "92.0%")]
for i, (lab, val) in enumerate(tiles):
    x = 0.05 + i * 0.185
    fig.text(x, 0.60, val, fontsize=30, color=INK, va="top", fontweight="bold")
    fig.text(x, 0.535, lab, fontsize=12, color=INK2, va="top")

ax = fig.add_axes([0.07, 0.11, 0.88, 0.36])
rows = [r for r in RES["node_ladder"]["rows"] if r["examples_seen"] == 696000000]
fin = [r for r in RES["node_ladder"]["rows"] if r["examples_seen"] == 2069438464]
ax.plot([r["sf_nodes"] for r in rows], [r["score"] for r in rows], color=BLUE, lw=2.5, zorder=2)
for r in rows:
    ax.plot(r["sf_nodes"], r["score"], "o", ms=13, color=SURF, zorder=3)
    ax.plot(r["sf_nodes"], r["score"], "o", ms=10, color=BLUE, zorder=4)
for r in fin:
    ax.plot(r["sf_nodes"], r["score"], "o", ms=13, color=SURF, zorder=3)
    ax.plot(r["sf_nodes"], r["score"], "o", ms=10, color=ORANGE, zorder=4)
ax.axhline(5, color=INK3, lw=1, zorder=1)
ax.text(74, 5.3, "even", color=INK2, fontsize=11)
ax.set_xscale("log")
ax.set_xticks([100, 400, 1600, 3200, 6400, 12800], ["100", "400", "1,600", "3,200", "6,400", "12,800"])
ax.set_xlim(70, 20000); ax.set_ylim(0, 10.5)
ax.tick_params(labelsize=12)
ax.set_xlabel("full-strength Stockfish 17.1, nodes per move", fontsize=13)
ax.set_ylabel("model score / 10 games", fontsize=13)
ax.grid(axis="x", visible=False)
ax.plot([], [], "o-", color=BLUE, label="model at 34% of training")
ax.plot([], [], "o", color=ORANGE, label="final checkpoint")
ax.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=INK2)
fig.text(0.05, 0.03, "github.com/bryjudy/oneply   ·   code, weights and every game result, MIT", fontsize=12, color=INK3)
fig.savefig(ROOT / "figures" / "x_card.png", dpi=150)
print("wrote figures/x_card.png")
