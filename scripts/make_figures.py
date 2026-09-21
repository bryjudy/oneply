"""Render the figures in figures/ from results/results.json and results/train_logs/.

  uv run --with matplotlib --with numpy python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "results.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# palette (light surface): categorical slots, text tokens, recessive grid
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e8e7e3"
MODEL_COLOR = {"nano": AQUA, "av9m": ORANGE, "av55m": BLUE}
MODEL_LABEL = {"nano": "nano (1.1M)", "av9m": "av9m (6.9M)", "av55m": "av55m (52M)"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def style(ax):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", visible=False)


def title(fig, t, sub, width=118):
    import textwrap
    fig.text(0.02, 0.965, t, fontsize=15, fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.915, "\n".join(textwrap.wrap(sub, width)), fontsize=10.5, color=INK2, ha="left", va="top", linespacing=1.4)


def dot(ax, x, y, c, s=9):
    ax.plot(x, y, "o", ms=s + 2, color=SURFACE, zorder=3)   # 2px surface ring
    ax.plot(x, y, "o", ms=s, color=c, zorder=4)


# ---------------------------------------------------------------- fig 1
def fig_search_amplification():
    rows = RES["puzzles"]
    ref = RES["deepmind_reference_puzzles"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), height_ratios=[3, 1.35],
                                  gridspec_kw={"hspace": 0.55})
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.09, right=0.97)
    title(fig, "One ply of search on a learned evaluation",
          "Lichess puzzle accuracy for each checkpoint, no search (depth-1 argmax) vs depth-2 beam-8 (~180 net evals/move). "
          "Gray lines: DeepMind's no-search transformers (Ruoss et al. 2024), trained on 7-13x more data.")
    labels = [f"{r['model']}\n{r['checkpoint']}" for r in rows]
    x = np.arange(len(rows))
    for xi, r in zip(x, rows):
        ax.plot([xi, xi], [r["d1"], r["d2"]], color=GRID, lw=3, zorder=1)
        dot(ax, xi, r["d1"], INK3)
        dot(ax, xi, r["d2"], MODEL_COLOR[r["model"]])
        ax.text(xi + 0.13, r["d2"], f"{r['d2']:.1f}", va="center", color=INK, fontsize=10)
        ax.text(xi + 0.13, r["d1"], f"{r['d1']:.1f}", va="center", color=INK2, fontsize=10)
    for v in (ref["9M"], ref["136M"], ref["270M"]):
        ax.axhline(v, color=INK3, lw=1, zorder=0)
    ax.text(-0.3, ref["136M"] - 0.9, f"DeepMind 270M {ref['270M']}  /  136M {ref['136M']}", color=INK2, fontsize=9, va="top")
    ax.text(-0.3, ref["9M"] - 4.2, f"DeepMind 9M {ref['9M']}", color=INK2, fontsize=9, va="top")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("puzzles solved (%)")
    style(ax)
    ax.plot([], [], "o", color=INK3, label="no search (depth-1)")
    ax.plot([], [], "o", color=BLUE, label="depth-2 beam-8")
    ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=2)

    gains = [r["d2"] - r["d1"] for r in rows]
    bars = ax2.bar(x, gains, width=0.36, color=[MODEL_COLOR[r["model"]] for r in rows], zorder=2)
    for b, g in zip(bars, gains):
        ax2.text(b.get_x() + b.get_width() / 2, g + 0.4, f"+{g:.1f}", ha="center", color=INK, fontsize=10)
    ax2.set_xticks(x, [r["d1"] for r in rows])
    ax2.set_xlabel("no-search accuracy of the same checkpoint (%)  →  the gain from one ply peaks in the middle")
    ax2.set_ylabel("gain (pts)")
    ax2.set_ylim(0, 17)
    style(ax2)
    fig.savefig(OUT / "search_amplification.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_node_ladder():
    rows = RES["node_ladder"]["rows"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.97)
    title(fig, "av55m + 180 net evals/move vs full-strength Stockfish 17.1",
          "Score out of 10 games against Stockfish limited to N nodes per move. 5/10 = even. "
          "Mid-training checkpoint (34% of data) traced; later checkpoints as single rungs.")
    mid = [r for r in rows if r["examples_seen"] == 696000000]
    ax.plot([r["sf_nodes"] for r in mid], [r["score"] for r in mid], color=BLUE, lw=2, zorder=2)
    for r in mid:
        dot(ax, r["sf_nodes"], r["score"], BLUE)
    for r in rows:
        if r["examples_seen"] == 907000000:
            dot(ax, r["sf_nodes"], r["score"], ORANGE)
        elif r["examples_seen"] == 1540000000:
            dot(ax, r["sf_nodes"], r["score"], AQUA)
        elif r["examples_seen"] == 2069438464:
            dot(ax, r["sf_nodes"], r["score"], INK)
    ax.axhline(5, color=INK3, lw=1, zorder=0)
    ax.text(70, 5.2, "even", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([100, 400, 1600, 3200, 6400, 12800], ["100", "400", "1,600", "3,200", "6,400", "12,800"])
    ax.set_xlim(70, 20000)
    ax.set_ylim(0, 10.4)
    ax.set_xlabel("Stockfish nodes per move (log)")
    ax.set_ylabel("model score / 10")
    style(ax)
    ax.plot([], [], "o-", color=BLUE, label="0.70B examples (34%)")
    ax.plot([], [], "o", color=ORANGE, label="0.91B (44%)")
    ax.plot([], [], "o", color=AQUA, label="1.54B (74%)")
    ax.plot([], [], "o", color=INK, label="2.07B (final)")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.text(0.02, 0.04, "Final checkpoint rungs (6.0 / 1.0 / 3.0) are non-monotone: 10-game noise. "
            "Honest final crossover ≈ 3,500-4,500 nodes.", transform=ax.transAxes, fontsize=9, color=INK2)
    fig.savefig(OUT / "node_ladder.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_crossover():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.1, right=0.97)
    title(fig, "Where the crossover sits as av55m trains",
          "Stockfish node count per move at which model+180-eval search scores 5/10. Estimated from the ladder rungs; the final point is a range.")
    xs = [0.70, 0.91, 1.54, 2.07]
    ys = [1750, 3300, 4500, 4000]
    ax.plot(xs[:3], ys[:3], color=BLUE, lw=2, zorder=2)
    for x, y in zip(xs[:3], ys[:3]):
        dot(ax, x, y, BLUE)
        ax.text(x, y + 230, f"~{y:,}", ha="center", color=INK, fontsize=10)
    ax.plot([2.07, 2.07], [3500, 4500], color=BLUE, lw=6, alpha=0.35, solid_capstyle="round", zorder=1)
    dot(ax, 2.07, 4000, BLUE)
    ax.text(2.07, 4750, "3,500-4,500", ha="center", color=INK, fontsize=10)
    ax.set_xlabel("training examples seen (billions)")
    ax.set_ylabel("Stockfish nodes/move at crossover")
    ax.set_ylim(0, 6000)
    ax.set_xlim(0.5, 2.3)
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000], ["0", "1,000", "2,000", "3,000", "4,000", "5,000", "6,000"])
    style(ax)
    ax.text(0.98, 0.05, "Tournament Stockfish searches ~10M+ nodes/move: the gap is still 11+ doublings.",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    fig.savefig(OUT / "crossover_vs_training.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_training_curves():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.9)
    title(fig, "Held-out loss vs data for the three runs",
          "Cross-entropy over 128 win-probability buckets on 100k held-out ChessBench action values. Log x.")
    for name in ["nano", "av9m", "av55m"]:
        xs, ys = [], []
        for line in (ROOT / "results" / "train_logs" / f"{name}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "eval_loss" in r and "examples_seen" in r:
                xs.append(r["examples_seen"]); ys.append(r["eval_loss"])
        ax.plot(xs, ys, color=MODEL_COLOR[name], lw=2)
        ax.text(xs[-1] * 1.06, ys[-1], f"{MODEL_LABEL[name]}  {ys[-1]:.2f}", va="center", color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([1e7, 1e8, 1e9], ["10M", "100M", "1B"])
    ax.set_xlim(2e6, 6e9)
    ax.set_ylim(2.0, 4.0)
    ax.set_xlabel("training examples seen")
    ax.set_ylabel("held-out loss")
    style(ax)
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fig 5: summary card (for the post)
def fig_summary_card():
    m = RES["models"]["av55m"]
    fig = plt.figure(figsize=(12, 6.75))
    fig.text(0.04, 0.93, "A 52M-parameter chess model + one ply of search", fontsize=22, fontweight="semibold", color=INK, va="top")
    fig.text(0.04, 0.845, "Trained on Stockfish-labeled positions for about $100 on one A100. 180 network evaluations per move.\n"
             "Scores 6/10 against full-strength Stockfish 17.1 limited to 3,200 nodes per move.",
             fontsize=11.5, color=INK2, va="top", linespacing=1.4)
    tiles = [("parameters", "51.6M"), ("training examples", "2.07B"), ("GPU-hours", "120"),
             ("approx. cost", "$100"), ("puzzles, no search", "86.9%"), ("puzzles, depth-2", "92.0%")]
    for i, (lab, val) in enumerate(tiles):
        x = 0.04 + i * 0.158
        fig.text(x, 0.72, val, fontsize=24, fontweight="semibold", color=INK, va="top")
        fig.text(x, 0.63, lab, fontsize=10, color=INK2, va="top")
    ax = fig.add_axes([0.07, 0.11, 0.9, 0.44])
    rows = [r for r in RES["node_ladder"]["rows"] if r["examples_seen"] == 696000000]
    fin = [r for r in RES["node_ladder"]["rows"] if r["examples_seen"] == 2069438464]
    ax.plot([r["sf_nodes"] for r in rows], [r["score"] for r in rows], color=BLUE, lw=2)
    for r in rows:
        dot(ax, r["sf_nodes"], r["score"], BLUE)
    for r in fin:
        dot(ax, r["sf_nodes"], r["score"], INK)
    ax.axhline(5, color=INK3, lw=1, zorder=0)
    ax.text(72, 5.25, "even", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([100, 400, 1600, 3200, 6400, 12800], ["100", "400", "1,600", "3,200", "6,400", "12,800"])
    ax.set_xlim(70, 20000); ax.set_ylim(0, 10.4)
    ax.set_xlabel("Stockfish 17.1 nodes per move (log)")
    ax.set_ylabel("model score / 10 games")
    style(ax)
    ax.plot([], [], "o-", color=BLUE, label="34% trained")
    ax.plot([], [], "o", color=INK, label="final checkpoint")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.text(0.04, 0.02, "github.com/bryjudy/little-search-chess", fontsize=10, color=INK3)
    fig.savefig(OUT / "summary_card.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_search_amplification()
    fig_node_ladder()
    fig_crossover()
    fig_training_curves()
    fig_summary_card()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
