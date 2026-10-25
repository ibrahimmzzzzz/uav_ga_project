"""All plotting helpers (matplotlib, headless)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from .chromosome import X, Y, A

COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def _background(ax, env):
    G = env.G
    ax.imshow(env.priority, origin="lower", cmap="YlOrRd", alpha=0.65, vmin=0, vmax=1,
              extent=(-.5, G - .5, -.5, G - .5))
    ax.contourf(env.hazard, levels=[0.25, 0.55, 1.01], colors=["#8e44ad", "#4a235a"], alpha=0.35)
    for (x0, y0, w, h) in env.nofly_rects:
        ax.add_patch(Rectangle((x0 - .5, y0 - .5), w, h, fill=True, facecolor="#555", alpha=.8,
                               hatch="///", edgecolor="k"))
    ax.set_xlim(-.5, G - .5); ax.set_ylim(-.5, G - .5); ax.set_aspect("equal")


def draw_solution(ax, env, evaluator, genes, title=""):
    m = evaluator.simulate(genes)
    _background(ax, env)
    if (m["cover_count"] > 0).any():
        ax.contour(m["cover_count"] > 0, levels=[0.5], colors="cyan", linewidths=1.2)
    t = env.targets_xy
    ax.scatter(t[m["detected"], 0], t[m["detected"], 1], c="lime", edgecolors="k", s=28, zorder=5)
    ax.scatter(t[~m["detected"], 0], t[~m["detected"], 1], c="red", marker="x", s=28, zorder=5)
    for d in range(genes.shape[0]):
        pts = np.vstack([env.base, genes[d][:, [X, Y]], env.base])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=COLORS[d % 6], lw=1.6, alpha=.9, zorder=4)
        on = genes[d][:, A] > .5
        ax.scatter(genes[d][on, X], genes[d][on, Y], color=COLORS[d % 6], s=22, zorder=6)
        ax.scatter(genes[d][~on, X], genes[d][~on, Y], facecolors="white", edgecolors=COLORS[d % 6], s=22, zorder=6)
    ax.scatter(*env.base, marker="*", s=180, c="gold", edgecolors="k", zorder=7)
    ax.set_title(f"{title}\ncov {m['coverage']:.2f}  det {m['detection']:.2f}  hazard {m['hazard']:.0f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return m


def plot_environment(env, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _background(ax, env)
    ax.scatter(env.targets_xy[:, 0], env.targets_xy[:, 1], c="k", s=15, label="targets")
    ax.scatter(*env.base, marker="*", s=200, c="gold", edgecolors="k", label="base")
    ax.set_title("Hazard grid: priority (orange), hazard (purple), no-fly (hatched)")
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_convergence(h, path, rs_hist=None, pop_size=None):
    gens = np.arange(len(h["best"]))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    axs[0].plot(gens, h["best"], label="GA best", lw=2)
    axs[0].plot(gens, h["avg"], label="GA average", lw=1.5)
    axs[0].fill_between(gens, h["worst"], h["best"], alpha=.12)
    if rs_hist is not None:
        axs[0].plot(np.arange(len(rs_hist["best"])), rs_hist["best"], "--", c="k", label="Random search best")
    axs[0].set_xlabel("generation (random search: batch of pop_size evaluations)")
    axs[0].set_ylabel("fitness"); axs[0].set_title("Fitness convergence"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(gens, h["coverage"], label="coverage"); axs[1].plot(gens, h["detection"], label="detection")
    axs[1].plot(gens, h["diversity"], label="diversity", ls=":")
    axs[1].plot(gens, h["feasible_frac"], label="feasible fraction", ls="--")
    axs[1].set_xlabel("generation"); axs[1].set_title("Best-individual objectives & population state")
    axs[1].legend(); axs[1].grid(alpha=.3)
    ax2 = axs[2]
    ax2.plot(gens, np.array(h["energy"]), c="tab:red", label="energy")
    ax2.set_ylabel("energy", color="tab:red"); ax2.set_xlabel("generation")
    ax3 = ax2.twinx(); ax3.plot(gens, h["hazard"], c="tab:purple", label="hazard"); ax3.plot(gens, h["time"], c="tab:green", label="time")
    ax3.set_ylabel("hazard / time"); ax2.set_title("Cost terms of best individual"); ax2.grid(alpha=.3)
    ax3.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_generation_snapshots(env, evaluator, genes_by_gen, path, picks=None):
    n = len(genes_by_gen) - 1
    picks = picks or sorted({0, max(1, n // 10), n // 3, (2 * n) // 3, n})
    fig, axs = plt.subplots(2, len(picks), figsize=(3.6 * len(picks), 7.4))
    for i, g in enumerate(picks):
        m = draw_solution(axs[0, i], env, evaluator, genes_by_gen[g], f"Best path - generation {g}")
        im = axs[1, i].imshow(m["cover_count"], origin="lower", cmap="Blues", vmin=0, vmax=env.cfg.n_drones)
        axs[1, i].contour(env.priority, levels=[0.5], colors="darkred", linewidths=.8)
        axs[1, i].set_title(f"Coverage heatmap - gen {g}", fontsize=9)
        axs[1, i].set_xticks([]); axs[1, i].set_yticks([])
    fig.colorbar(im, ax=axs[1, :], shrink=.7, label="# drones covering cell (red contour = priority>0.5)")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_final(env, evaluator, ga_genes, rs_genes, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    draw_solution(axs[0], env, evaluator, ga_genes, "Evolved (GA) solution")
    draw_solution(axs[1], env, evaluator, rs_genes, "Best random-search solution")
    fig.suptitle("Filled dot = sensor ON, hollow = OFF; green = detected target, red x = missed; cyan = coverage")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
