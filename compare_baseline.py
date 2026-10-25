"""GA vs. random search over many seeds + uniqueness analysis of the evolved solutions."""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga.config import EnvConfig, DroneConfig, FitnessWeights, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.genetic_algorithm import GeneticAlgorithm
from uav_ga.random_search import random_search


def jaccard(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def mean_pairwise(masks):
    n = len(masks)
    return float(np.mean([jaccard(masks[i], masks[j]) for i in range(n) for j in range(i + 1, n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env, drone, weights = Environment(EnvConfig()), DroneConfig(), FitnessWeights()

    ga_f, rs_f, ga_m, rs_m, ga_feas, rs_feas = [], [], [], [], [], []
    for s in range(a.runs):
        cfg = GAConfig(seed=s, pop_size=a.pop, generations=a.gens, patience=10 ** 6)
        ga = GeneticAlgorithm(env, drone, Evaluator(env, drone, weights), cfg)
        res = ga.run(verbose=False)
        rs, _ = random_search(env, drone, Evaluator(env, drone, weights), res.n_evals, a.pop,
                              ga.penalty_scale(cfg.generations), seed=100 + s)
        ga_f.append(res.best.fitness); rs_f.append(rs.fitness)
        ga_feas.append(res.best.metrics["feasible"]); rs_feas.append(rs.metrics["feasible"])
        ga_m.append(res.best.metrics["cover_count"] > 0); rs_m.append(rs.metrics["cover_count"] > 0)
        print(f"seed {s}: GA {res.best.fitness:+.3f} (feasible={ga_feas[-1]})  |  RS {rs.fitness:+.3f} (feasible={rs_feas[-1]})")

    ga_f, rs_f = np.array(ga_f), np.array(rs_f)
    # Welch t statistic (no scipy needed)
    t = (ga_f.mean() - rs_f.mean()) / np.sqrt(ga_f.var(ddof=1) / len(ga_f) + rs_f.var(ddof=1) / len(rs_f))
    print(f"\nGA mean {ga_f.mean():+.3f} ± {ga_f.std():.3f} | RS mean {rs_f.mean():+.3f} ± {rs_f.std():.3f} | Welch t = {t:.1f}")
    print(f"feasible solutions: GA {sum(ga_feas)}/{a.runs}, RS {sum(rs_feas)}/{a.runs}")
    print(f"Mean pairwise coverage-map Jaccard similarity between runs: GA {mean_pairwise(ga_m):.3f}, RS {mean_pairwise(rs_m):.3f}")
    print(f"Overlap of GA-run coverage maps with the high-priority core (priority>0.5): "
          f"{np.mean([(m & (env.priority > .5)).sum() / (env.priority > .5).sum() for m in ga_m]):.3f} "
          f"vs RS {np.mean([(m & (env.priority > .5)).sum() / (env.priority > .5).sum() for m in rs_m]):.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    axs[0].boxplot([ga_f, rs_f]); axs[0].set_xticklabels(["GA", "Random search"])
    axs[0].set_ylabel("final fitness"); axs[0].set_title(f"Final fitness over {a.runs} seeds (equal evaluations)"); axs[0].grid(alpha=.3)
    axs[1].imshow(np.mean(ga_m, axis=0), origin="lower", cmap="viridis", vmin=0, vmax=1)
    axs[1].contour(env.priority, levels=[0.5], colors="r", linewidths=.8)
    axs[1].set_title("How often each cell is covered by GA solutions\n(red = high-priority core)")
    fig.tight_layout(); fig.savefig(f"{a.out}/06_ga_vs_random_stats.png", dpi=150)


if __name__ == "__main__":
    main()
