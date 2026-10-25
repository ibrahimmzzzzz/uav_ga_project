"""Hyper-parameter sensitivity: population size, mutation rate, selection pressure (tournament size).
Each setting is run with several seeds; results go to outputs/sensitivity.csv and sensitivity.png."""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga.config import EnvConfig, DroneConfig, FitnessWeights, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.genetic_algorithm import GeneticAlgorithm

SWEEPS = {
    "pop_size": [20, 40, 80],
    "mutation_rate": [0.01, 0.05, 0.12, 0.30],
    "tournament_size": [1, 2, 3, 6],        # 1 = no selection pressure (random mating)
}


def run_one(env, drone, weights, param, value, seed, gens):
    cfg = GAConfig(seed=seed, generations=gens, patience=10 ** 6)
    setattr(cfg, param, value)
    if param != "pop_size":
        cfg.pop_size = 40
    res = GeneticAlgorithm(env, drone, Evaluator(env, drone, weights), cfg).run(verbose=False)
    b = res.best
    return dict(fitness=b.fitness, coverage=b.metrics["coverage"], detection=b.metrics["detection"],
                feasible=int(b.metrics["feasible"]), evals=res.n_evals,
                final_div=res.history["diversity"][-1],
                gen_to_90=int(np.argmax(np.array(res.history["best"]) >= 0.9 * max(res.history["best"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env, drone, weights = Environment(EnvConfig()), DroneConfig(), FitnessWeights()

    rows = []
    for param, values in SWEEPS.items():
        for v in values:
            rs = [run_one(env, drone, weights, param, v, s, a.gens) for s in range(a.seeds)]
            row = dict(param=param, value=v)
            for k in rs[0]:
                row[k + "_mean"] = float(np.mean([r[k] for r in rs]))
                row[k + "_std"] = float(np.std([r[k] for r in rs]))
            rows.append(row)
            print(f"{param:16s}={v:<6} fitness {row['fitness_mean']:+.3f} ± {row['fitness_std']:.3f} | "
                  f"cov {row['coverage_mean']:.3f} | feasible {row['feasible_mean']:.2f} | "
                  f"final diversity {row['final_div_mean']:.2f} | gen->90% {row['gen_to_90_mean']:.0f}")
    with open(f"{a.out}/sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (param, values) in zip(axs, SWEEPS.items()):
        sub = [r for r in rows if r["param"] == param]
        ax.errorbar([str(r["value"]) for r in sub], [r["fitness_mean"] for r in sub],
                    yerr=[r["fitness_std"] for r in sub], marker="o", capsize=4, label="final fitness")
        ax.set_xlabel(param); ax.set_ylabel("final best fitness (mean ± std)"); ax.grid(alpha=.3)
        ax2 = ax.twinx()
        ax2.plot([str(r["value"]) for r in sub], [r["final_div_mean"] for r in sub], "r--s", label="final diversity")
        ax2.set_ylabel("final population diversity", color="r")
        ax.set_title(f"Sensitivity to {param}")
    fig.tight_layout(); fig.savefig(f"{a.out}/05_sensitivity.png", dpi=150)
    print(f"saved {a.out}/sensitivity.csv and 05_sensitivity.png")


if __name__ == "__main__":
    main()
