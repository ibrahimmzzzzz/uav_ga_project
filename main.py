"""Run the UAV Strategy Engine: GA + random-search baseline, with all figures saved to ./outputs."""
import argparse, json, os, time
import numpy as np
from uav_ga.config import EnvConfig, DroneConfig, FitnessWeights, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.genetic_algorithm import GeneticAlgorithm
from uav_ga.random_search import random_search
from uav_ga import visualization as viz


def summarize(name, c):
    m = c.metrics
    print(f"{name:14s} fitness {c.fitness:+.4f} | coverage {m['coverage']:.3f} detection {m['detection']:.3f} "
          f"energy {m['energy']:.0f} time {m['time']:.1f}s hazard {m['hazard']:.1f} "
          f"feasible={m['feasible']} (violation {m['violation']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--mut", type=float, default=None)
    ap.add_argument("--tournament", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    env_cfg, drone, weights, ga_cfg = EnvConfig(), DroneConfig(), FitnessWeights(), GAConfig(seed=a.seed)
    if a.pop: ga_cfg.pop_size = a.pop
    if a.gens: ga_cfg.generations = a.gens
    if a.mut is not None: ga_cfg.mutation_rate = a.mut
    if a.tournament: ga_cfg.tournament_size = a.tournament

    env = Environment(env_cfg)
    viz.plot_environment(env, f"{a.out}/01_environment.png")

    t0 = time.time()
    ga = GeneticAlgorithm(env, drone, Evaluator(env, drone, weights), ga_cfg)
    res = ga.run()
    print(f"GA finished in {time.time() - t0:.1f}s ({res.n_evals} evaluations, stop: {res.stop_reason})")

    # random-search baseline with the SAME number of evaluations, scored with the same final penalty
    ev_rs = Evaluator(env, drone, weights)
    rs_best, rs_hist = random_search(env, drone, ev_rs, res.n_evals, ga_cfg.pop_size,
                                     ga.penalty_scale(ga_cfg.generations), seed=a.seed + 1000)
    print(); summarize("Genetic Alg.", res.best); summarize("Random search", rs_best)

    ev = Evaluator(env, drone, weights)
    viz.plot_convergence(res.history, f"{a.out}/02_convergence.png", rs_hist, ga_cfg.pop_size)
    viz.plot_generation_snapshots(env, ev, res.best_genes_by_gen, f"{a.out}/03_evolution_snapshots.png")
    viz.plot_final(env, ev, res.best.genes, rs_best.genes, f"{a.out}/04_ga_vs_random.png")

    np.save(f"{a.out}/best_chromosome.npy", res.best.genes)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump({"ga": {k: v for k, v in res.best.metrics.items() if np.isscalar(v) or isinstance(v, bool)} | {"fitness": res.best.fitness},
                   "random_search": {k: v for k, v in rs_best.metrics.items() if np.isscalar(v) or isinstance(v, bool)} | {"fitness": rs_best.fitness},
                   "evaluations": res.n_evals, "stop_reason": res.stop_reason}, f, indent=2, default=float)
    print(f"\nFigures and results saved in {a.out}/")


if __name__ == "__main__":
    main()
