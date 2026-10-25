"""Baseline: pure random search with the same evaluation budget and same random-genome sampler as GA gen-0."""
import numpy as np
from .chromosome import Chromosome


def random_search(env, drone, evaluator, n_evals, batch, penalty_scale, seed=0):
    rng = np.random.default_rng(seed)
    best, best_f = None, -np.inf
    hist = {"best": [], "avg": []}
    done = 0
    while done < n_evals:
        fs = []
        for _ in range(min(batch, n_evals - done)):
            c = Chromosome.random(env, drone, rng)
            c.fitness, c.metrics = evaluator.evaluate(c.genes, penalty_scale)
            fs.append(c.fitness)
            if c.fitness > best_f:
                best, best_f = c, c.fitness
        done += len(fs)
        hist["best"].append(best_f)          # best-so-far
        hist["avg"].append(float(np.mean(fs)))  # mean of this batch of random samples
    return best, hist
