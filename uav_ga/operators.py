"""Genetic operators: selection, crossovers (route / segment / geometric), spatial mutations."""
import numpy as np
from .chromosome import X, Y, V, P, A


# ----------------------------------------------------------------------- selection
def tournament_select(fits, k, rng):
    idx = rng.integers(0, len(fits), k)
    return idx[np.argmax(fits[idx])]


def rank_select(fits, rng):
    ranks = np.argsort(np.argsort(fits)) + 1.0          # worst = 1 ... best = N
    return rng.choice(len(fits), p=ranks / ranks.sum())


# ----------------------------------------------------------------------- crossover
def route_swap(p1, p2, rng):
    """Swap entire drone routes between parents (keeps each drone's good tour intact)."""
    c1, c2 = p1.copy(), p2.copy()
    s = rng.random(p1.shape[0]) < 0.5
    if not s.any():
        s[rng.integers(p1.shape[0])] = True
    c1[s], c2[s] = p2[s].copy(), p1[s].copy()
    return c1, c2


def segment_swap(p1, p2, rng):
    """Swap a contiguous waypoint segment (all 5 genes) inside each selected drone's route."""
    c1, c2 = p1.copy(), p2.copy()
    K = p1.shape[1]
    for d in range(p1.shape[0]):
        if rng.random() < 0.6:
            i, j = sorted(rng.integers(0, K, 2))
            c1[d, i:j + 1], c2[d, i:j + 1] = p2[d, i:j + 1].copy(), p1[d, i:j + 1].copy()
    return c1, c2


def geometric_blend(p1, p2, rng):
    """Geometric (arithmetic) crossover on continuous genes; uniform crossover on schedule bits."""
    a = rng.random((p1.shape[0], p1.shape[1], 1))
    c1, c2 = p1.copy(), p2.copy()
    c1[..., :A] = a * p1[..., :A] + (1 - a) * p2[..., :A]
    c2[..., :A] = (1 - a) * p1[..., :A] + a * p2[..., :A]
    m = rng.random(p1.shape[:2]) < 0.5
    c1[..., A] = np.where(m, p1[..., A], p2[..., A])
    c2[..., A] = np.where(m, p2[..., A], p1[..., A])
    return c1, c2


CROSSOVERS = (route_swap, segment_swap, geometric_blend)
CROSSOVER_PROBS = (0.30, 0.40, 0.30)


def crossover(p1, p2, rng):
    f = CROSSOVERS[rng.choice(len(CROSSOVERS), p=CROSSOVER_PROBS)]
    return f(p1, p2, rng)


# ----------------------------------------------------------------------- mutation
def mutate(g, env, drone, rate, sigma, rng):
    """Spatially-aware mutation.
    - Gaussian jitter of coordinates (local search, step size `sigma` shrinks over time)
    - priority-biased reset of a waypoint (long jump -> escapes local optima / keeps diversity)
    - Gaussian tweak of speed and sensor power, bit-flip of the activation schedule
    - route-order mutation (reverse a sub-route or swap two waypoints, 2-opt style)
    """
    g = g.copy()
    D, K, _ = g.shape
    m = rng.random((D, K)) < rate
    g[..., [X, Y]] += m[..., None] * rng.normal(0, sigma, (D, K, 2))

    m = rng.random((D, K)) < rate * 0.3
    n = int(m.sum())
    if n:
        g[m, X:Y + 1] = env.sample_priority_cells(n, rng)

    m = rng.random((D, K)) < rate
    g[..., V] += m * rng.normal(0, 0.15 * (drone.v_max - drone.v_min), (D, K))
    g[..., P] += m * rng.normal(0, 0.15, (D, K))
    m = rng.random((D, K)) < rate * 0.5
    g[..., A] = np.where(m, 1.0 - g[..., A], g[..., A])

    for d in range(D):
        if rng.random() < min(1.0, rate * 2):
            i, j = sorted(rng.integers(0, K, 2))
            if rng.random() < 0.5:
                g[d, i:j + 1] = g[d, i:j + 1][::-1].copy()
            else:
                g[d, [i, j]] = g[d, [j, i]]
    return g
