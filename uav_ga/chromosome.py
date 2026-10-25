"""Chromosome design.

A chromosome is a float array of shape (D, K, 5): D drones x K waypoints x 5 genes

    gene 0  x      waypoint x-coordinate           real  [0, G-1]
    gene 1  y      waypoint y-coordinate           real  [0, G-1]
    gene 2  v      speed on the leg INTO waypoint  real  [v_min, v_max]
    gene 3  p      sensor power at the waypoint    real  [0, 1]   -> sensor radius
    gene 4  a      sensor activation schedule      binary {0, 1}

(hybrid real-valued + binary encoding, multi-segment: one segment per drone).
Waypoint 0 of each drone is its deployment / staging position. Every drone launches
from the base, visits its waypoints in order, then returns to the base.
"""
import numpy as np

X, Y, V, P, A = range(5)
N_GENES = 5


class Chromosome:
    __slots__ = ("genes", "fitness", "metrics")

    def __init__(self, genes):
        self.genes = genes
        self.fitness = -np.inf
        self.metrics = {}

    def copy(self):
        c = Chromosome(self.genes.copy())
        c.fitness, c.metrics = self.fitness, self.metrics
        return c

    @staticmethod
    def random(env, drone, rng):
        D, K, G = env.cfg.n_drones, env.cfg.n_waypoints, env.G
        g = np.empty((D, K, N_GENES))
        g[..., X] = rng.uniform(0, G - 1, (D, K))
        g[..., Y] = rng.uniform(0, G - 1, (D, K))
        g[..., V] = rng.uniform(drone.v_min, drone.v_max, (D, K))
        g[..., P] = rng.uniform(0, 1, (D, K))
        g[..., A] = rng.integers(0, 2, (D, K))
        return Chromosome(repair(g, env, drone))


def repair(g, env, drone):
    """Repair mechanism: clip to bounds, binarise schedule, push waypoints out of no-fly zones."""
    G = env.G
    g[..., X] = np.clip(g[..., X], 0, G - 1)
    g[..., Y] = np.clip(g[..., Y], 0, G - 1)
    g[..., V] = np.clip(g[..., V], drone.v_min, drone.v_max)
    g[..., P] = np.clip(g[..., P], 0, 1)
    g[..., A] = (g[..., A] >= 0.5).astype(float)
    ix = np.rint(g[..., X]).astype(int)
    iy = np.rint(g[..., Y]).astype(int)
    bad = env.nofly[iy, ix]
    if bad.any():
        d, k = np.nonzero(bad)
        g[d, k, X] = env.nearest_free[iy[d, k], ix[d, k], 0]
        g[d, k, Y] = env.nearest_free[iy[d, k], ix[d, k], 1]
    return g
