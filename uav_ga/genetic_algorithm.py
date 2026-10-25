"""GA engine: initialisation, selection, crossover, mutation, elitism, diversity control, termination."""
from dataclasses import dataclass, field
import numpy as np
from .chromosome import Chromosome, repair, X, Y
from . import operators as ops


@dataclass
class GAResult:
    best: Chromosome
    history: dict = field(default_factory=dict)
    best_genes_by_gen: list = field(default_factory=list)
    n_evals: int = 0
    stop_reason: str = ""


class GeneticAlgorithm:
    def __init__(self, env, drone, evaluator, cfg):
        self.env, self.dc, self.ev, self.cfg = env, drone, evaluator, cfg
        self.rng = np.random.default_rng(cfg.seed)

    # penalty multiplier grows linearly -> early exploration of infeasible space, late strictness
    def penalty_scale(self, gen):
        c = self.cfg
        return c.penalty_start + (c.penalty_end - c.penalty_start) * gen / max(1, c.generations)

    def _evaluate(self, pop, scale):
        for c in pop:
            c.fitness, c.metrics = self.ev.evaluate(c.genes, scale)

    def diversity(self, pop):
        """Mean per-gene std of waypoint coordinates, relative to a uniform random population."""
        pos = np.stack([c.genes[..., [X, Y]].ravel() for c in pop])
        return float(pos.std(axis=0).mean() / (self.env.G / np.sqrt(12)))

    def _select(self, fits):
        if self.cfg.selection == "rank":
            return ops.rank_select(fits, self.rng)
        return ops.tournament_select(fits, self.cfg.tournament_size, self.rng)

    def run(self, verbose=True):
        c, rng = self.cfg, self.rng
        pop = [Chromosome.random(self.env, self.dc, rng) for _ in range(c.pop_size)]
        scale = self.penalty_scale(0)
        self._evaluate(pop, scale)
        h = {k: [] for k in ("best", "avg", "worst", "diversity", "feasible_frac",
                              "coverage", "detection", "energy", "time", "hazard", "mut_rate")}
        res = GAResult(best=None)
        best_so_far, stall, reason = -np.inf, 0, "max generations"
        mut_rate = c.mutation_rate

        for gen in range(c.generations + 1):
            fits = np.array([p.fitness for p in pop])
            bi = int(fits.argmax())
            div = self.diversity(pop)
            m = pop[bi].metrics
            for k, v in (("best", fits[bi]), ("avg", fits.mean()), ("worst", fits.min()),
                         ("diversity", div), ("feasible_frac", np.mean([p.metrics["feasible"] for p in pop])),
                         ("coverage", m["coverage"]), ("detection", m["detection"]), ("energy", m["energy"]),
                         ("time", m["time"]), ("hazard", m["hazard"]), ("mut_rate", mut_rate)):
                h[k].append(float(v))
            res.best_genes_by_gen.append(pop[bi].genes.copy())
            if verbose and (gen % 10 == 0 or gen == c.generations):
                print(f"gen {gen:4d} | best {fits[bi]:+.4f} avg {fits.mean():+.4f} | div {div:.2f} | "
                      f"cov {m['coverage']:.2f} det {m['detection']:.2f} feasible {h['feasible_frac'][-1]:.2f}")
            if fits[bi] > best_so_far + 1e-5:
                best_so_far, stall = fits[bi], 0
            else:
                stall += 1
            if gen == c.generations:
                break
            if stall >= c.patience:
                reason = f"no improvement for {c.patience} generations"
                break

            # ---- breed next generation -------------------------------------------------
            order = np.argsort(-fits)
            nxt = [pop[i].copy() for i in order[:c.elite]]
            sigma = c.sigma_frac * self.env.G * (1 - 0.6 * gen / c.generations)
            while len(nxt) < c.pop_size:
                p1, p2 = pop[self._select(fits)].genes, pop[self._select(fits)].genes
                if rng.random() < c.crossover_rate:
                    a, b = ops.crossover(p1, p2, rng)
                else:
                    a, b = p1.copy(), p2.copy()
                for ch in (a, b):
                    ch = ops.mutate(ch, self.env, self.dc, mut_rate, sigma, rng)
                    nxt.append(Chromosome(repair(ch, self.env, self.dc)))
                    if len(nxt) >= c.pop_size:
                        break
            # ---- diversity maintenance: adaptive mutation + random immigrants ----------
            if div < c.diversity_threshold:
                mut_rate = min(0.5, mut_rate * 1.3)
                n_imm = max(1, int(c.immigrant_frac * c.pop_size))
                for i in range(n_imm):
                    nxt[-1 - i] = Chromosome.random(self.env, self.dc, rng)
            else:
                mut_rate = max(c.mutation_rate, mut_rate * 0.9)
            pop = nxt[:c.pop_size]
            scale = self.penalty_scale(gen + 1)
            self._evaluate(pop, scale)

        # final answer: re-score at the strictest penalty and return the best one
        self._evaluate(pop, self.penalty_scale(c.generations))
        res.best = max(pop, key=lambda p: p.fitness).copy()
        res.history, res.n_evals, res.stop_reason = h, self.ev.n_evals, reason
        return res
