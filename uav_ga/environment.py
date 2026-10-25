"""Discretised hazard grid: priority zones, hazard pockets, no-fly zones, hidden targets."""
import numpy as np
from .config import EnvConfig


class Environment:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        G = self.G = cfg.grid_size
        rng = np.random.default_rng(cfg.seed)
        yy, xx = np.mgrid[0:G, 0:G]
        self.base = np.array(cfg.base, dtype=float)

        def blob(cx, cy, s):
            return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))

        def far_from_base(margin):
            while True:
                c = rng.uniform(margin, G - margin, 2)
                if np.hypot(*(c - self.base)) > 14:
                    return c

        # --- priority map (0..1): where rescue value is concentrated -----------------
        pr = np.zeros((G, G))
        self.priority_centres = []
        for _ in range(cfg.n_priority_zones):
            c = far_from_base(8)
            self.priority_centres.append(c)
            pr += rng.uniform(0.6, 1.0) * blob(c[0], c[1], rng.uniform(3.0, 5.0))
        self.priority = 0.05 + 0.95 * pr / pr.max()

        # --- hazard map (0..1): wind / radiation pockets; some overlap priority zones --
        hz = np.zeros((G, G))
        for i in range(cfg.n_hazard_zones):
            if i < 2:                                  # force a trade-off: hazard on top of value
                c = self.priority_centres[i] + rng.normal(0, 4, 2)
            else:
                c = far_from_base(6)
            hz += rng.uniform(0.6, 1.0) * blob(c[0], c[1], rng.uniform(3.0, 6.0))
        hz = np.clip(hz / hz.max(), 0, 1)
        hz[hz < 0.15] = 0.0
        self.hazard = hz

        # --- no-fly zones (boolean) ----------------------------------------------------
        nf = np.zeros((G, G), dtype=bool)
        self.nofly_rects = []
        while len(self.nofly_rects) < cfg.n_nofly:
            w, h = rng.integers(4, 9, 2)
            x0, y0 = rng.integers(6, G - 10, 2)
            if np.hypot(x0 + w / 2 - self.base[0], y0 + h / 2 - self.base[1]) < 12:
                continue
            nf[y0:y0 + h, x0:x0 + w] = True
            self.nofly_rects.append((x0, y0, w, h))
        self.nofly = nf

        # --- hidden targets (more likely in high-priority areas) ----------------------
        p = (self.priority ** 2) * (~nf)
        p = (p / p.sum()).ravel()
        idx = rng.choice(G * G, size=cfg.n_targets, replace=False, p=p)
        ty, tx = np.divmod(idx, G)
        self.targets_xy = np.stack([tx, ty], axis=1).astype(float)
        self.target_weight = self.priority[ty, tx]

        # --- helpers --------------------------------------------------------------------
        self.total_priority = self.priority.sum()
        self._cdf = np.cumsum((self.priority * (~nf)).ravel())
        self._cdf /= self._cdf[-1]
        self._disc = {}
        free = np.argwhere(~nf)[:, ::-1].astype(float)           # (x, y) of free cells
        grid = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(float)
        d2 = ((grid[:, None, :] - free[None, :, :]) ** 2).sum(-1)
        self.nearest_free = free[d2.argmin(1)].reshape(G, G, 2)  # lookup used by repair

    def disc(self, r):
        """Integer offsets (dy, dx) of a filled disc of radius r."""
        if r not in self._disc:
            o = np.arange(-r, r + 1)
            dy, dx = np.meshgrid(o, o, indexing="ij")
            m = dx * dx + dy * dy <= r * r
            self._disc[r] = (dy[m], dx[m])
        return self._disc[r]

    def sample_priority_cells(self, n, rng):
        """Draw n (x, y) positions with probability proportional to priority (used by mutation)."""
        idx = np.searchsorted(self._cdf, rng.random(n))
        y, x = np.divmod(np.minimum(idx, self.G * self.G - 1), self.G)
        return np.stack([x, y], axis=1) + rng.uniform(-0.5, 0.5, (n, 2))
