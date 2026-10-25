"""Mission simulator + multi-objective fitness with dynamic penalties for hard constraints."""
import numpy as np
from .chromosome import X, Y, V, P, A


class Evaluator:
    def __init__(self, env, drone, weights):
        self.env, self.dc, self.w = env, drone, weights
        self.n_evals = 0

    # ------------------------------------------------------------------ simulation
    def simulate(self, genes):
        env, dc = self.env, self.dc
        G = env.G
        D, K, _ = genes.shape
        cover = np.zeros((G, G), dtype=np.int16)            # how many drones cover each cell
        detected = np.zeros(len(env.targets_xy), dtype=bool)
        energy = np.zeros(D); times = np.zeros(D); lengths = np.zeros(D)
        hazard = 0.0; nofly_hits = 0
        for d in range(D):
            wp = genes[d]
            pts = np.vstack([env.base, wp[:, [X, Y]], env.base])
            dmask = np.zeros((G, G), dtype=bool)
            for k in range(K + 1):
                a, b = pts[k], pts[k + 1]
                j = min(k, K - 1)                             # return leg reuses last speed
                v = wp[j, V]
                L = float(np.hypot(*(b - a)))
                t = L / v
                lengths[d] += L; times[d] += t
                energy[d] += L * (dc.c_move0 + dc.c_move_v2 * v * v)
                n = max(1, int(np.ceil(L)))
                s = np.arange(n + 1) / n
                px = a[0] + s * (b[0] - a[0]); py = a[1] + s * (b[1] - a[1])
                cx = np.clip(np.rint(px).astype(int), 0, G - 1)
                cy = np.clip(np.rint(py).astype(int), 0, G - 1)
                hazard += env.hazard[cy[1:], cx[1:]].sum() * (L / n)
                nofly_hits += int(env.nofly[cy[1:], cx[1:]].sum())
                if k < K and wp[j, A] > 0.5:                  # sensor ON for this leg
                    p = wp[j, P]
                    energy[d] += dc.c_sensor * p * p * t
                    r = dc.r_min + p * (dc.r_max - dc.r_min)
                    r *= 1.0 - dc.speed_blur * (v - dc.v_min) / (dc.v_max - dc.v_min)
                    ri = max(1, int(round(r)))
                    sel = np.arange(0, n + 1, max(1, int(0.7 * ri)))
                    if sel[-1] != n:
                        sel = np.append(sel, n)
                    dy, dx = env.disc(ri)
                    yy = cy[sel][:, None] + dy[None, :]
                    xx = cx[sel][:, None] + dx[None, :]
                    ok = (yy >= 0) & (yy < G) & (xx >= 0) & (xx < G)
                    dmask[yy[ok], xx[ok]] = True
                    sp = np.stack([px[sel], py[sel]], axis=1)
                    d2 = ((env.targets_xy[:, None, :] - sp[None]) ** 2).sum(-1)
                    detected |= (d2 <= ri * ri).any(axis=1)
            cover += dmask
        covered = cover > 0
        return dict(
            cover_count=cover, detected=detected, energy_d=energy, time_d=times, length_d=lengths,
            coverage=float(env.priority[covered].sum() / env.total_priority),
            detection=float(env.target_weight[detected].sum() / env.target_weight.sum()),
            energy=float(energy.sum()), time=float(times.max()), hazard=float(hazard),
            nofly_hits=nofly_hits,
        )

    # ------------------------------------------------------------------ fitness
    def evaluate(self, genes, penalty_scale=1.0):
        """Return (fitness, metrics). Higher fitness is better."""
        self.n_evals += 1
        dc, w = self.dc, self.w
        m = self.simulate(genes)
        D = genes.shape[0]
        e_norm = m["energy"] / (D * dc.battery)
        t_norm = m["time"] / dc.deadline
        h_norm = m["hazard"] / (D * dc.max_range)
        objective = (w.coverage * m["coverage"] + w.detection * m["detection"]
                     - w.energy * e_norm - w.time * t_norm - w.hazard * h_norm)
        v_range = np.maximum(0, m["length_d"] - dc.max_range).sum() / dc.max_range
        v_batt = np.maximum(0, m["energy_d"] - dc.battery).sum() / dc.battery
        v_time = np.maximum(0, m["time_d"] - dc.deadline).sum() / dc.deadline
        v_nofly = m["nofly_hits"] / 10.0
        penalty = penalty_scale * (w.lam_range * v_range + w.lam_battery * v_batt
                                   + w.lam_time * v_time + w.lam_nofly * v_nofly)
        m.update(objective=float(objective), penalty=float(penalty),
                 violation=float(v_range + v_batt + v_time + v_nofly),
                 feasible=bool(v_range + v_batt + v_time + v_nofly < 1e-9),
                 e_norm=e_norm, t_norm=t_norm, h_norm=h_norm)
        return float(objective - penalty), m
