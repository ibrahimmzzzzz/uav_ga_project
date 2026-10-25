"""Central configuration (all tunable numbers live here)."""
from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_size: int = 50          # grid is grid_size x grid_size cells
    n_drones: int = 4
    n_waypoints: int = 8         # waypoints per drone (waypoint 0 = deployment/staging point)
    n_targets: int = 30          # hidden targets to detect
    n_priority_zones: int = 4    # high-priority disaster zones
    n_hazard_zones: int = 5      # wind / radiation pockets
    n_nofly: int = 4             # rectangular no-fly zones
    base: tuple = (2.0, 2.0)     # launch / recovery station (x, y)
    seed: int = 7                # environment seed (same map for every experiment)


@dataclass
class DroneConfig:
    v_min: float = 1.0           # cells / s
    v_max: float = 5.0
    r_min: float = 1.5           # sensor radius at power 0 (cells)
    r_max: float = 6.0           # sensor radius at power 1
    speed_blur: float = 0.35     # fraction of sensor radius lost at v_max (fast flight = blurry sensing)
    max_range: float = 190.0     # HARD: max path length per drone (cells)
    battery: float = 650.0       # HARD: energy budget per drone
    deadline: float = 110.0      # HARD: max flight time per drone (s)
    c_move0: float = 1.0         # energy per cell, constant part
    c_move_v2: float = 0.12      # energy per cell, speed^2 (drag) part
    c_sensor: float = 1.5        # sensor energy = c_sensor * power^2 * time


@dataclass
class FitnessWeights:
    # objective weights (all terms are normalised to ~[0, 1])
    coverage: float = 0.30       # maximise
    detection: float = 0.25      # maximise
    energy: float = 0.15         # minimise
    time: float = 0.10           # minimise
    hazard: float = 0.20         # minimise
    # constraint penalty multipliers
    lam_range: float = 3.0
    lam_battery: float = 3.0
    lam_time: float = 3.0
    lam_nofly: float = 4.0


@dataclass
class GAConfig:
    pop_size: int = 60
    generations: int = 100
    selection: str = "tournament"   # "tournament" | "rank"
    tournament_size: int = 3        # selection pressure
    crossover_rate: float = 0.85
    mutation_rate: float = 0.12     # per-gene mutation probability
    sigma_frac: float = 0.12        # initial Gaussian step as a fraction of the grid size
    elite: int = 2
    patience: int = 40              # stop if no improvement for this many generations
    immigrant_frac: float = 0.10    # random immigrants injected when diversity collapses
    diversity_threshold: float = 0.30
    penalty_start: float = 1.0      # dynamic penalty multiplier, ramps up linearly
    penalty_end: float = 3.0
    seed: int = 0
