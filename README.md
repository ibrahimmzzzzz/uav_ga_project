# UAV Strategy Engine – Genetic Algorithm for Multi-Objective Drone Mission Planning

Artificial Intelligence – Assignment 2 (CLO-4), Bahria University Lahore.
A GA that evolves deployment positions, waypoint trajectories, speeds and sensor schedules for a fleet
of UAVs over a 50x50 hazard grid, trading off coverage / detection against energy, time and hazard exposure.

## Repository layout
```
main.py                 run GA + random-search baseline, save all figures
sensitivity.py          population-size / mutation-rate / selection-pressure sweeps
compare_baseline.py     GA vs random search over several seeds + uniqueness statistics
requirements.txt
uav_ga/
  config.py             every tunable number (environment, drone physics, weights, GA params)
  environment.py        grid: priority zones, hazard pockets, no-fly zones, hidden targets
  chromosome.py         chromosome encoding + repair mechanism
  fitness.py            mission simulator, multi-objective fitness, dynamic penalties
  operators.py          selection, 3 crossovers, spatial mutation
  genetic_algorithm.py  GA loop, elitism, diversity control, termination criteria
  random_search.py      baseline
  visualization.py      plots
outputs/                generated figures, logs, CSV
```

## Run
```bash
pip install -r requirements.txt
python main.py                          # default: pop 60, 100 generations (~35 s)
python main.py --pop 100 --gens 150 --mut 0.1 --tournament 4 --seed 3
python compare_baseline.py --runs 6     # GA vs random search statistics
python sensitivity.py --seeds 3         # hyper-parameter study -> outputs/sensitivity.csv
```

## Design summary
**Chromosome** – array `(D drones, K waypoints, 5 genes)`: `x, y` (real), `speed` (real), `sensor power`
(real -> sensor radius), `sensor ON/OFF` (binary). Hybrid real/binary, multi-segment (one segment per drone);
waypoint 0 is the drone's deployment position. Each drone flies base -> waypoints -> base.

**Fitness** (maximise)
```
F = 0.30*coverage + 0.25*detection - 0.15*energy - 0.10*time - 0.20*hazard - penalty
```
coverage = priority-weighted fraction of cells sensed; detection = priority-weighted fraction of targets found;
energy = sum of `L(c0 + c1 v^2)` + sensor energy; time = slowest drone; hazard = integral of hazard along the paths.
Faster flight costs more energy and blurs the sensor (smaller effective radius), so objectives truly conflict.

**Hard constraints** – max range, battery, deadline, no-fly zones. Handled with
(1) a *dynamic penalty* whose multiplier grows from 1 to 3 over the run, and
(2) a *repair* step that clips genes and moves waypoints out of no-fly cells to the nearest free cell.

**Operators** – tournament (or rank) selection; crossover chosen at random among *route swap*, *segment swap*
and *geometric blend*; mutation = Gaussian coordinate jitter (annealed step), priority-biased waypoint reset,
speed/power tweak, schedule bit-flip, sub-route reversal/swap. Diversity is monitored; when it drops below a
threshold the mutation rate is raised and random immigrants are injected.

**Termination** – max generations or `patience` generations without improvement.

**Baseline** – random search with the identical evaluation budget and identical random-genome sampler.
