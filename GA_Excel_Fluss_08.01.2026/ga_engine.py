# GENETIC ALGORITHM MODULE

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import config
from helpers import (
    cell_center_from_topleft,
    effective_dims,
    normalize_individual,
    occupied_cells,
    random_individual,
    rect_corners,
    route_all_flows,
)


def init_population() -> List[List[Dict]]:
    return [random_individual() for _ in range(int(config.POPULATION_SIZE))]


def fitness(ind: List[Dict]) -> float:
    """Fitness mit Maschinen Kollisionen Obstacles Bounds und Flow Geometrie."""
    cost = 0.0

    flows, flow_pen = route_all_flows(ind)
    cost += float(flow_pen)

    for f in flows:
        path = f.get("path")
        if path is None:
            continue
        cost += float(config.DIST_SCALE) * float(len(path) - 1) * float(config.GRID_SIZE)

    cell_owner = {}
    for i, m in enumerate(ind):
        for c in occupied_cells(m):
            if c in cell_owner:
                cost += float(config.OVERLAP_PENALTY)
            else:
                cell_owner[c] = i

    for m in ind:
        for c in occupied_cells(m):
            if c in config.OBSTACLES:
                cost += float(config.OBSTACLE_PENALTY)

    for m in ind:
        w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
        h_m = float(m["h_cells"]) * float(config.GRID_SIZE)
        poly = rect_corners((float(m["x"]), float(m["y"])), w_m, h_m, int(m.get("z", 0)))
        out_count = 0
        for x, y in poly:
            if x < 0.0 or x > float(config.FLOOR_W) or y < 0.0 or y > float(config.FLOOR_H):
                out_count += 1
        if out_count:
            cost += float(config.OUT_OF_BOUNDS_PENALTY) * float(out_count)

    return cost


def uniform_crossover(a: List[Dict], b: List[Dict]) -> List[Dict]:
    child = []
    n = int(config.MACHINE_COUNT)
    for i in range(n):
        child.append(copy.deepcopy(a[i] if random.random() < 0.5 else b[i]))
    normalize_individual(child)
    return child


def _placement_ok(ind: List[Dict], machine_idx: int, cand: Dict) -> bool:
    cand_cells = occupied_cells(cand)

    for c in cand_cells:
        if c in config.OBSTACLES:
            return False
        if c[0] < 0 or c[1] < 0 or c[0] >= config.GRID_COLS or c[1] >= config.GRID_ROWS:
            return False

    for j, other in enumerate(ind):
        if j == machine_idx:
            continue
        if cand_cells & occupied_cells(other):
            return False

    return True


def mutate(ind: List[Dict]) -> None:
    """Mutation im Raster
    Position Mutation und Rotation Toggle haben eigene Wahrscheinlichkeiten
    Ungültige Kandidaten werden verworfen"""

    pos_prob = float(config.MUTATION_PROB)
    rot_prob = float(getattr(config, "MUTATION_ROT_PROB", 0.0))

    for i, m in enumerate(ind):
        base = copy.deepcopy(m)
        changed = False

        new_col = int(m["gx"])
        new_row = int(m["gy"])
        new_z = int(m.get("z", 0))

        if random.random() < pos_prob:
            delta_col = int(round(random.uniform(-float(config.MUTATION_POS_STD), float(config.MUTATION_POS_STD))))
            delta_row = int(round(random.uniform(-float(config.MUTATION_POS_STD), float(config.MUTATION_POS_STD))))
            new_col += delta_col
            new_row += delta_row
            changed = True

        if random.random() < rot_prob:
            new_z = new_z + 90 * random.choice([1,2,3])
            changed = True

        if not changed:
            continue

        w_eff, h_eff = effective_dims(m, new_z)
        max_col = max(0, config.GRID_COLS - int(w_eff))
        max_row = max(0, config.GRID_ROWS - int(h_eff))
        new_col = max(0, min(max_col, int(new_col)))
        new_row = max(0, min(max_row, int(new_row)))

        cand = copy.deepcopy(m)
        cand["z"] = int(new_z)
        cand["gx"] = int(new_col)
        cand["gy"] = int(new_row)
        cand["x"], cand["y"] = cell_center_from_topleft(cand["gx"], cand["gy"], w_eff, h_eff)

        if _placement_ok(ind, i, cand):
            ind[i] = cand
        else:
            ind[i] = base

    normalize_individual(ind)


def run_ga(generations: int, progress_callback=None) -> Tuple[Optional[List[Dict]], float]:
    from helpers import update_grid_counts

    update_grid_counts()

    pop = init_population()
    best_ind = None
    best_score = float("inf")

    for g in range(1, int(generations) + 1):
        if config.STOP_REQUESTED:
            if progress_callback:
                try:
                    progress_callback(g, generations, best_score, best_ind, pop)
                except TypeError:
                    progress_callback(g, generations, best_score, best_ind)
            break

        scores = [fitness(ind) for ind in pop]
        paired = list(zip(scores, pop))
        paired.sort(key=lambda p: p[0])

        elites = [copy.deepcopy(p[1]) for p in paired[: int(config.ELITE_KEEP)]]
        elite_scores = [p[0] for p in paired[: int(config.ELITE_KEEP)]]

        if elite_scores and elite_scores[0] < best_score:
            best_score = float(elite_scores[0])
            best_ind = copy.deepcopy(elites[0])

        if progress_callback:
            try:
                progress_callback(g, generations, best_score, best_ind, pop)
            except TypeError:
                progress_callback(g, generations, best_score, best_ind)

        new_pop: List[List[Dict]] = []
        new_pop.extend(copy.deepcopy(elites))

        while len(new_pop) < int(config.POPULATION_SIZE):
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = uniform_crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        pop = new_pop

    return best_ind, best_score
