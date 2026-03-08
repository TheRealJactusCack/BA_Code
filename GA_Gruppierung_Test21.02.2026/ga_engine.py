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
    distance_cost,
)


def init_population() -> List[List[Dict]]:
    return [random_individual() for _ in range(int(config.POPULATION_SIZE))]

def is_fixed_machine(m: Dict) -> bool:
    """
    True, wenn diese Maschine feste Koordinaten/Rotation aus Excel hat.
    Quelle der Wahrheit: config.MACHINE_FIXED (aligned list).
    """
    fixed_list = getattr(config, "MACHINE_FIXED", [])
    idx = int(m.get("idx", -1))
    return 0 <= idx < len(fixed_list) and fixed_list[idx] is not None

def enforce_fixed(ind: List[Dict]) -> None:
    fixed_list = getattr(config, "MACHINE_FIXED", [])
    if not fixed_list:
        return

    for m in ind:
        idx = int(m.get("idx", -1))
        if idx < 0 or idx >= len(fixed_list):
            continue
        fx = fixed_list[idx]
        if fx is None:
            continue

        z = fx.get("z", None)
        if z is not None:
            m["z"] = int(z)

        #Erwartung: fx enthält world-center x/y (so wie importiert)
        m["x"] = float(fx["x"])
        m["y"] = float(fx["y"])

        #gx/gy aus x/y + w_eff/h_eff ableiten, damit Grid/occupied korrekt bleibt
        w_eff, h_eff = effective_dims(m, int(m.get("z", 0)))
        gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
        m["gx"] = int(round((m["x"] / gs) - (float(w_eff) / 2.0)))
        m["gy"] = int(round((m["y"] / gs) - (float(h_eff) / 2.0)))

def fitness(ind: List[Dict]) -> float:
#Fitness mit Maschinen Kollisionen Obstacles Bounds und Flow Geometrie
    cost = 0.0
    
    cost += distance_cost(ind, config)

    cell_owner = {}
    for i, m in enumerate(ind):
        for c in occupied_cells(m, Clearance= True):
            if c in cell_owner:
                cost += float(config.OVERLAP_PENALTY)
            else:
                cell_owner[c] = i

    for m in ind:
        for c in occupied_cells(m, Clearance = True):
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
        enforce_fixed(child)
    normalize_individual(child)
    return child

def tauschen(ind: List[Dict], swap_prob: float) -> None:
    """Tauscht die koordianten von zwei Maschinen im Individuum"""
    if random.random() > swap_prob:
        return False
    i, j = random.sample(range(len(ind)), 2)  # garantiert unterschiedlich Maschinen
    m1, m2 = ind[i], ind[j]

    #feste Maschinen dürfen nicht getauscht werden
    if is_fixed_machine(m1) or is_fixed_machine(m2):
        return False

    m1_gx, m1_gy = m1["gx"], m1["gy"]
    m2_gx, m2_gy = m2["gx"], m2["gy"]

    m1["gx"], m2["gx"] = m2_gx, m1_gx
    m1["gy"], m2["gy"] = m2_gy, m1_gy

    w1, h1 = effective_dims(m1, int(m1.get("z", 0)))
    w2, h2 = effective_dims(m2, int(m2.get("z", 0)))

    m1["x"], m1["y"] = cell_center_from_topleft(int(m1["gx"]), int(m1["gy"]), int(w1), int(h1))
    m2["x"], m2["y"] = cell_center_from_topleft(int(m2["gx"]), int(m2["gy"]), int(w2), int(h2))
    return True

def _placement_ok(ind: List[Dict], machine_idx: int, cand: Dict) -> bool:
    cand_cells = occupied_cells(cand, Clearance = False)

    for c in cand_cells:
        if c in config.OBSTACLES:
            return False
        if c[0] < 0 or c[1] < 0 or c[0] >= config.GRID_COLS or c[1] >= config.GRID_ROWS:
            return False

    for j, other in enumerate(ind):
        if j == machine_idx:
            continue
        if cand_cells & occupied_cells(other, Clearance = False):
            return False

    return True


def mutate(ind: List[Dict]) -> None:
    """Mutation im Raster
    Position Mutation und Rotation Toggle haben eigene Wahrscheinlichkeiten
    Ungültige Kandidaten werden verworfen"""

    pos_prob = float(config.MUTATION_PROB)
    rot_prob = float(getattr(config, "MUTATION_ROT_PROB", 0.0))

    for i, m in enumerate(ind):

        if is_fixed_machine(m):
            continue

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
            new_z = new_z + 90 * random.choice([1,2,3]) % 360
            changed = True

        if not changed:
            continue

        new_z = int(new_z) % 360
        w_eff, h_eff = effective_dims(m, new_z)

        max_col = max(0, config.GRID_COLS - w_eff)
        max_row = max(0, config.GRID_ROWS - h_eff)

        # new_col/new_row sind gx/gy der Maschine (ohne Puffer)
        # Für die Randklemme müssen wir trotzdem gepufferten Platz sicherstellen:
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
    IndSwaps = [False for _ in pop]

    for g in range(1, int(generations) + 1):
        if config.STOP_REQUESTED:
            if progress_callback:
                try:
                    progress_callback(g, generations, best_score, best_ind, pop, False)
                except TypeError:
                    progress_callback(g, generations, best_score, best_ind)
            break

        scores = [fitness(ind) for ind in pop]
        paired = list(zip(scores, pop, IndSwaps))
        paired.sort(key=lambda p: p[0])

        EliteKeep = int(config.ELITE_KEEP)
        elites = [copy.deepcopy(p[1]) for p in paired[:EliteKeep]]
        elite_scores = [p[0] for p in paired[:EliteKeep]]

        SwapStateBestThisGen = bool(paired[0][2]) if paired else False
        ImprovedThisGen = elite_scores and elite_scores[0] < best_score
        SwapImproved = ImprovedThisGen and SwapStateBestThisGen

        if ImprovedThisGen:
            best_score = float(elite_scores[0])
            best_ind = copy.deepcopy(elites[0])

        new_pop: List[List[Dict]] = []
        NewSwaps: List[bool] = []
        new_pop.extend(copy.deepcopy(elites))
        NewSwaps.extend([False] * len(elites))
        while len(new_pop) < int(config.POPULATION_SIZE):
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = uniform_crossover(p1, p2)
            mutate(child)
            enforce_fixed(child)
            SwapsHappened = bool(tauschen(child, float(config.SWAP_PROB)))
            new_pop.append(child)
            NewSwaps.append(SwapsHappened)

        pop = new_pop
        IndSwaps = NewSwaps
        
        if progress_callback:
            progress_callback(g, generations, best_score, best_ind, pop, SwapImproved)

    #Hier habe ich Platz um die gruppierung von Maschinen nachbeeindigung zu machen
    """
    best_ind übergeben
    Maschinen Gruppieren abhängig vom Workflow (und nähe zueinander) Oder wenn man ganz verrückt ist nach ports und alle varianten werden getestet
    Testen ob durch verschieben, drehen, tauschen von maschinen innerhalb der Gruppe Kosten sinken
    GA mit den neuen Maschinen durchführen (run_ga) 
    """
    return best_ind, best_score
