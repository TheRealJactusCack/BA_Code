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
    compute_routed_edges,
    worker_clearance_cells,
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
    """Fitness mit harten Constraints (Maschinen/Obstacles/Worker-Clearance/Worker-Weg-Breite)."""
    hard = float(getattr(config, "HARD_CONSTRAINT_PENALTY", 1e9))
    cost = 0.0

    routed = compute_routed_edges(ind)
    cost += distance_cost(ind, config, routed=routed)

    obstacle_cells = set(getattr(config, "OBSTACLES", set()) or set())

    footprints: List[set[Tuple[int, int]]] = [occupied_cells(m, pad_cells=0) for m in ind]
    clearances: List[set[Tuple[int, int]]] = [worker_clearance_cells(m) for m in ind]

    # Maschine vs Maschine (Footprint)
    cell_owner: dict[Tuple[int, int], int] = {}
    for i, fp in enumerate(footprints):
        for c in fp:
            if c in cell_owner:
                cost += hard
            else:
                cell_owner[c] = i

    # Maschine vs Obstacles + Bounds
    for fp in footprints:
        for c in fp:
            if c[0] < 0 or c[1] < 0 or c[0] >= int(config.GRID_COLS) or c[1] >= int(config.GRID_ROWS):
                cost += hard
            elif c in obstacle_cells:
                cost += float(getattr(config, "OBSTACLE_PENALTY", hard))  # behält alte Skala

    # Worker-Clearance Constraints:
    # - Maschinen dürfen nicht auf fremder Worker-Clearance stehen
    # - Worker-Clearance darf nicht in Maschinen/Obstacles/Beyond liegen
    for i, fp in enumerate(footprints):
        for j, cl in enumerate(clearances):
            if i == j:
                continue
            if fp & cl:
                cost += hard

    for i, cl in enumerate(clearances):
        for c in cl:
            if c[0] < 0 or c[1] < 0 or c[0] >= int(config.GRID_COLS) or c[1] >= int(config.GRID_ROWS):
                cost += hard
            elif c in obstacle_cells:
                cost += hard
            elif c in footprints[i]:
                # sollte durch worker_clearance_cells() schon ausgenommen sein, aber sicher ist sicher
                cost += hard

        # kein anderes Maschinen-Footprint in der Clearance
        for j, fp in enumerate(footprints):
            if i == j:
                continue
            if cl & fp:
                cost += hard

    # Worker-Weg Breite (1m) als Clearance: keine Maschinen dürfen diese Zellen belegen.
    worker_corridor: set[Tuple[int, int]] = set()
    for e in routed.get("worker", []):
        cc = e.get("clearance_cells", None)
        if isinstance(cc, set):
            worker_corridor |= cc

    if worker_corridor:
        for fp in footprints:
            if fp & worker_corridor:
                cost += hard

    # Out-of-bounds Penalty via Polygon corners (bestehende Logik)
    for m in ind:
        w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
        h_m = float(m["h_cells"]) * float(config.GRID_SIZE)
        poly = rect_corners((float(m["x"]), float(m["y"])), w_m, h_m, int(m.get("z", 0)))
        out_count = 0
        for x, y in poly:
            if x < 0.0 or x > float(config.FLOOR_W) or y < 0.0 or y > float(config.FLOOR_H):
                out_count += 1
        if out_count:
            cost += float(getattr(config, "OUT_OF_BOUNDS_PENALTY", hard)) * float(out_count)

    return float(cost)

def uniform_crossover(a: List[Dict], b: List[Dict]) -> List[Dict]:
    child = []
    n = int(config.MACHINE_COUNT)
    for i in range(n):
        child.append(copy.deepcopy(a[i] if random.random() < 0.5 else b[i]))
        enforce_fixed(child)
    normalize_individual(child)
    return child

def tauschen(ind: List[Dict], swap_prob: float) -> None:
    """Tauscht die Koordinaten (gx/gy) von zwei Maschinen, wenn danach alles gültig bleibt."""
    if random.random() > float(swap_prob):
        return

    i, j = random.sample(range(len(ind)), 2)
    m1, m2 = ind[i], ind[j]

    if is_fixed_machine(m1) or is_fixed_machine(m2):
        return

    base1 = copy.deepcopy(m1)
    base2 = copy.deepcopy(m2)

    m1["gx"], m2["gx"] = int(base2["gx"]), int(base1["gx"])
    m1["gy"], m2["gy"] = int(base2["gy"]), int(base1["gy"])

    w1, h1 = effective_dims(m1, int(m1.get("z", 0)))
    w2, h2 = effective_dims(m2, int(m2.get("z", 0)))

    m1["x"], m1["y"] = cell_center_from_topleft(int(m1["gx"]), int(m1["gy"]), int(w1), int(h1))
    m2["x"], m2["y"] = cell_center_from_topleft(int(m2["gx"]), int(m2["gy"]), int(w2), int(h2))

    if not (_placement_ok(ind, i, m1) and _placement_ok(ind, j, m2)):
        ind[i] = base1
        ind[j] = base2

def _placement_ok(ind: List[Dict], machine_idx: int, cand: Dict) -> bool:
    """Hard-constraint Check für Mutation/Swap.

    Regeln:
        - Maschinen-Footprints dürfen sich nicht überlappen.
        - Maschinen-Footprints dürfen nicht in Obstacles oder in Worker-Clearance liegen.
        - Worker-Clearance (cand) darf nicht in Obstacles oder in andere Maschinen-Footprints liegen.
        - Worker-Clearance darf andere Worker-Clearance überlappen.
    """
    obstacle_cells = set(getattr(config, "OBSTACLES", set()) or set())

    cand_fp = occupied_cells(cand, pad_cells=0)
    for c in cand_fp:
        if c[0] < 0 or c[1] < 0 or c[0] >= int(config.GRID_COLS) or c[1] >= int(config.GRID_ROWS):
            return False
        if c in obstacle_cells:
            return False

    cand_cl = worker_clearance_cells(cand)
    for c in cand_cl:
        if c[0] < 0 or c[1] < 0 or c[0] >= int(config.GRID_COLS) or c[1] >= int(config.GRID_ROWS):
            return False
        if c in obstacle_cells:
            return False

    for j, other in enumerate(ind):
        if j == machine_idx:
            continue
        other_fp = occupied_cells(other, pad_cells=0)
        if cand_fp & other_fp:
            return False

        other_cl = worker_clearance_cells(other)
        # cand darf nicht auf fremder Worker-Clearance stehen
        if cand_fp & other_cl:
            return False
        # cand-Worker-Clearance darf keine andere Maschine blocken
        if cand_cl & other_fp:
            return False

    return True

def mutate(ind: List[Dict]) -> None:
    """Mutation im Raster (Position + Rotation).

    Constraints werden in _placement_ok geprüft.
    """
    pos_prob = float(getattr(config, "MUTATION_PROB", 0.0))
    rot_prob = float(getattr(config, "MUTATION_ROT_PROB", 0.0))

    for i, m in enumerate(ind):
        if is_fixed_machine(m):
            continue

        base = copy.deepcopy(m)
        changed = False

        new_col = int(m.get("gx", 0))
        new_row = int(m.get("gy", 0))
        new_z = int(m.get("z", 0))

        if random.random() < pos_prob:
            delta_col = int(round(random.uniform(-float(config.MUTATION_POS_STD), float(config.MUTATION_POS_STD))))
            delta_row = int(round(random.uniform(-float(config.MUTATION_POS_STD), float(config.MUTATION_POS_STD))))
            new_col += delta_col
            new_row += delta_row
            changed = True

        if random.random() < rot_prob:
            new_z = int(new_z + 90 * random.choice([1, 2, 3])) % 360
            changed = True

        if not changed:
            continue

        new_z = int(new_z) % 360

        w_eff, h_eff = effective_dims(m, new_z)
        max_col = max(0, int(config.GRID_COLS) - int(w_eff))
        max_row = max(0, int(config.GRID_ROWS) - int(h_eff))
        new_col = max(0, min(max_col, int(new_col)))
        new_row = max(0, min(max_row, int(new_row)))

        cand = copy.deepcopy(m)
        cand["z"] = int(new_z)
        cand["gx"] = int(new_col)
        cand["gy"] = int(new_row)
        cand["x"], cand["y"] = cell_center_from_topleft(cand["gx"], cand["gy"], int(w_eff), int(h_eff))

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
            score_differenz = (best_score - elite_scores[0]) / best_score * 100
            best_score = float(elite_scores[0])
            best_ind = copy.deepcopy(elites[0])

            last_improved_gen = int(g)


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
            enforce_fixed(child)
            new_pop.append(child)

        for ind in new_pop:                     # new_pop: List[List[Dict]]
            tauschen(ind, config.SWAP_PROB)     # pro Individuum 10% Chance
        
        pop = new_pop
    #Hier habe ich Platz um die gruppierung von Maschinen nachbeeindigung zu machen
    """
    best_ind übergeben
    Maschinen Gruppieren abhängig vom Workflow (und nähe zueinander) Oder wenn man ganz verrückt ist nach ports und alle varianten werden getestet
    Testen ob durch verschieben, drehen, tauschen von maschinen innerhalb der Gruppe Kosten sinken
    GA mit den neuen Maschinen durchführen (run_ga) 
    """
    return best_ind, best_score
