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
    random_Group_Leader,
    is_fixed_machine,
    swap_grid_positions,
    MemberDict,
    rect_corners,
    distance_cost,
    Optimize_Groups,
)

def init_population() -> List[List[Dict]]:
    return [random_individual() for _ in range(int(config.POPULATION_SIZE))]

def init_group_population() -> List[List[Dict]]:
    return [random_Group_Leader() for _ in range(int(config.POPULATION_SIZE))]

def is_group_member(Machine: Dict) -> bool:
    idx = Machine.get("idx", None)
    groups = getattr(config, "GROUPS_FOR_GA", []) or []
    for group in groups:
        leader = group.get("Leader", None)
        members = group.get("Member", []) or []
        if idx in members and idx != leader:
            return True
    return False
    
def get_leader_from_member(Machine: Dict, ind: List[Dict]) -> Dict:
    groups = getattr(config, "GROUPS_FOR_GA", []) or []
    for group in groups:
        members = group.get("Member", []) or []
        if Machine["idx"] in members:
            return ind[group["Leader"]]
    return Machine

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

def enforce_group_members(ind: List[Dict]) -> None:
    groups = getattr(config, "GROUPS_FOR_GA", []) or []
    if not groups:
        return
    for group in groups:
        members = group.get("Member", []) or []
        if len(members) < 2: #Falls ein fehler beim erstellen der Member gemacht wurde 
            print(f"Gruppe hat < 2 member! Fehler bei der gruppierungsart: {config.GROUP_BY}")
        member_index = int(members[1])
        if not (0 <= member_index < len(ind)):
            continue
        if is_fixed_machine(ind[member_index]):
            continue
        ind[member_index] = MemberDict(group, ind)
 
def fitness(ind: List[Dict]) -> float:
#Fitness mit Maschinen Kollisionen Obstacles Bounds und Flow Geometrie
    cost = 0.0
    cost += distance_cost(ind, config)
    cell_owner: Dict[Tuple[int, int], int] = {}
    obstacles = config.OBSTACLES
    overlap_pen = float(config.OVERLAP_PENALTY)
    obstacle_pen = float(config.OBSTACLE_PENALTY)
    for i, m in enumerate(ind):
        cells = occupied_cells(m, Clearance = True)
        for c in cells:
            if c in cell_owner:
                cost += overlap_pen
            else:
                cell_owner[c] = i
            if c in obstacles:
                cost += obstacle_pen

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
        if config.GROUP_PHASE:
            if is_group_member(a[i]):
                child.append(copy.deepcopy(a[i]))
                continue
        child.append(copy.deepcopy(a[i] if random.random() < 0.5 else b[i])) 

    enforce_fixed(child)
    if config.GROUP_PHASE:
        enforce_group_members(child)
    normalize_individual(child)
    return child

def tauschen(ind: List[Dict], swap_prob: float) -> None:
    """Tauscht die koordianten von zwei Maschinen im Individuum"""
    if random.random() > swap_prob:
        return False
    i, j = random.sample(range(len(ind)), 2)  # garantiert unterschiedlich Maschinen
    m1, m2 = ind[i], ind[j]

    if config.GROUP_PHASE:
        if is_group_member(m1):
            m1 = get_leader_from_member(m1, ind)
        if is_group_member(m2):
            m2 = get_leader_from_member(m2, ind)

    #feste Maschinen dürfen nicht getauscht werden
    if is_fixed_machine(m1) or is_fixed_machine(m2):
        return False
    
    swap_grid_positions(m1, m2)
    print("Swap")
    if config.GROUP_PHASE:
        enforce_group_members(ind)
    return True

def _placement_ok(
    ind: List[Dict], 
    machine_idx: int, 
    cand: Dict,
    *,
    cand_cells: Optional[set[Tuple[int, int]]] = None,
    footprints: Optional[List[set[Tuple[int, int]]]] = None,
    ) -> bool:
    cand_cells = cand_cells if cand_cells is not None else occupied_cells(cand, Clearance=False)

    for c in cand_cells:
        if c in config.OBSTACLES:
            return False
        if c[0] < 0 or c[1] < 0 or c[0] >= config.GRID_COLS or c[1] >= config.GRID_ROWS:
            return False

    for j, other in enumerate(ind):
        if j == machine_idx:
            continue
        other_cells = footprints[j] if footprints is not None else occupied_cells(other, Clearance=False)
        if cand_cells & other_cells:
            return False
    return True

def mutate(ind: List[Dict]) -> None:
    """Mutation im Raster

    - Position und Rotation haben eigene Wahrscheinlichkeiten
    - Feste Maschinen werden ignoriert
    - In der Gruppen-Phase werden Member ignoriert
    - Ungültige Kandidaten werden verworfen
    """
    use_footprint = not bool(config.GROUP_PHASE)
    footprints: Optional[List[set[Tuple[int, int]]]] = None
    if use_footprint:
        footprints = [occupied_cells(m, Clearance = False) for m in ind]
    change = False

    pos_prob = float(config.MUTATION_PROB)
    rot_prob = float(getattr(config, "MUTATION_ROT_PROB", 0.0))

    for i, m in enumerate(ind):
        if is_fixed_machine(m):
            continue
        if config.GROUP_PHASE and is_group_member(m):
            continue

        do_pos = random.random() < pos_prob
        do_rot = random.random() < rot_prob
        if not (do_pos or do_rot):
            continue

        new_col = int(m["gx"])
        new_row = int(m["gy"])
        new_z = int(m.get("z", 0))

        if do_pos:
            new_col += random.randint(-config.MUTATION_POS_STD, config.MUTATION_POS_STD)
            new_row += random.randint(-config.MUTATION_POS_STD, config.MUTATION_POS_STD)

        if do_rot:
            new_z = new_z + 90 * random.choice([1,2,3]) % 360

        w_eff, h_eff = effective_dims(m, int(new_z))

        max_col = max(0, config.GRID_COLS - w_eff)
        max_row = max(0, config.GRID_ROWS - h_eff)
        # Für die Randklemme muss gepufferter Platz sichergestellt sein
        new_col = max(0, min(max_col, int(new_col)))
        new_row = max(0, min(max_row, int(new_row)))

        cand = dict(m)
        cand["z"] = int(new_z) % 360
        cand["gx"] = int(new_col)
        cand["gy"] = int(new_row)
        cand["x"], cand["y"] = cell_center_from_topleft(cand["gx"], cand["gy"], w_eff, h_eff)
        
        cand_cells = occupied_cells(cand, Clearance = False)
        if _placement_ok(ind, i, cand, cand_cells = cand_cells, footprints = footprints):
            ind[i] = cand
            change = True
        if footprints is not None:
            footprints[i] = cand_cells
        if config.GROUP_PHASE:
            enforce_group_members(ind)

    if config.GROUP_PHASE and change:
        enforce_group_members(ind)
    if change:
        normalize_individual(ind)

def teleport(ind: List[Dict]) -> None:
    """Mutation im Raster

    - Position und Rotation haben eigene Wahrscheinlichkeiten
    - Feste Maschinen werden ignoriert
    - In der Gruppen-Phase werden Member ignoriert
    - Ungültige Kandidaten werden verworfen
    """
    use_footprint = not bool(config.GROUP_PHASE)
    footprints: Optional[List[set[Tuple[int, int]]]] = None
    if use_footprint:
        footprints = [occupied_cells(m, Clearance = False) for m in ind]
    change = False

    tel_prob = float(config.TELEPORT_PROB)

    for i, m in enumerate(ind):
        if is_fixed_machine(m):
            continue
        if config.GROUP_PHASE and is_group_member(m):
            continue

        do_tel = random.random() < tel_prob
        if not do_tel:
            continue

        new_col = int(m["gx"])
        new_row = int(m["gy"])
        machine_z = int(m.get("z", 0))

        w_eff, h_eff = effective_dims(m, int(machine_z))
        max_col = max(0, config.GRID_COLS - w_eff)
        max_row = max(0, config.GRID_ROWS - h_eff)

        new_col = random.randint(0, max_col)
        new_row = random.randint(0, max_row)

        cand = dict(m)
        cand["z"] = int(machine_z)
        cand["gx"] = int(new_col)
        cand["gy"] = int(new_row)
        cand["x"], cand["y"] = cell_center_from_topleft(cand["gx"], cand["gy"], w_eff, h_eff)
        
        cand_cells = occupied_cells(cand, Clearance = False)
        if _placement_ok(ind, i, cand, cand_cells = cand_cells, footprints = footprints):
            ind[i] = cand
            change = True
        if footprints is not None:
            footprints[i] = cand_cells
        if config.GROUP_PHASE:
            enforce_group_members(ind)

    if config.GROUP_PHASE and change:
        enforce_group_members(ind)
    if change:
        normalize_individual(ind)

def dynamic_parametrisation():
    """Dynamische Anpassung der Parameter im verlauf des GAs, abhängig von der verbesserungsrate"""
    config.MUTATION_PROB
    config.MUTATION_ROT_PROB
    config.SWAP_PROB
    config.TELEPORT_PROB
    return

def run_ga(generations: int, progress_callback=None) -> Tuple[Optional[List[Dict]], float]:
    from helpers import update_grid_counts
    cross_prob = config.CROSSOVER_PROB
    def evolve(
        *,
        pop: List[List[Dict]],
        generations: int,
        progress_callback,
        best_ind: Optional[List[Dict]],
        best_score: float,
        stagnation_stop: bool,
        ) -> Tuple[Optional[List[Dict]], float, List[List[Dict]]]:
        ind_swaps: List[bool] = [False for _ in pop]
        stagnated: int = 0
    
        for g in range(1, int(generations) + 1):
            #teleport wahrscheinlichkeit kontinuierlich erhöhen
            config.TELEPORT_PROB += 1 / (generations * 10)

            if config.STOP_REQUESTED:
                if progress_callback:
                    try:
                        progress_callback(g, generations, best_score, best_ind, pop, False)
                    except TypeError:
                        progress_callback(g, generations, best_score, best_ind)
                break
            scores = [fitness(ind) for ind in pop]
            paired = list(zip(scores, pop, ind_swaps))
            paired.sort(key=lambda p: p[0])

            EliteKeep = int(config.ELITE_KEEP)
            elites = [copy.deepcopy(p[1]) for p in paired[:EliteKeep]]
            elite_scores = [p[0] for p in paired[:EliteKeep]]

            SwapStateBestThisGen = bool(paired[0][2]) if paired else False
            ImprovedThisGen = bool(elite_scores) and elite_scores[0] < best_score
            SwapImproved = ImprovedThisGen and SwapStateBestThisGen
            old_best_score = best_score
            if ImprovedThisGen:
                print("verbessert")
                best_score = float(elite_scores[0])
                best_ind = copy.deepcopy(elites[0])
                stagnated = 0
            elif stagnation_stop and old_best_score == best_score:
                stagnated += 1
            else:
                stagnated = 0
            if  stagnation_stop and stagnated >= 50:
                return best_ind, best_score, pop
            
            new_pop: List[List[Dict]] = []
            NewSwaps: List[bool] = []
            new_pop.extend(elites)
            NewSwaps.extend([False] * len(elites))

            while len(new_pop) < int(config.POPULATION_SIZE):
                if random.random() > cross_prob:
                    child = random_Group_Leader() if config.GROUP_PHASE else random_individual()
                    if config.GROUP_PHASE:
                        enforce_group_members(child)
                else:
                    p1 = random.choice(elites)
                    p2 = random.choice(elites)
                    child = uniform_crossover(p1, p2)
                mutate(child)
                teleport(child)
                enforce_fixed(child)
                SwapsHappened = bool(tauschen(child, float(config.SWAP_PROB)))
                new_pop.append(child)
                NewSwaps.append(SwapsHappened)

            pop = new_pop
            ind_swaps = NewSwaps
            
            if progress_callback:
                progress_callback(g, generations, best_score, best_ind, pop, SwapImproved)
        return best_ind, best_score, pop
    
    config.GROUP_PHASE = False
    config.GROUPS_FOR_GA = []
    config.DEBUG_GROUP_ASSERT = True

    update_grid_counts()

    pop = init_population()
    best_ind: Optional[List[Dict]]  = None
    best_score = float("inf")

    best_ind, best_score, pop = evolve(
        pop = pop,
        generations = generations,
        progress_callback = progress_callback,
        best_ind = best_ind,
        best_score = best_score,
        stagnation_stop = True
    )
    if config.CREATE_GROUPS and best_ind:
        config.GROUP_PHASE = True
        best_ind_copy = copy.deepcopy(best_ind)
        config.GROUPS_FOR_GA = Optimize_Groups(best_ind_copy)

        group_pop = init_group_population()
        best_ind, best_score, _ = evolve(
            pop = group_pop,
            generations = generations,
            progress_callback = progress_callback,
            best_ind = None,
            best_score = float("inf"),
            stagnation_stop = True
        )        
    return best_ind, best_score