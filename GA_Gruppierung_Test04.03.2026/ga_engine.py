# GENETIC ALGORITHM MODULE

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple, Sequence

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

    floor_w = float(config.FLOOR_W)
    floor_h = float(config.FLOOR_H)
    grid_size = float(config.GRID_SIZE)
    out_pen = float(config.OUT_OF_BOUNDS_PENALTY)
    for m in ind:
        z = int(m.get("z", 0))
        w_eff, h_eff = effective_dims(m, z)
        machine_width = float(w_eff) * grid_size
        machine_height = float(h_eff) * grid_size
        centerX = float(m["x"])
        centerY = float(m["y"])
        left = centerX - (machine_width / 2.0)
        right = centerX + (machine_width/ 2.0)
        bottom = centerY - (machine_height / 2.0)
        top = centerY + (machine_height / 2.0)
        out_count = 0
        for x, y in ((left, bottom), (left, top), (right, bottom), (right, top)):
            if x < 0.0 or x > floor_w or y < 0.0 or y > floor_h:
                out_count += 1
        if out_count:
            cost += out_pen * float(out_count)
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
            # new_col += random.randint(-1, 1)
            # new_row += random.randint(-1, 1)

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

    if config.GROUP_PHASE and change:
        enforce_group_members(ind)
    if change:
        normalize_individual(ind)

#========================================================================================================
#========================================================================================================

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _acro_diversity(pop: Sequence[List[Dict]], mode: str = "all", *, sample_pairs: int = 256) -> float:
    """
    Diversity-Proxy: mittlere normalisierte Paar-Distanz (L1).
    mode:
      - "all": gx, gy, z
      - "pos": gx, gy
      - "rot": z
    """
    n = len(pop)
    if n < 2:
        return 0.0

    cols = max(2, int(getattr(config, "GRID_COLS", 2)))
    rows = max(2, int(getattr(config, "GRID_ROWS", 2)))
    gx_den = float(cols - 1)
    gy_den = float(rows - 1)

    genes: List[List[float]] = []
    for ind in pop:
        v: List[float] = []
        for m in ind:
            if mode in ("all", "pos"):
                gx = float(m.get("gx", 0))
                gy = float(m.get("gy", 0))
                v.append(_clamp01(gx / gx_den))
                v.append(_clamp01(gy / gy_den))
            if mode in ("all", "rot"):
                z = int(m.get("z", 0)) % 360
                z_idx = float((z // 90) % 4)
                v.append(_clamp01(z_idx / 3.0))
        genes.append(v)

    d = len(genes[0]) if genes else 0
    if d == 0:
        return 0.0

    total_pairs = n * (n - 1) // 2
    used = min(int(sample_pairs), int(total_pairs))
    if used <= 0:
        return 0.0

    acc = 0.0
    for _ in range(used):
        i = random.randrange(0, n)
        j = random.randrange(0, n - 1)
        if j >= i:
            j += 1
        gi, gj = genes[i], genes[j]
        acc += sum(abs(gi[k] - gj[k]) for k in range(d)) / float(d)

    return acc / float(used)


def _smooth_value(old: float, target: float, alpha: float = 0.25) -> float:
    """Dämpft Parametersprünge zwischen zwei Generationen."""
    return float(old) + float(alpha) * (float(target) - float(old))


def _quality_update_params(
    pop: Sequence[List[Dict]],
    scores: Sequence[float],
    generation: int,
    max_generations: int,
    stagnated: int,
) -> float:
    """Qualitätsorientierte dynamische Parametrisierung.

    Grundidee:
    - Gute Layoutstrukturen werden geschützt. Deshalb bleiben Rotation und
      Teleportation niedrig.
    - Die Positionsmutation wird im Verlauf abgekühlt und nur bei echter
      Stagnation leicht angehoben.
    - Teleportation ist kein Standardoperator, sondern ein seltener
      Diversitätsimpuls bei niedriger Positionsdiversität und Stagnation.
    """
    if not pop or not scores:
        return float(getattr(config, "CROSSOVER_PROB", 0.85))

    progress = _clamp01((int(generation) - 1) / max(int(max_generations) - 1, 1))
    exploration = 1.0 - progress

    spd_all = _acro_diversity(pop, "all")
    spd_pos = _acro_diversity(pop, "pos")
    spd_rot = _acro_diversity(pop, "rot")

    # Für Layoutprobleme ist Positionsdiversität wichtiger als reine Rotationsdiversität.
    low_pos_div = _clamp01((0.22 - spd_pos) / 0.22)
    low_all_div = _clamp01((0.24 - spd_all) / 0.24)
    low_rot_div = _clamp01((0.18 - spd_rot) / 0.18)
    stagnation_factor = _clamp01(float(stagnated) / 25.0)

    base_pc = float(getattr(config, "BASE_CROSSOVER_PROB", 0.85))
    base_pm = float(getattr(config, "BASE_MUTATION_PROB", 0.12))
    base_rot = float(getattr(config, "BASE_MUTATION_ROT_PROB", 0.04))
    base_swap = float(getattr(config, "BASE_SWAP_PROB", 0.04))

    # Crossover bleibt hoch, aber nicht maximal. Bei sehr geringer Diversität
    # wird er minimal gesenkt, damit nicht nur ähnliche Eltern rekombiniert werden.
    pc_target = base_pc + 0.06 * exploration - 0.04 * low_all_div
    pc_target = max(0.72, min(0.92, pc_target))

    # Positionsmutation: am Anfang moderat, später klein. Bei Stagnation und
    # geringer Positionsdiversität wird sie leicht angehoben, aber begrenzt.
    pm_target = (
        0.04
        + 0.08 * exploration
        + 0.04 * low_pos_div
        + 0.04 * stagnation_factor
    )
    pm_target = max(0.03, min(0.18, max(pm_target, min(base_pm, 0.12))))

    # Rotation ist stark destruktiv für Maschinenlayouts. Deshalb deutlich
    # niedriger als Positionsmutation und nur leicht reaktiv.
    rot_target = (
        0.005
        + 0.025 * exploration
        + 0.015 * low_rot_div
        + 0.010 * stagnation_factor
    )
    rot_target = max(0.005, min(0.06, min(rot_target, max(base_rot, 0.02))))

    # Schrittweite: frühe Suche darf gröber sein, später wird lokal optimiert.
    min_dim = max(1, min(int(config.GRID_COLS), int(config.GRID_ROWS)))
    std_max_start = min(max(2, round(min_dim * 0.12)), 6)
    std_target = 1 + int(round(exploration * (std_max_start - 1)))
    if stagnation_factor > 0.55 and low_pos_div > 0.50:
        std_target = min(std_target + 1, std_max_start)
    std_target = max(1, int(std_target))

    # Swap kann bei Layoutproblemen nützlich sein, darf aber nicht dauernd
    # komplette Nachbarschaften zerreißen.
    swap_target = base_swap + 0.03 * stagnation_factor * low_pos_div
    swap_target = max(0.01, min(0.08, swap_target))

    # Teleportation nur als Notfallimpuls. Wichtig: Die Wahrscheinlichkeit
    # wirkt pro Maschine, deshalb sind selbst 0.02 bereits deutlich spürbar.
    if stagnated >= 12 and spd_pos < 0.16:
        teleport_target = 0.004 + 0.016 * stagnation_factor * low_pos_div
    else:
        teleport_target = 0.0
    teleport_target = max(0.0, min(0.02, teleport_target))

    config.CROSSOVER_PROB = _clamp01(_smooth_value(float(config.CROSSOVER_PROB), pc_target, 0.25))
    config.MUTATION_PROB = _clamp01(_smooth_value(float(config.MUTATION_PROB), pm_target, 0.25))
    config.MUTATION_ROT_PROB = _clamp01(_smooth_value(float(config.MUTATION_ROT_PROB), rot_target, 0.25))
    config.SWAP_PROB = _clamp01(_smooth_value(float(config.SWAP_PROB), swap_target, 0.20))
    config.TELEPORT_PROB = _clamp01(_smooth_value(float(config.TELEPORT_PROB), teleport_target, 0.20))
    config.MUTATION_POS_STD = int(round(_smooth_value(float(config.MUTATION_POS_STD), float(std_target), 0.35)))
    config.MUTATION_POS_STD = max(1, int(config.MUTATION_POS_STD))

    print(
        f"Adaptive Update: Gen={generation}/{max_generations}, progress={progress:.3f}, "
        f"Pc={config.CROSSOVER_PROB:.3f}, PM={config.MUTATION_PROB:.3f}, "
        f"PM_rot={config.MUTATION_ROT_PROB:.3f}, MUT_STD={config.MUTATION_POS_STD}, "
        f"SWAP={config.SWAP_PROB:.3f}, TP={config.TELEPORT_PROB:.3f}, "
        f"stagnated={stagnated}, SPD_all={spd_all:.3f}, "
        f"SPD_pos={spd_pos:.3f}, SPD_rot={spd_rot:.3f}"
    )

    return float(config.CROSSOVER_PROB)

#========================================================================================================



#=======================================Tournament Selection============================================
def _tournament_select(
    pop: Sequence[List[Dict]],
    scores: Sequence[float],
    *,
    tournament_size: int = 3,
) -> List[Dict]:
    """Wählt ein Elternindividuum über Tournament Selection aus."""
    n = len(pop)
    if n == 0:
        raise ValueError("Tournament Selection benötigt eine nicht leere Population")

    k = max(1, min(int(tournament_size), n))
    candidates = random.sample(range(n), k)
    best_idx = min(candidates, key=lambda i: float(scores[i]))
    return copy.deepcopy(pop[best_idx])


def _dynamic_tournament_size(generation: int, max_generations: int) -> int:
    """Erhöht den Selektionsdruck leicht im Verlauf der Optimierung."""
    progress = _clamp01((int(generation) - 1) / max(int(max_generations) - 1, 1))
    if progress < 0.50:
        return 2
    if progress < 0.85:
        return 3
    return 4
#========================================================================================================



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
        nonlocal cross_prob
        ind_swaps: List[bool] = [False for _ in pop]
        stagnated: int = 0
    
        for g in range(1, int(generations) + 1):
            if config.STOP_REQUESTED:
                if progress_callback:
                    try:
                        progress_callback(g, generations, best_score, best_ind, pop, False)
                    except TypeError:
                        progress_callback(g, generations, best_score, best_ind)
                break
            scores = [fitness(ind) for ind in pop] #10
            cross_prob = _quality_update_params(pop, scores, g, generations, stagnated)
            paired = list(zip(scores, pop, ind_swaps))
            paired.sort(key=lambda p: p[0])


            #==========================================================================
            # Elternauswahl über Tournament Selection. Das beste Individuum wird zusätzlich
            # einmal direkt übernommen, damit die bisher beste Lösung nicht verloren geht.

            best_this_gen_score = float(paired[0][0])
            best_this_gen_ind = paired[0][1]
            tournament_size = _dynamic_tournament_size(g, generations)

            #==========================================================================


            SwapStateBestThisGen = bool(paired[0][2]) if paired else False
            ImprovedThisGen = best_this_gen_score < best_score
            SwapImproved = ImprovedThisGen and SwapStateBestThisGen
            old_best_score = best_score
            if ImprovedThisGen:
                best_score = float(best_this_gen_score)
                best_ind = copy.deepcopy(best_this_gen_ind)
                stagnated = 0
            elif stagnation_stop and old_best_score == best_score:
                stagnated += 1
            else:
                stagnated = 0
            if  stagnation_stop and stagnated >= 50:
                return best_ind, best_score, pop
            
            new_pop: List[List[Dict]] = []
            NewSwaps: List[bool] = []
            # Qualitätsorientierter Elitismus: mehrere gute Individuen werden
            # unverändert übernommen und nicht mutiert, rotiert oder teleportiert.
            elite_count = max(1, min(int(getattr(config, "ELITE_KEEP", 1)), int(config.POPULATION_SIZE), len(paired)))
            for _, elite_ind, _ in paired[:elite_count]:
                new_pop.append(copy.deepcopy(elite_ind))
                NewSwaps.append(False)

            while len(new_pop) < int(config.POPULATION_SIZE):
                if random.random() > cross_prob:
                    child = random_Group_Leader() if config.GROUP_PHASE else random_individual()
                    if config.GROUP_PHASE:
                        enforce_group_members(child)
                else:
                    p1 = _tournament_select(pop, scores, tournament_size=tournament_size)
                    p2 = _tournament_select(pop, scores, tournament_size=tournament_size)
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

    update_grid_counts()

    # GA-Lauf immer aus den Basiswerten der UI/Konfiguration starten.
    config.CROSSOVER_PROB = float(getattr(config, "BASE_CROSSOVER_PROB", config.CROSSOVER_PROB))
    config.MUTATION_PROB = float(getattr(config, "BASE_MUTATION_PROB", config.MUTATION_PROB))
    config.MUTATION_ROT_PROB = float(getattr(config, "BASE_MUTATION_ROT_PROB", config.MUTATION_ROT_PROB))
    config.MUTATION_POS_STD = int(getattr(config, "BASE_MUTATION_POS_STD", config.MUTATION_POS_STD))
    config.SWAP_PROB = float(getattr(config, "BASE_SWAP_PROB", config.SWAP_PROB))
    config.TELEPORT_PROB = float(getattr(config, "BASE_TELEPORT_PROB", 0.0))

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