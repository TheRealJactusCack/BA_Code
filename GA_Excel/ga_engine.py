# -------------------------
# GENETIC ALGORITHM MODULE
# -------------------------
# Enthält Fitness-Funktion, genetische Operatoren und GA-Engine

import random
import copy
import math
import traceback
import config
from helpers import (
    occupied_cells, rect_corners, normalize_individual, 
    cell_center_from_topleft, effective_dims, random_individual, random_machine_nonoverlap,
    machine_output_point, machine_input_point
)
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


def init_population():
    """Erstellt die Startpopulation mit POPULATION_SIZE Individuen."""
    return [random_individual() for _ in range(config.POPULATION_SIZE)]


def fitness(ind):
    """
    Kostenschätzer (niedriger ist besser):
    - Distanzkosten zwischen aufeinanderfolgenden Maschinen
    - Strafe für Überschneidungen
    - Strafe für Out-of-bounds
    - Strafe für Hindernisse
    - Distanzkosten zu Entry/Exit
    """
    from helpers import cell_center_from_topleft
    
    cost = 0.0

    # 1) Distanzkosten entlang der Kette: Manhattan (L1) zwischen Ausgang(A) und Eingang(B)
    n = min(config.MACHINE_COUNT, len(ind))
    # compute grid-based disjoint Manhattan paths between consecutive machines
    from helpers import find_manhattan_path, get_terminal_cell, cells_to_world
    # build set of blocked cells: obstacles + machine footprints
    blocked = set(config.OBSTACLES)
    machine_cells_all = []
    for m in ind:
        for c in occupied_cells(m):
            blocked.add(c)
        machine_cells_all.append(set(occupied_cells(m)))

    flow_used = set()
    for i in range(n - 1):
        a = ind[i]
        b = ind[i+1]
        start_cell = get_terminal_cell(a, kind='out')
        goal_cell = get_terminal_cell(b, kind='in')
        # allow start/goal even if they are adjacent to machines
        local_blocked = set(blocked)
        # remove start and goal from blocked to allow stepping into them
        if start_cell in local_blocked:
            local_blocked.remove(start_cell)
        if goal_cell in local_blocked:
            local_blocked.remove(goal_cell)
        # also block already used flow cells to prevent crossings
        local_blocked |= flow_used
        path = find_manhattan_path(start_cell, goal_cell, blocked_cells=local_blocked)
        if path is None:
            # heavy penalty when no non-crossing path exists
            cost += config.OUT_OF_BOUNDS_PENALTY * 5
        else:
            # add path length (in meters) to cost: (steps)*GRID_SIZE
            path_len_m = (len(path) - 1) * config.GRID_SIZE
            cost += config.DIST_SCALE * path_len_m
            # mark flow cells as used (except goal to allow next flow to end there)
            for cell in path[:-1]:
                flow_used.add(cell)

    # 2) Überlappungsstrafe über Zellen
    cell_owner = {}
    for i, m in enumerate(ind):
        cells = occupied_cells(m)
        for c in cells:
            if c in cell_owner:
                cost += config.OVERLAP_PENALTY
            else:
                cell_owner[c] = i

    # 2b) Hindernisse
    for i, m in enumerate(ind):
        cells = occupied_cells(m)
        for c in cells:
            if c in config.OBSTACLES:
                cost += config.OBSTACLE_PENALTY

    # 3) Out-of-bounds-Prüfung über reale Eckpunkte
    for m in ind:
        w = m['w_cells'] * config.GRID_SIZE
        h = m['h_cells'] * config.GRID_SIZE
        poly = rect_corners((m['x'], m['y']), w, h, m['z'])
        out_count = 0
        for (x, y) in poly:
            if x < 0.0 or x > config.FLOOR_W or y < 0.0 or y > config.FLOOR_H:
                out_count += 1
        if out_count > 0:
            cost += config.OUT_OF_BOUNDS_PENALTY * out_count

    # 4) Distanz zu Entry/Exit (Entry -> input(first), output(last) -> Exit)
    if ind:
        first = ind[0]
        last = ind[-1]
        entry_x, entry_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
        exit_x, exit_y = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
        in_x, in_y = machine_input_point(first)
        out_x, out_y = machine_output_point(last)
        cost += config.DIST_SCALE * (abs(in_x - entry_x) + abs(in_y - entry_y))
        cost += config.DIST_SCALE * (abs(out_x - exit_x) + abs(out_y - exit_y))

    return cost


def tournament_selection(pop, scores, k=3):
    """Turnierselektion auf der aktuellen Population."""
    selected_idx = random.sample(range(len(pop)), k)
    best = selected_idx[0]
    for idx in selected_idx[1:]:
        if scores[idx] < scores[best]:
            best = idx
    return copy.deepcopy(pop[best])


def uniform_crossover(a, b):
    """Uniform Crossover: pro Maschinenindex von Parent A oder B übernehmen."""
    child = []
    for i in range(config.MACHINE_COUNT):
        if random.random() < 0.5:
            child.append(copy.deepcopy(a[i]))
        else:
            child.append(copy.deepcopy(b[i]))
    normalize_individual(child)
    return child


def mutate(ind):
    """
    Mutation im Raster:
    - Verschiebe Maschine in Zellen (Gauss, gerundet) innerhalb erlaubter Top-Left-Range
    - Ändere Rotation auf eines der 4 Werte
    """
    def _snap_rotation(z_candidate: float) -> int:
        allowed = (0, 90, 180, 270)

        def _dist(a: float, b: float) -> float:
            return abs(((a - b + 180.0) % 360.0) - 180.0)

        return int(min(allowed, key=lambda a: _dist(z_candidate, float(a))))

    # use runtime values from config
    for i, m in enumerate(ind):
        if random.random() < config.MUTATION_PROB:
            delta_col = int(round(random.gauss(0, config.MUTATION_POS_STD)))
            delta_row = int(round(random.gauss(0, config.MUTATION_POS_STD)))
            new_col = int(m['gx']) + delta_col
            new_row = int(m['gy']) + delta_row

            # Rotation (`z`) ist nicht veränderbar durch die GA (Eingang/Ausgang bleiben fix)
            if random.random() < config.MUTATION_PROB and float(getattr(config, "MUTATION_ANGLE_STD", 0.0)) > 0.0:
                z0 = float(m.get("z", 0.0))
                z_candidate = (z0 + random.gauss(0.0, float(config.MUTATION_ANGLE_STD))) % 360.0
                m["z"] = _snap_rotation(z_candidate)


            # always compute effective dims (after possible rotation change)
            w_eff, h_eff = effective_dims(m)
            max_col = max(0, config.GRID_COLS - int(w_eff))
            max_row = max(0, config.GRID_ROWS - int(h_eff))
            new_col = max(0, min(max_col, new_col))
            new_row = max(0, min(max_row, new_row))
            m['gx'] = int(new_col)
            m['gy'] = int(new_row)
            m['x'], m['y'] = cell_center_from_topleft(m['gx'], m['gy'], w_eff, h_eff)
    normalize_individual(ind)


def run_ga(generations, progress_callback=None):
    """
    Hauptfunktion des Genetischen Algorithmus.
    Input:  generations: Anzahl der durchzulaufenden Generationen
            progress_callback: Optional, wird aufgerufen mit (g, generations, best_score, best_ind, population)
    Output: (best_ind, best_score)
    """
    import config
    # Update grid counts
    from helpers import update_grid_counts
    update_grid_counts()
    
    pop = init_population()
    best_ind = None
    best_score = float('inf')

    for g in range(1, generations + 1):
        if config.STOP_REQUESTED:
            if progress_callback:
                try:
                    progress_callback(g, generations, best_score, best_ind, pop)
                except TypeError:
                    progress_callback(g, generations, best_score, best_ind)
            break

        # Bewertung
        scores = [fitness(ind) for ind in pop]
        """print(config.GRID_COLS, config.GRID_ROWS)"""
        # Sortiere Population nach Score
        paired = list(zip(scores, pop))
        paired.sort(key=lambda p: p[0])
        elites = [copy.deepcopy(p[1]) for p in paired[:config.ELITE_KEEP]]
        elite_scores = [p[0] for p in paired[:config.ELITE_KEEP]]

        # Update global best
        if elite_scores and elite_scores[0] < best_score:
            best_score = elite_scores[0]
            best_ind = copy.deepcopy(elites[0])

        if progress_callback:
            try:
                progress_callback(g, generations, best_score, best_ind, pop)
            except TypeError:
                progress_callback(g, generations, best_score, best_ind)

        # Erzeugung neuer Individuen
        new_pop = []
        new_pop.extend(copy.deepcopy(elites))

        while len(new_pop) < config.POPULATION_SIZE:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = uniform_crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        pop = new_pop

    return best_ind, best_score


# -------------------------
# KLASSE: GA Engine
# -------------------------
class GAEngine:
    """Schrittweise Iteration des GA (für interaktive Steuerung)."""
    
    def __init__(self, total_generations):
        from helpers import update_grid_counts
        update_grid_counts()
        self.total_generations = int(total_generations)
        self.generation = 0
        self.population = init_population()
        self.best_ind = None
        self.best_score = float('inf')

    def step(self):
        """Ein Schritt: bewerten, Eliten wählen, neue Population bilden, generation++"""
        import config
        
        if config.STOP_REQUESTED or self.generation >= self.total_generations:
            return self.best_score, self.best_ind

        # Bewertung
        scores = [fitness(ind) for ind in self.population]
        paired = list(zip(scores, self.population))
        paired.sort(key=lambda p: p[0])
        elites = [copy.deepcopy(p[1]) for p in paired[:config.ELITE_KEEP]]
        elite_scores = [p[0] for p in paired[:config.ELITE_KEEP]]

        if elite_scores and elite_scores[0] < self.best_score:
            self.best_score = elite_scores[0]
            self.best_ind = copy.deepcopy(elites[0])

        # neue Population bauen
        new_pop = []
        new_pop.extend(copy.deepcopy(elites))
        while len(new_pop) < config.POPULATION_SIZE:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = uniform_crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        return self.best_score, self.best_ind


# -------------------------
# KLASSE: GA Worker (Background Thread)
# -------------------------
class GAWorker(QObject):
    """Background worker that runs the GA in a QThread and emits progress signals."""
    progress = pyqtSignal(int, int, float, object, list)
    finished = pyqtSignal(object, float)

    def __init__(self, generations):
        super().__init__()
        self.generations = int(generations)
        self.config = None

    @pyqtSlot()
    def run(self):
        import config
        config.STOP_REQUESTED = False
        
        # Speichere aktuelle Konfiguration
        saved_globals = {}
        try:
            if self.config:
                names = ['POPULATION_SIZE','ELITE_KEEP','MACHINE_COUNT','MACHINE_SIZES','MUTATION_PROB',
                         'MUTATION_POS_STD','MUTATION_ANGLE_STD','FLOOR_W','FLOOR_H','GRID_COLS','GRID_ROWS',
                         'OBSTACLES','ENTRY_CELL','EXIT_CELL','GRID_SIZE']
                for n in names:
                    saved_globals[n] = getattr(config, n, None)
                    if n in self.config:
                        setattr(config, n, copy.deepcopy(self.config[n]))
        except Exception:
            pass

        def _cb(g, total, best_score_cb, best_ind_cb, population=None):
            try:
                self.progress.emit(g, total, best_score_cb, best_ind_cb, population if population is not None else [])
            except Exception:
                pass

        best_ind = None
        best_score = float('inf')
        try:
            best_ind, best_score = run_ga(self.generations, progress_callback=_cb)
        except Exception as e:
            try:
                traceback.print_exc()
            except Exception:
                pass
            try:
                self.finished.emit(None, float('inf'))
            except Exception:
                pass
        finally:
            # Restore saved globals
            try:
                for k, v in saved_globals.items():
                    if v is not None:
                        setattr(config, k, v)
            except Exception:
                pass
            # Ensure finished emitted
            try:
                self.finished.emit(best_ind, best_score)
            except Exception:
                pass

    @pyqtSlot()
    def request_stop(self):
        import config
        config.STOP_REQUESTED = True
