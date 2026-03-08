print("Test1")
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QDoubleSpinBox, QCheckBox,
    QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QSpinBox,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QGridLayout, QMessageBox,
)
from PyQt6.QtCore import Qt, QPointF, QRectF, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QPainter, QColor, QTransform, QPen, QBrush
import sys
import random
import math
import copy
import traceback

# Uncaught exception logging: schreibt Tracebacks in crash_log.txt
def _log_excepthook(exc_type, exc_value, exc_tb):
    try:
        with open('crash_log.txt', 'a', encoding='utf-8') as f:
            f.write('\n=== Uncaught exception ===\n')
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    except Exception:
        pass

# Restore default excepthook (disable verbose crash_log writing)
sys.excepthook = sys.__excepthook__


def write_crash_log(msg=None):
    # Logging temporarily disabled to avoid huge log output.
    # Calls remain in code but this function is a no-op.
    return

# -------------------------
# Konfiguration / Parameter (Defaultwerte)
# -------------------------
POPULATION_SIZE = 50
ELITE_KEEP = 25               # Anzahl der besten Layouts, die überleben
MACHINE_COUNT = 6             # Default; wird in UI einstellbar

# Raumgröße (Floor) in Einheiten (Meter)
FLOOR_W = 20.0
FLOOR_H = 15.0

# Raster / Grundeinheit: 1 m Raster
GRID_SIZE = 1.0

# Maschinen-Größen: Liste mit (w_cells, h_cells) pro Maschine (in Rasterzellen)
# Wird initialisiert bei Start / Änderung der Maschinenanzahl
MACHINE_SIZES = [(1, 1) for _ in range(MACHINE_COUNT)]

# Rasterzellen (werden initial berechnet)
GRID_COLS = max(1, int(FLOOR_W // GRID_SIZE))
GRID_ROWS = max(1, int(FLOOR_H // GRID_SIZE))

# Wareneingang / -ausgang in Zellkoordinaten (top-left basierte Zellen)
ENTRY_CELL = (0, 0)
EXIT_CELL = (GRID_COLS - 1, GRID_ROWS - 1)

# Strafen / Gewichtungen
OVERLAP_PENALTY = 1e9         # sehr harte Strafe bei Überlappung (gemeinsame Zellen)
OUT_OF_BOUNDS_PENALTY = 1e7   # harte Strafe wenn Maschine (teilweise) außerhalb liegt
DIST_SCALE = 1.0              # Distanzgewicht in der Kostenfunktion

# Mutations-Grundwerte (in Zellen / Indexen)
BASE_MUTATION_PROB = 0.15
BASE_MUTATION_POS_STD = 1.0    # Gauß für Zellverschiebung
BASE_MUTATION_ANGLE_STD = 1    # Schritte für 90°-Rotationen (Indexänderung)

MUTATION_PROB = BASE_MUTATION_PROB
MUTATION_POS_STD = BASE_MUTATION_POS_STD
MUTATION_ANGLE_STD = BASE_MUTATION_ANGLE_STD

# erlaubte Rotationswerte (nur Vielfache von 90°)
ROTATIONS = [0, 90, 180, 270]

# Globaler Abbruch-Flag für GA
STOP_REQUESTED = False
# Hindernisse (Set von (col,row)-Zellen) — durch den Editor gesetzt
OBSTACLES = set()
OBSTACLE_PENALTY = 1e9

# -------------------------
# Hilfsfunktionen
# -------------------------
def update_grid_counts():
    """Aktualisiert globale Grid-Spalten/Zeilen nach Änderung von FLOOR_W/H."""
    global GRID_COLS, GRID_ROWS
    GRID_COLS = max(1, int(FLOOR_W // GRID_SIZE))
    GRID_ROWS = max(1, int(FLOOR_H // GRID_SIZE))

def rebuild_machine_sizes(count):
    """(Re-)initialisiert MACHINE_SIZES-Liste wenn Maschinenanzahl geändert wird."""
    global MACHINE_SIZES
    old = MACHINE_SIZES[:]
    MACHINE_SIZES = []
    for i in range(count):
        if i < len(old):
            MACHINE_SIZES.append(old[i])
        else:
            MACHINE_SIZES.append((1, 1))

update_grid_counts()
rebuild_machine_sizes(MACHINE_COUNT)

def rect_corners(center, w, h, angle_deg):
    """
    Berechnet Eckpunkte eines rotierenden Rechtecks (für Kollisionstest).
    center: (x,y), w/h: in Metern.
    """
    cx, cy = center
    angle = math.radians(angle_deg)
    dx = w / 2.0
    dy = h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    world = []
    for x, y in pts:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        world.append((rx, ry))
    return world


def _draw_arrowhead(painter, x1, y1, x2, y2, length=0.6):
    """Draw a filled triangular arrowhead at (x2,y2) pointing from (x1,y1).
    length is in same units as coordinates (meters in floor coordinates)."""
    dx = x2 - x1
    dy = y2 - y1
    ang = math.atan2(dy, dx)
    # base point a bit back from tip
    bx = x2 - length * math.cos(ang)
    by = y2 - length * math.sin(ang)
    # perpendicular offset
    side = length * 0.4
    pvx = -math.sin(ang) * side
    pvy = math.cos(ang) * side
    leftx = bx + pvx
    lefty = by + pvy
    rightx = bx - pvx
    righty = by - pvy
    try:
        from PyQt6.QtGui import QPolygonF
        pts = [QPointF(x2, y2), QPointF(leftx, lefty), QPointF(rightx, righty)]
        poly = QPolygonF(pts)
        painter.save()
        painter.setBrush(painter.pen().color())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(poly)
        painter.restore()
    except Exception:
        pass

# --- Connector-Punkte (Input/Output) einer Maschine berechnen ---
def machine_output_point(m):
    """Gibt (x,y) des Output-Anschlusses der Maschine m in Weltkoordinaten zurück.
    Output wird als rechter Mittelpunkt des Maschinen-Rechtecks relativ zur Rotation angenommen."""
    cx = m['x']
    cy = m['y']
    rot = math.radians(int(m.get('z', 0)) % 360)
    w_m = m['w_cells'] * GRID_SIZE
    # Lokaler Punkt am rechten Rand (Mittelpunkt der Kante)
    lx, ly = (w_m / 2.0, 0.0)
    ox = cx + math.cos(rot) * lx - math.sin(rot) * ly
    oy = cy + math.sin(rot) * lx + math.cos(rot) * ly
    return ox, oy

def machine_input_point(m):
    """Gibt (x,y) des Input-Anschlusses der Maschine m in Weltkoordinaten zurück.
    Input wird als linker Mittelpunkt des Maschinen-Rechtecks relativ zur Rotation angenommen."""
    cx = m['x']
    cy = m['y']
    rot = math.radians(int(m.get('z', 0)) % 360)
    w_m = m['w_cells'] * GRID_SIZE
    # Lokaler Punkt am linken Rand (Mittelpunkt der Kante)
    lx, ly = (-w_m / 2.0, 0.0)
    ix = cx + math.cos(rot) * lx - math.sin(rot) * ly
    iy = cy + math.sin(rot) * lx + math.cos(rot) * ly
    return ix, iy

# cell_center für variable Maschinen-Größen:
def cell_center_from_topleft(col, row, w_cells, h_cells):
    """
    Berechnet Zentrum (x,y in m) einer Maschine, die an (col,row) als top-left Zelle verankert ist,
    und w_cells/h_cells groß ist.
    """
    x = (col + w_cells / 2.0) * GRID_SIZE
    y = (row + h_cells / 2.0) * GRID_SIZE
    return x, y

# --- Neue Hilfsfunktion: effektive Dimensionen unter Rotation ---
def effective_dims(m_or_w_h, z=None):
    """
    Gibt (w_eff, h_eff) in Zellen zurück.
    - m_or_w_h kann ein Maschinen-Dict sein oder ein (w_cells,h_cells)-Tuple.
    - z ist optional: Rotation in Grad. Wenn m_or_w_h ein dict ist, wird z daraus gelesen.
    """
    if isinstance(m_or_w_h, tuple) or isinstance(m_or_w_h, list):
        w_cells, h_cells = m_or_w_h
    else:
        w_cells = m_or_w_h['w_cells']
        h_cells = m_or_w_h['h_cells']
        if z is None:
            z = m_or_w_h.get('z', 0)

    if z is None:
        z = 0
    # Rotation 90 oder 270 tauscht effektive Breite/Höhe
    if int(z) % 180 == 90:
        return (h_cells, w_cells)
    else:
        return (w_cells, h_cells)

# --- occupied_cells: berücksichtigt Rotation (effektive Abmessungen) ---
def occupied_cells(m):
    """Gibt Menge der (col,row)-Zellen zurück, die Maschine m belegt (top-left, integer Zellen),
       berücksichtigt Rotation durch effektive Dimensionsbestimmung."""
    w_eff, h_eff = effective_dims(m)
    cells = set()
    for dx in range(int(w_eff)):
        for dy in range(int(h_eff)):
            cells.add((int(m['gx']) + dx, int(m['gy']) + dy))
    return cells

# --- normalize_individual: clamp unter Berücksichtigung effektiver Abmessungen ---
def normalize_individual(ind):
    """
    Stellt sicher, dass gx/gy ganzzahlige Zellkoordinaten sind und x/y aus ihnen berechnet werden.
    Berücksichtigt die Rotation (effektive Dimensionsänderung).
    """
    for m in ind:
        # sicherstellen ganze Zellen
        m['gx'] = int(round(m.get('gx', 0)))
        m['gy'] = int(round(m.get('gy', 0)))
        # effektive Abmessungen je nach Rotation
        w_eff, h_eff = effective_dims(m)
        max_col = max(0, GRID_COLS - int(w_eff))
        max_row = max(0, GRID_ROWS - int(h_eff))
        m['gx'] = max(0, min(max_col, m['gx']))
        m['gy'] = max(0, min(max_row, m['gy']))
        # mittelpunkte setzen basierend auf effektiven Zellen (w_eff/h_eff)
        # center berechnen von top-left und eff-dims:
        m['x'], m['y'] = cell_center_from_topleft(m['gx'], m['gy'], w_eff, h_eff)
        # Rotation auf erlaubte Werte clamped (falls fehlerhaft)
        if m.get('z') not in ROTATIONS:
            m['z'] = ROTATIONS[0]

# --- Hilfsfunktion: prüfen ob an Position (col,row) Platz (ohne Überlappung) ist ---
def can_place_at(col, row, w_cells, h_cells, occupied_set):
    """Gibt True zurück falls die Zellen (col..col+w-1, row..row+h-1) keine Überschneidung mit occupied_set haben."""
    for dx in range(int(w_cells)):
        for dy in range(int(h_cells)):
            cell = (col + dx, row + dy)
            # Prüfe Hindernisse global
            if cell in OBSTACLES:
                return False
            if cell in occupied_set:
                return False
    return True

# --- Zufällige Maschine (mit Rotation) erzeugen, aber ohne Überlappungen wenn möglich ---
def random_machine_nonoverlap(idx, occupied_set, max_attempts=200):
    """
    Versuch, Maschine idx so zu platzieren, dass sie nicht mit occupied_set überlappt.
    Wenn nach max_attempts kein passender Platz gefunden, wird eine random-Position zurückgegeben (evtl. mit Overlap).
    """
    w_cells, h_cells = MACHINE_SIZES[idx]
    # Wähle zufällige Rotation
    z = random.choice(ROTATIONS)
    # effektive dims abhängig von Rotation
    w_eff, h_eff = effective_dims((w_cells, h_cells), z)
    max_col = max(0, GRID_COLS - int(w_eff))
    max_row = max(0, GRID_ROWS - int(h_eff))

    for _ in range(max_attempts):
        col = random.randint(0, max_col)
        row = random.randint(0, max_row)
        if can_place_at(col, row, w_eff, h_eff, occupied_set):
            x, y = cell_center_from_topleft(col, row, w_eff, h_eff)
            return {'x': x, 'y': y, 'z': z, 'gx': int(col), 'gy': int(row),
                    'w_cells': int(w_cells), 'h_cells': int(h_cells)}

    # Fallback: random Platz (kann überlappen)
    col = random.randint(0, max_col)
    row = random.randint(0, max_row)
    x, y = cell_center_from_topleft(col, row, w_eff, h_eff)
    return {'x': x, 'y': y, 'z': z, 'gx': int(col), 'gy': int(row),
            'w_cells': int(w_cells), 'h_cells': int(h_cells)}

# --- random_individual: baut Individuum auf, versucht Kollisionen beim initialen Platzieren zu vermeiden ---
def random_individual():
    #Erstellt ein Layout/Individuum mit MACHINE_COUNT Maschinen; initial möglichst ohne Überschneidung
    occupied = set()
    ind = []
    for i in range(MACHINE_COUNT):
        m = random_machine_nonoverlap(i, occupied, max_attempts=250)
        # bestimme effektive dims (nach Rotation) und markiere Zellen als belegt
        w_eff, h_eff = effective_dims(m)
        for dx in range(int(w_eff)):
            for dy in range(int(h_eff)):
                occupied.add((m['gx'] + dx, m['gy'] + dy))
        ind.append(m)
    normalize_individual(ind)
    return ind
# -------------------------
# Individuum / Population
# -------------------------


def init_population():
    """Erstellt die Startpopulation mit POPULATION_SIZE Individuen."""
    return [random_individual() for _ in range(POPULATION_SIZE)]

# -------------------------
# Kostenfunktion (Fitness) - Raster-basiert
# -------------------------
def fitness(ind):
    """
    Kostenschätzer (niedriger ist besser):
    - Distanzkosten zwischen aufeinanderfolgenden Maschinen (Mittelpunkt Euklid)
    - sehr harte Strafe, wenn sich Maschinenzellen überschneiden (gemeinsame Zellen)
    - harte Strafe, wenn Maschine außerhalb des Floors liegt
    - Distanzkosten zu Entry/Exit
    """
    cost = 0.0

    # 1) Distanzkosten entlang der Kette (Mitte zu Mitte) — Manhattan (L1)
    n = min(MACHINE_COUNT, len(ind))
    for i in range(n - 1):
        a = ind[i]
        b = ind[i+1]
        dx = abs(a['x'] - b['x'])
        dy = abs(a['y'] - b['y'])
        cost += DIST_SCALE * (dx + dy)

    # 2) Überlappungsstrafe über Zellen (jede doppelt belegte Zelle sehr teuer)
    cell_owner = {}
    for i, m in enumerate(ind):
        cells = occupied_cells(m)
        for c in cells:
            if c in cell_owner:
                # für jede zusätzliche Maschine, die dieselbe Zelle belegt, sehr harte Strafe
                cost += OVERLAP_PENALTY
            else:
                cell_owner[c] = i

    # 2b) Hindernisse: wenn Maschine auf Hinderniszellen steht
    for i, m in enumerate(ind):
        cells = occupied_cells(m)
        for c in cells:
            if c in OBSTACLES:
                cost += OBSTACLE_PENALTY

    # 3) Out-of-bounds-Prüfung über reale Eckpunkte
    for m in ind:
        w = m['w_cells'] * GRID_SIZE
        h = m['h_cells'] * GRID_SIZE
        poly = rect_corners((m['x'], m['y']), w, h, m['z'])
        out_count = 0
        for (x, y) in poly:
            if x < 0.0 or x > FLOOR_W or y < 0.0 or y > FLOOR_H:
                out_count += 1
        if out_count > 0:
            cost += OUT_OF_BOUNDS_PENALTY * out_count

    # 4) Distanz zu Entry/Exit (leichte Kosten, fördert kurze Wege)
    if ind:
        first = ind[0]
        last = ind[-1]
        entry_x, entry_y = cell_center_from_topleft(ENTRY_CELL[0], ENTRY_CELL[1], 1, 1)
        exit_x, exit_y = cell_center_from_topleft(EXIT_CELL[0], EXIT_CELL[1], 1, 1)
        cost += abs(first['x'] - entry_x) + abs(first['y'] - entry_y)
        cost += abs(last['x'] - exit_x) + abs(last['y'] - exit_y)

    return cost

# -------------------------
# Genetische Operatoren
# -------------------------
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
    for i in range(MACHINE_COUNT):
        if random.random() < 0.5:
            child.append(copy.deepcopy(a[i]))
        else:
            child.append(copy.deepcopy(b[i]))
    # Normalisieren (ganzzahlige Zell-Positionen, x/y neu berechnen)
    normalize_individual(child)
    return child
def mutate(ind):
    """
    Mutation im Raster:
    - Verschiebe Maschine in Zellen (Gauss, gerundet) innerhalb erlaubter Top-Left-Range
    - Ändere Rotation auf eines der 4 Werte (durch Indexverschiebung)
    """
    for i, m in enumerate(ind):
        if random.random() < MUTATION_PROB:
            # aktuelle effektive dims berücksichtigen vor Mutation
            # Positionsdelta in Zellen
            delta_col = int(round(random.gauss(0, MUTATION_POS_STD)))
            delta_row = int(round(random.gauss(0, MUTATION_POS_STD)))
            new_col = int(m['gx']) + delta_col
            new_row = int(m['gy']) + delta_row

            # Rotation-Mutation: ggf. zuerst die Rotation ändern und dann Grenzen neu berechnen
            if random.random() < 0.5:
                cur_idx = ROTATIONS.index(m['z']) if m['z'] in ROTATIONS else 0
                delta_rot = int(round(random.gauss(0, MUTATION_ANGLE_STD)))
                new_idx = (cur_idx + delta_rot) % len(ROTATIONS)
                m['z'] = ROTATIONS[new_idx]

            # effektive dims nach möglicher Rotation
            w_eff, h_eff = effective_dims(m)
            max_col = max(0, GRID_COLS - int(w_eff))
            max_row = max(0, GRID_ROWS - int(h_eff))
            new_col = max(0, min(max_col, new_col))
            new_row = max(0, min(max_row, new_row))
            m['gx'] = int(new_col)
            m['gy'] = int(new_row)
            m['x'], m['y'] = cell_center_from_topleft(m['gx'], m['gy'], w_eff, h_eff)
    # Nach Mutation sicherstellen, dass alles integer und im Bereich ist
    normalize_individual(ind)

# -------------------------
# GA Lauf (tatsächlicher Selektions-/Eliminationsprozess)
# -------------------------
def run_ga(generations, progress_callback=None):
    """
    Ältere Full-run Funktion (bleibt erhalten). progress_callback wird optional mit
    (g, generations, best_score, best_ind, population) aufgerufen wenn angegeben.
    """
    update_grid_counts()
    pop = init_population()
    best_ind = None
    best_score = float('inf')

    global STOP_REQUESTED

    for g in range(1, generations + 1):
        if STOP_REQUESTED:
            # Log explicit when a stop was requested so we can trace premature aborts
            try:
                write_crash_log(f"run_ga: STOP_REQUESTED True at gen {g}; reporting final progress and breaking")
            except Exception:
                pass
            if progress_callback:
                try:
                    progress_callback(g, generations, best_score, best_ind, pop)
                except TypeError:
                    progress_callback(g, generations, best_score, best_ind)
            break

        # Bewertung
        scores = [fitness(ind) for ind in pop]

        # Sortiere Population nach Score (aufsteigend)
        paired = list(zip(scores, pop))
        paired.sort(key=lambda p: p[0])
        # Elite extrahieren
        elites = [copy.deepcopy(p[1]) for p in paired[:ELITE_KEEP]]
        elite_scores = [p[0] for p in paired[:ELITE_KEEP]]

        # Update global best
        if elite_scores and elite_scores[0] < best_score:
            best_score = elite_scores[0]
            best_ind = copy.deepcopy(elites[0])

        if progress_callback:
            try:
                progress_callback(g, generations, best_score, best_ind, pop)
            except TypeError:
                progress_callback(g, generations, best_score, best_ind)

        # Erzeugung neuer Individuen:
        new_pop = []
        # 1) Behalte Eliten
        new_pop.extend(copy.deepcopy(elites))

        # 2) Erzeuge weitere Individuen durch Kreuzung/Mutation aus den Eliten
        while len(new_pop) < POPULATION_SIZE:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = uniform_crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        pop = new_pop

    return best_ind, best_score

# -------------------------
# Kleine GA-Engine für Schrittweise Iteration
# -------------------------
class GAEngine:
    def __init__(self, total_generations):
        update_grid_counts()
        self.total_generations = int(total_generations)
        self.generation = 0
        self.population = init_population()
        self.best_ind = None
        self.best_score = float('inf')

    def step(self):
        """Ein Schritt: bewerten, Eliten wählen, neue Population bilden, generation++"""
        global STOP_REQUESTED
        if STOP_REQUESTED or self.generation >= self.total_generations:
            return self.best_score, self.best_ind

        # Bewertung
        scores = [fitness(ind) for ind in self.population]
        paired = list(zip(scores, self.population))
        paired.sort(key=lambda p: p[0])
        elites = [copy.deepcopy(p[1]) for p in paired[:ELITE_KEEP]]
        elite_scores = [p[0] for p in paired[:ELITE_KEEP]]

        if elite_scores and elite_scores[0] < self.best_score:
            self.best_score = elite_scores[0]
            self.best_ind = copy.deepcopy(elites[0])

        # neue Population bauen
        new_pop = []
        new_pop.extend(copy.deepcopy(elites))
        while len(new_pop) < POPULATION_SIZE:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = uniform_crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        return self.best_score, self.best_ind


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
        global STOP_REQUESTED
        STOP_REQUESTED = False
        # Wenn eine Konfigurationskopie vorhanden ist, setze temporär die Modul-Globals
        saved_globals = {}
        try:
            if self.config:
                import copy as _copy
                names = ['POPULATION_SIZE','ELITE_KEEP','MACHINE_COUNT','MACHINE_SIZES','MUTATION_PROB','MUTATION_POS_STD','MUTATION_ANGLE_STD','FLOOR_W','FLOOR_H','GRID_COLS','GRID_ROWS','OBSTACLES','ENTRY_CELL','EXIT_CELL','GRID_SIZE']
                for n in names:
                    saved_globals[n] = globals().get(n)
                    if n in self.config:
                        globals()[n] = _copy.deepcopy(self.config[n])

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
            write_crash_log(f"GAWorker.run: starting GAWorker for {self.generations} generations")
            best_ind, best_score = run_ga(self.generations, progress_callback=_cb)
            write_crash_log(f"GAWorker.run: run_ga returned best_score={best_score} best_ind_present={best_ind is not None}")
        except Exception as e:
            write_crash_log(f"Exception in GAWorker.run: {e}")
            try:
                with open('crash_log.txt', 'a', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            try:
                self.finished.emit(None, float('inf'))
            except Exception:
                pass
        finally:
            # restore saved globals
            try:
                for k, v in saved_globals.items():
                    globals()[k] = v
            except Exception:
                pass
            # ensure finished emitted and log emission
            try:
                write_crash_log(f"GAWorker.run: emitting finished(best_present={best_ind is not None}, best_score={best_score})")
                self.finished.emit(best_ind, best_score)
            except Exception as e:
                write_crash_log(f"GAWorker.run: exception while emitting finished: {e}")

    @pyqtSlot()
    def request_stop(self):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        try:
            write_crash_log("GAWorker.request_stop: STOP_REQUESTED set to True")
        except Exception:
            pass

# -------------------------
# UI-Komponenten
# -------------------------
class SizesDialog(QDialog):
    """Dialog zum Bearbeiten der Maschinen-Größen (in Rasterzellen)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Maschinengrößen bearbeiten (Zellen)")
        self.layout = QVBoxLayout()
        self.form = QGridLayout()
        self.spin_w = []
        self.spin_h = []
        # falls GRID_COLS/ROWS aktuell sind, verwenden
        max_w = max(1, GRID_COLS)
        max_h = max(1, GRID_ROWS)
        for i in range(MACHINE_COUNT):
            w_spin = QSpinBox()
            w_spin.setRange(1, max_w)
            w_spin.setValue(MACHINE_SIZES[i][0] if i < len(MACHINE_SIZES) else 1)
            h_spin = QSpinBox()
            h_spin.setRange(1, max_h)
            h_spin.setValue(MACHINE_SIZES[i][1] if i < len(MACHINE_SIZES) else 1)
            self.spin_w.append(w_spin)
            self.spin_h.append(h_spin)
            self.form.addWidget(QLabel(f"M{i+1} Breite (Zellen)"), i, 0)
            self.form.addWidget(w_spin, i, 1)
            self.form.addWidget(QLabel("Höhe (Zellen)"), i, 2)
            self.form.addWidget(h_spin, i, 3)
        self.layout.addLayout(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)
        self.setLayout(self.layout)

    def get_sizes(self):
        return [(s_w.value(), s_h.value()) for s_w, s_h in zip(self.spin_w, self.spin_h)]

class SettingsDialog(QDialog):
    """Ein einfacher Einstellungsdialog (Platzhalter)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        layout = QFormLayout()
        self.floor_w = QDoubleSpinBox()
        self.floor_w.setRange(1.0, 1000.0)
        self.floor_w.setValue(FLOOR_W)
        self.floor_h = QDoubleSpinBox()
        self.floor_h.setRange(1.0, 1000.0)
        self.floor_h.setValue(FLOOR_H)
        layout.addRow("Floor Breite (m):", self.floor_w)
        layout.addRow("Floor Höhe (m):", self.floor_h)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        return float(self.floor_w.value()), float(self.floor_h.value())


class EditorCanvas(QWidget):
    """Canvas für den Factory Editor: zeichnet Raster und Hindernisse, ermöglicht Klick/Drag zum Setzen/Löschen."""
    def __init__(self, cols, rows, parent=None):
        super().__init__(parent)
        self.cols = cols
        self.rows = rows
        self.cell_size = 24
        self.setMinimumSize(max(200, self.cols * self.cell_size), max(200, self.rows * self.cell_size))
        self.setMouseTracking(True)
        self.drawing = False
        self.draw_mode = True  # True = setzen, False = löschen

    def size_for_grid(self):
        return self.cols * self.cell_size, self.rows * self.cell_size

    def set_grid(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(max(200, self.cols * self.cell_size), max(200, self.rows * self.cell_size))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        w = self.cols * self.cell_size
        h = self.rows * self.cell_size
        # draw grid
        pen = QPen(QColor(200, 200, 200))
        painter.setPen(pen)
        for c in range(self.cols + 1):
            x = c * self.cell_size
            painter.drawLine(x, 0, x, h)
        for r in range(self.rows + 1):
            y = r * self.cell_size
            painter.drawLine(0, y, w, y)

        # draw obstacles
        painter.setBrush(QBrush(QColor(80, 80, 80)))
        painter.setPen(Qt.PenStyle.NoPen)
        for (col, row) in OBSTACLES:
            if 0 <= col < self.cols and 0 <= row < self.rows:
                painter.drawRect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.draw_mode = True
            self._toggle_cell(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.drawing = True
            self.draw_mode = False
            self._toggle_cell(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self._toggle_cell(event)

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def _toggle_cell(self, event):
        x = event.position().x() if hasattr(event, 'position') else event.x()
        y = event.position().y() if hasattr(event, 'position') else event.y()
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
            return
        if self.draw_mode:
            OBSTACLES.add((col, row))
        else:
            if (col, row) in OBSTACLES:
                OBSTACLES.remove((col, row))
        self.update()


class FactoryEditorDialog(QDialog):
    """Dialog, um Floor-Größe zu setzen und Hindernisse (Säulen) zu malen."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Factory Editor")
        self.layout = QVBoxLayout()

        # Floor size controls
        form = QGridLayout()
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, 1000)
        self.w_spin.setValue(GRID_COLS)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, 1000)
        self.h_spin.setValue(GRID_ROWS)
        form.addWidget(QLabel("Grid Cols (m):"), 0, 0)
        form.addWidget(self.w_spin, 0, 1)
        form.addWidget(QLabel("Grid Rows (m):"), 1, 0)
        form.addWidget(self.h_spin, 1, 1)
        self.layout.addLayout(form)

        # Editor canvas
        self.canvas = EditorCanvas(self.w_spin.value(), self.h_spin.value(), parent=self)
        self.layout.addWidget(self.canvas)

        # Buttons: Clear, OK/Cancel
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Alles löschen")
        clear_btn.clicked.connect(self.clear_obstacles)
        btn_layout.addWidget(clear_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        self.layout.addLayout(btn_layout)

        # Update canvas when sizes change
        self.w_spin.valueChanged.connect(self.on_size_changed)
        self.h_spin.valueChanged.connect(self.on_size_changed)

        self.setLayout(self.layout)

    def on_size_changed(self):
        cols = int(self.w_spin.value())
        rows = int(self.h_spin.value())
        self.canvas.set_grid(cols, rows)

    def clear_obstacles(self):
        OBSTACLES.clear()
        self.canvas.update()

    def get_values(self):
        return int(self.w_spin.value()), int(self.h_spin.value()), set(OBSTACLES)

class LayoutCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_data = None
        self.setMinimumSize(800, 480)
        self.setMaximumSize(1600, 900)

    def set_layout(self, layout_data):
        self.layout_data = layout_data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        w = self.width()
        h = self.height()
        sx = w / FLOOR_W if FLOOR_W > 0 else 1.0
        sy = h / FLOOR_H if FLOOR_H > 0 else 1.0

        painter.save()
        painter.scale(sx, sy)

        # Floor
        pen = QPen(QColor(180, 180, 180))
        pen.setWidthF(2.0 / sx)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawRect(QRectF(0.0, 0.0, float(FLOOR_W), float(FLOOR_H)))

        # Rasterlinien
        grid_pen = QPen(QColor(200, 200, 200))
        grid_pen.setWidthF(0.5 / sx)
        painter.setPen(grid_pen)
        for c in range(GRID_COLS + 1):
            x = c * GRID_SIZE
            painter.drawLine(QPointF(x, 0.0), QPointF(x, FLOOR_H))
        for r in range(GRID_ROWS + 1):
            y = r * GRID_SIZE
            painter.drawLine(QPointF(0.0, y), QPointF(FLOOR_W, y))

        # Hindernisse zeichnen (Säulen) als dunkle Zellen
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(90, 90, 90)))
        for (col, row) in OBSTACLES:
            if 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS:
                rx = col * GRID_SIZE
                ry = row * GRID_SIZE
                painter.drawRect(QRectF(rx, ry, GRID_SIZE, GRID_SIZE))

        # Entry / Exit Marker (bleibt unverändert)
        ex_x, ex_y = cell_center_from_topleft(ENTRY_CELL[0], ENTRY_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(255, 200, 150)))
        painter.setPen(QPen(QColor(180, 120, 80)))
        painter.drawEllipse(QPointF(ex_x, ex_y), 0.3, 0.3)
        ox, oy = cell_center_from_topleft(EXIT_CELL[0], EXIT_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(200, 255, 200)))
        painter.setPen(QPen(QColor(80, 160, 80)))
        painter.drawEllipse(QPointF(ox, oy), 0.3, 0.3)

        # Farbzuordnung nur basierend auf Rotation (0/90/180/270)
        rotation_color = {
            0: QColor(120, 160, 255),    # blau-ish
            90: QColor(160, 220, 160),   # grün-ish
            180: QColor(255, 200, 120),  # orange-ish
            270: QColor(200, 160, 255),  # violett-ish
        }

        # Maschinen zeichnen — einfarbig gefüllt, ohne Umrandung
        if self.layout_data:
            for idx, m in enumerate(self.layout_data):
                cx = m['x']
                cy = m['y']
                rot = int(m.get('z', 0)) % 360
                # effektive Größe in Metern
                w_m = m['w_cells'] * GRID_SIZE
                h_m = m['h_cells'] * GRID_SIZE

                # Farbe nur von Rotation abhängig wählen
                color = rotation_color.get(rot, rotation_color[0])

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)

                rect = QRectF(-w_m/2, -h_m/2, w_m, h_m)

                # Fülle ohne Rahmen
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(rect)

                # Text: Font-Größe 8, schwarze Schrift, ohne Rahmeninterferenzen
                font = painter.font()
                font.setPointSize(1)
                painter.setFont(font)
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(idx+1))

                painter.restore()

            # Materialfluss zeichnen: Linien zwischen aufeinanderfolgenden Maschinen + Entry/Exit
            try:
                flow_pen = QPen(QColor(60, 60, 60))
                flow_pen.setWidthF(0.06 / sx)
                painter.setPen(flow_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # Entry -> erster Maschine (Entry -> Input der Maschine)
                if len(self.layout_data) >= 1:
                    first = self.layout_data[0]
                    ex_x, ex_y = cell_center_from_topleft(ENTRY_CELL[0], ENTRY_CELL[1], 1, 1)
                    in_x, in_y = machine_input_point(first)
                    painter.drawLine(QPointF(ex_x, ex_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, ex_x, ex_y, in_x, in_y, length=0.6)

                # Zwischen Maschinen: Output der aktuellen -> Input der nächsten
                for i in range(len(self.layout_data) - 1):
                    a = self.layout_data[i]
                    b = self.layout_data[i+1]
                    out_x, out_y = machine_output_point(a)
                    in_x, in_y = machine_input_point(b)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, out_x, out_y, in_x, in_y, length=0.5)

                # letzte -> Exit (Output der letzten Maschine -> Exit)
                if len(self.layout_data) >= 1:
                    last = self.layout_data[-1]
                    exit_x, exit_y = cell_center_from_topleft(EXIT_CELL[0], EXIT_CELL[1], 1, 1)
                    out_x, out_y = machine_output_point(last)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(exit_x, exit_y))
                    _draw_arrowhead(painter, out_x, out_y, exit_x, exit_y, length=0.6)
            except Exception:
                pass

        painter.restore()

class PopulationCanvas(QWidget):
    def __init__(self, parent=None, cols=10, rows=5):
        super().__init__(parent)
        self.population = None
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(900, 480)
        self.show_grid = True

    def set_population(self, population):
        """population: Liste von Individuen (Layouts)."""
        self.population = population
        self.update()

    def set_show_grid(self, show: bool):
        self.show_grid = bool(show)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        if not self.population:
            return

        W = self.width()
        H = self.height()
        cell_w = W / self.cols
        cell_h = H / self.rows

        # Farben für Rotationen (lokal definieren)
        rotation_color = {
            0: QColor(120, 160, 255),
            90: QColor(160, 220, 160),
            180: QColor(255, 200, 120),
            270: QColor(200, 160, 255),
        }

        for idx, ind in enumerate(self.population):
            col = idx % self.cols
            row = idx // self.cols
            if row >= self.rows:
                break

            cell_x = col * cell_w
            cell_y = row * cell_h

            painter.save()
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.setBrush(QBrush(QColor(250, 250, 250)))
            painter.drawRect(QRectF(cell_x + 2, cell_y + 2, cell_w - 4, cell_h - 4))

            margin = 6.0
            avail_w = max(1.0, cell_w - margin * 2)
            avail_h = max(1.0, cell_h - margin * 2)
            scale = min(avail_w / FLOOR_W, avail_h / FLOOR_H)

            painter.translate(cell_x + margin, cell_y + margin)
            painter.scale(scale, scale)

            # Zeichne Mini-Floor (grob)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(245, 245, 245)))
            painter.drawRect(QRectF(0.0, 0.0, float(FLOOR_W), float(FLOOR_H)))

            # Zeichne Hindernisse in der Mini-Ansicht
            painter.setBrush(QBrush(QColor(90, 90, 90)))
            painter.setPen(Qt.PenStyle.NoPen)
            for (col_o, row_o) in OBSTACLES:
                if 0 <= col_o < GRID_COLS and 0 <= row_o < GRID_ROWS:
                    ox = col_o * GRID_SIZE
                    oy = row_o * GRID_SIZE
                    painter.drawRect(QRectF(ox, oy, GRID_SIZE, GRID_SIZE))

            # optionales Mini-Raster
            if self.show_grid:
                grid_pen = QPen(QColor(220, 220, 220))
                grid_pen.setWidthF(0.05)
                painter.setPen(grid_pen)
                for c in range(GRID_COLS + 1):
                    x = c * GRID_SIZE
                    painter.drawLine(QPointF(x, 0.0), QPointF(x, FLOOR_H))
                for r in range(GRID_ROWS + 1):
                    y = r * GRID_SIZE
                    painter.drawLine(QPointF(0.0, y), QPointF(FLOOR_W, y))

            # Entry / Exit (skalierte Mini-Markierungen)
            ex_x, ex_y = cell_center_from_topleft(ENTRY_CELL[0], ENTRY_CELL[1], 1, 1)
            ox, oy = cell_center_from_topleft(EXIT_CELL[0], EXIT_CELL[1], 1, 1)
            painter.setBrush(QBrush(QColor(255, 200, 150)))
            painter.setPen(QPen(QColor(180, 120, 80)))
            painter.drawEllipse(QPointF(ex_x, ex_y), 0.25, 0.25)
            painter.setBrush(QBrush(QColor(200, 255, 200)))
            painter.setPen(QPen(QColor(80, 160, 80)))
            painter.drawEllipse(QPointF(ox, oy), 0.25, 0.25)

            # Zeichne Mini-Maschinen (einfarbig, ohne Rahmen)
            for m in ind:
                cx = m['x']
                cy = m['y']
                rot = int(m.get('z', 0)) % 360
                w_m = m['w_cells'] * GRID_SIZE
                h_m = m['h_cells'] * GRID_SIZE
                color = rotation_color.get(rot, rotation_color[0])

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)

                rect = QRectF(-w_m / 2, -h_m / 2, w_m, h_m)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(rect)

                painter.restore()

            # Materialfluss in Mini-Ansicht: Linien zwischen Maschinen + Entry/Exit
            try:
                flow_pen = QPen(QColor(60, 60, 60))
                flow_pen.setWidthF(0.04)
                painter.setPen(flow_pen)
                painter.setBrush(Qt.PenStyle.NoBrush)

                if len(ind) >= 1:
                    first = ind[0]
                    ex_x, ex_y = cell_center_from_topleft(ENTRY_CELL[0], ENTRY_CELL[1], 1, 1)
                    in_x, in_y = machine_input_point(first)
                    painter.drawLine(QPointF(ex_x, ex_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, ex_x, ex_y, in_x, in_y, length=0.5)

                for i in range(len(ind) - 1):
                    a = ind[i]
                    b = ind[i+1]
                    out_x, out_y = machine_output_point(a)
                    in_x, in_y = machine_input_point(b)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, out_x, out_y, in_x, in_y, length=0.4)

                if len(ind) >= 1:
                    last = ind[-1]
                    exit_x, exit_y = cell_center_from_topleft(EXIT_CELL[0], EXIT_CELL[1], 1, 1)
                    out_x, out_y = machine_output_point(last)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(exit_x, exit_y))
                    _draw_arrowhead(painter, out_x, out_y, exit_x, exit_y, length=0.5)
            except Exception:
                pass

            painter.resetTransform()
            painter.restore()

class BestDialog(QDialog):
    """Zeigt ein Layout (größere Ansicht) an."""
    def __init__(self, layout_data, parent=None, title="Beste Lösung"):
        super().__init__(parent)
        self.setWindowTitle(title)
        v = QVBoxLayout()
        self.canvas = LayoutCanvas(self)
        self.canvas.set_layout(layout_data)
        v.addWidget(self.canvas)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        v.addWidget(buttons)
        self.setLayout(v)
        self.resize(900, 600)

# -------------------------
# Hauptfenster
# -------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fabrikplaner GA")
        self.setMinimumSize(1024, 768)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- Menüleiste
        menu_bar = self.menuBar() if hasattr(self, 'menuBar') else None
        if menu_bar is not None:
            file_menu = menu_bar.addMenu("Datei")
            edit_menu = menu_bar.addMenu("Bearbeiten")
            view_menu = menu_bar.addMenu("Ansicht")
            help_menu = menu_bar.addMenu("Hilfe")

            # --- Datei-Menü
            import_action = file_menu.addAction("Importieren")
            import_action.triggered.connect(self.import_layout)
            export_action = file_menu.addAction("Exportieren")
            export_action.triggered.connect(self.export_layout)
            file_menu.addSeparator()
            exit_action = file_menu.addAction("Beenden")
            exit_action.triggered.connect(self.close)

            # --- Bearbeiten-Menü
            undo_action = edit_menu.addAction("Rückgängig")
            undo_action.triggered.connect(self.undo)
            redo_action = edit_menu.addAction("Wiederherstellen")
            redo_action.triggered.connect(self.redo)
            edit_menu.addSeparator()
            settings_action = edit_menu.addAction("Einstellungen")
            settings_action.triggered.connect(self.show_settings)
            # Factory Editor
            factory_editor_action = edit_menu.addAction("Factory Editor")
            factory_editor_action.triggered.connect(self.open_factory_editor)

            # --- Ansicht-Menü
            show_grid_action = view_menu.addAction("Raster anzeigen")
            show_grid_action.setCheckable(True)
            show_grid_action.setChecked(True)
            show_grid_action.triggered.connect(self.toggle_grid)

            # --- Hilfe-Menü
            about_action = help_menu.addAction("Über")
            about_action.triggered.connect(self.show_about)

        # --- Steuerungsbereich (oben)
        controls_layout = QHBoxLayout()
        self.layout.addLayout(controls_layout)

        # --- Linke Seite: Maschinenanzahl und -größen
        machine_group = QGroupBox("Maschinen")
        controls_layout.addWidget(machine_group)
        machine_layout = QVBoxLayout()
        machine_group.setLayout(machine_layout)

        self.machine_count_spin = QSpinBox()
        self.machine_count_spin.setRange(1, 100)
        self.machine_count_spin.setValue(MACHINE_COUNT)
        self.machine_count_spin.valueChanged.connect(self.on_machine_count_changed)
        machine_layout.addWidget(QLabel("Anzahl Maschinen:"))
        machine_layout.addWidget(self.machine_count_spin)

        self.size_button = QPushButton("Größen bearbeiten")
        self.size_button.clicked.connect(self.edit_machine_sizes)
        machine_layout.addWidget(self.size_button)
        # Factory Editor Button
        self.factory_button = QPushButton("Factory Editor")
        self.factory_button.clicked.connect(self.open_factory_editor)
        machine_layout.addWidget(self.factory_button)

        # Entry / Exit Controls
        entry_group = QGroupBox("Entry / Exit (Zellen)")
        controls_layout.addWidget(entry_group)
        entry_layout = QGridLayout()
        entry_group.setLayout(entry_layout)

        self.entry_col = QSpinBox()
        self.entry_col.setRange(0, max(0, GRID_COLS - 1))
        self.entry_col.setValue(ENTRY_CELL[0])
        self.entry_row = QSpinBox()
        self.entry_row.setRange(0, max(0, GRID_ROWS - 1))
        self.entry_row.setValue(ENTRY_CELL[1])
        self.exit_col = QSpinBox()
        self.exit_col.setRange(0, max(0, GRID_COLS - 1))
        self.exit_col.setValue(EXIT_CELL[0])
        self.exit_row = QSpinBox()
        self.exit_row.setRange(0, max(0, GRID_ROWS - 1))
        self.exit_row.setValue(EXIT_CELL[1])

        entry_layout.addWidget(QLabel("Entry Col"), 0, 0)
        entry_layout.addWidget(self.entry_col, 0, 1)
        entry_layout.addWidget(QLabel("Entry Row"), 0, 2)
        entry_layout.addWidget(self.entry_row, 0, 3)
        entry_layout.addWidget(QLabel("Exit Col"), 1, 0)
        entry_layout.addWidget(self.exit_col, 1, 1)
        entry_layout.addWidget(QLabel("Exit Row"), 1, 2)
        entry_layout.addWidget(self.exit_row, 1, 3)

        self.entry_col.valueChanged.connect(self.on_entry_exit_changed)
        self.entry_row.valueChanged.connect(self.on_entry_exit_changed)
        self.exit_col.valueChanged.connect(self.on_entry_exit_changed)
        self.exit_row.valueChanged.connect(self.on_entry_exit_changed)

        # --- Rechte Seite: GA-Parameter
        ga_group = QGroupBox("Genetischer Algorithmus")
        controls_layout.addWidget(ga_group)
        ga_layout = QVBoxLayout()
        ga_group.setLayout(ga_layout)

        self.population_size_spin = QSpinBox()
        self.population_size_spin.setRange(1, 1000)
        self.population_size_spin.setValue(POPULATION_SIZE)
        ga_layout.addWidget(QLabel("Population Größe:"))
        ga_layout.addWidget(self.population_size_spin)

        self.elite_keep_spin = QSpinBox()
        self.elite_keep_spin.setRange(1, 100)
        self.elite_keep_spin.setValue(ELITE_KEEP)
        ga_layout.addWidget(QLabel("Eliten behalten:"))
        ga_layout.addWidget(self.elite_keep_spin)

        self.mutation_prob_spin = QDoubleSpinBox()
        self.mutation_prob_spin.setRange(0.0, 1.0)
        self.mutation_prob_spin.setValue(BASE_MUTATION_PROB)
        self.mutation_prob_spin.setSingleStep(0.01)
        ga_layout.addWidget(QLabel("Mutationswahrscheinlichkeit:"))
        ga_layout.addWidget(self.mutation_prob_spin)

        self.mutation_pos_std_spin = QDoubleSpinBox()
        self.mutation_pos_std_spin.setRange(0.1, 10.0)
        self.mutation_pos_std_spin.setValue(BASE_MUTATION_POS_STD)
        self.mutation_pos_std_spin.setSingleStep(0.1)
        ga_layout.addWidget(QLabel("Positions-Mutations-StdAbw.:"))
        ga_layout.addWidget(self.mutation_pos_std_spin)

        self.mutation_angle_std_spin = QDoubleSpinBox()
        self.mutation_angle_std_spin.setRange(0.1, 10.0)
        self.mutation_angle_std_spin.setValue(BASE_MUTATION_ANGLE_STD)
        self.mutation_angle_std_spin.setSingleStep(0.1)
        ga_layout.addWidget(QLabel("Rotations-Mutations-StdAbw.:"))
        ga_layout.addWidget(self.mutation_angle_std_spin)

        # --- Anzahl Generationen + Start-Button
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(1, 10000)
        self.generations_spin.setValue(200)
        ga_layout.addWidget(QLabel("Generationen (Total):"))
        ga_layout.addWidget(self.generations_spin)

        self.start_btn = QPushButton("GA starten (vollständig)")
        self.start_btn.clicked.connect(self.start_ga)
        ga_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Abbrechen")
        self.stop_btn.setEnabled(False)
        ga_layout.addWidget(self.stop_btn)

        # --- Generations-per-click und Next-Button
        adv_layout = QHBoxLayout()
        ga_layout.addLayout(adv_layout)
        self.advance_spin = QSpinBox()
        self.advance_spin.setRange(1, 1000)
        self.advance_spin.setValue(1)
        adv_layout.addWidget(QLabel("Generationen / Klick:"))
        adv_layout.addWidget(self.advance_spin)

        self.next_gen_btn = QPushButton("Nächste Generationen")
        self.next_gen_btn.clicked.connect(self.on_next_generation)
        adv_layout.addWidget(self.next_gen_btn)

        self.show_best_spin = QSpinBox()
        self.show_best_spin.setRange(0, 1000)
        self.show_best_spin.setValue(10)
        adv_layout.addWidget(QLabel("Bestanzeige alle N Gen.:"))
        adv_layout.addWidget(self.show_best_spin)

        # Anzeige Generationszähler
        self.gencounter_label = QLabel("0/0")
        ga_layout.addWidget(self.gencounter_label)

        # --- Zentraler Bereich: Population und Fitness (10x5 Darstellung)
        self.pop_canvas = PopulationCanvas(self, cols=10, rows=5)
        self.layout.addWidget(self.pop_canvas)

        # --- Statusleiste (unten)
        self.status_label = QLabel("Willkommen beim Fabrikplaner!")
        self.layout.addWidget(self.status_label)

        # --- GA Engine (wird später initialisiert)
        self.ga_engine = None

        # --- Info Label (Score-Anzeige)
        info = QLabel("")
        self.layout.addWidget(info)

    def closeEvent(self, event):
        """Überprüfen, ob das Fenster geschlossen werden soll."""
        # Hier können Sie Abfragen oder Bereinigungen durchführen
        # Beispiel: Bestätigungsdialog
        reply = QMessageBox.question(self, 'Bestätigung', 'Möchten Sie das Programm wirklich beenden?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    def on_machine_count_changed(self, count):
        """Ändert die Anzahl der Maschinen und aktualisiert die Größen-Eingabefelder."""
        global MACHINE_COUNT
        MACHINE_COUNT = int(count)
        rebuild_machine_sizes(MACHINE_COUNT)
        if self.ga_engine:
            self.ga_engine.machine_count = count
        # Aktualisiere die Größe SpinBoxen im Größen-Dialog, falls offen
        if hasattr(self, 'sizes_dialog') and self.sizes_dialog.isVisible():
            self.sizes_dialog.get_sizes()
        self.status_label.setText(f"Anzahl Maschinen: {count}")

    def edit_machine_sizes(self):
        """Öffnet den Dialog zum Bearbeiten der Maschinen-Größen."""
        self.sizes_dialog = SizesDialog(self)
        self.sizes_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.sizes_dialog.exec()
        new_sizes = self.sizes_dialog.get_sizes()
        for i, (w, h) in enumerate(new_sizes):
            MACHINE_SIZES[i] = (w, h)
        if self.ga_engine:
            self.ga_engine.machine_sizes = MACHINE_SIZES
        self.status_label.setText(f"Maschinen-Größen aktualisiert: {MACHINE_SIZES}")


    def show_settings(self):
        """Öffnet das Einstellungsfenster."""
        dialog = SettingsDialog(self)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_w, new_h = dialog.get_values()
            global FLOOR_W, FLOOR_H
            FLOOR_W = new_w
            FLOOR_H = new_h
            update_grid_counts()
            # Entferne Hindernisse außerhalb des neuen Rasters
            global OBSTACLES
            OBSTACLES = {c for c in OBSTACLES if c[0] < GRID_COLS and c[1] < GRID_ROWS}
            # Update Entry/Exit spin ranges
            self.entry_col.setRange(0, max(0, GRID_COLS - 1))
            self.exit_col.setRange(0, max(0, GRID_COLS - 1))
            self.entry_row.setRange(0, max(0, GRID_ROWS - 1))
            self.exit_row.setRange(0, max(0, GRID_ROWS - 1))
            self.pop_canvas.update()
            self.status_label.setText(f"Floor gesetzt: {FLOOR_W} x {FLOOR_H}")

    def open_factory_editor(self):
        # globale Namen vorher deklarieren, da sie in dieser Funktion verwendet werden
        global FLOOR_W, FLOOR_H, GRID_COLS, GRID_ROWS, OBSTACLES
        dialog = FactoryEditorDialog(self)
        # Vorbefüllen
        dialog.w_spin.setValue(GRID_COLS)
        dialog.h_spin.setValue(GRID_ROWS)
        dialog.canvas.set_grid(GRID_COLS, GRID_ROWS)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_cols = int(dialog.w_spin.value())
            new_rows = int(dialog.h_spin.value())
            # Aktualisiere Floor in Metern
            FLOOR_W = float(new_cols * GRID_SIZE)
            FLOOR_H = float(new_rows * GRID_SIZE)
            update_grid_counts()
            # OBSTACLES sind bereits in globalem Set (Dialog modifiziert dieses)
            OBSTACLES = {c for c in OBSTACLES if c[0] < GRID_COLS and c[1] < GRID_ROWS}
            # Update Entry/Exit spin ranges
            self.entry_col.setRange(0, max(0, GRID_COLS - 1))
            self.exit_col.setRange(0, max(0, GRID_COLS - 1))
            self.entry_row.setRange(0, max(0, GRID_ROWS - 1))
            self.exit_row.setRange(0, max(0, GRID_ROWS - 1))
            # Refresh views
            self.pop_canvas.update()
            try:
                self.pop_canvas.set_population(self.ga_engine.population if self.ga_engine else None)
            except Exception:
                pass
            self.status_label.setText(f"Factory aktualisiert: {GRID_COLS}x{GRID_ROWS} Zellen")

    def toggle_grid(self, checked):
        """Ein/Ausblenden des Rasters."""
        self.pop_canvas.set_show_grid(checked)
        self.pop_canvas.update()

    def show_about(self):
        """Zeigt Informationen über die Anwendung."""
        QMessageBox.information(self, "Über", "Fabrikplaner GA\nVersion 1.0\nEinfacher 2D Fabrikplaner mit genetischem Algorithmus\n\n(c) 2023 Ihr Name")

    def on_next_generation(self):
        """Fortschritt um 'advance_spin' Generationen voranschreiten (GAEngine erforderlich)."""
        if self.ga_engine is None:
            # initialisiere GAEngine mit Total (falls leer)
            total = int(self.generations_spin.value())
            self.ga_engine = GAEngine(total)
            # initiale Anzeige der Startpopulation
            self.pop_canvas.set_population(self.ga_engine.population)
            self.gencounter_label.setText(f"{self.ga_engine.generation}/{self.ga_engine.total_generations}")

        advance = int(self.advance_spin.value())
        show_every = int(self.show_best_spin.value())

        last_best_score = None
        for _ in range(advance):
            if self.ga_engine.generation >= self.ga_engine.total_generations:
                break
            best_score, best_ind = self.ga_engine.step()
            last_best_score = best_score
            # Update Population-Anzeige
            try:
                self.pop_canvas.set_population(self.ga_engine.population)
                self.pop_canvas.update()
            except Exception:
                pass
            # Update counter
            self.gencounter_label.setText(f"{self.ga_engine.generation}/{self.ga_engine.total_generations}")
            self.status_label.setText(f"Gen {self.ga_engine.generation}: bester Score {self.ga_engine.best_score:.2f}")
            QApplication.processEvents()

            # optional best-Dialog nach Intervall
            if show_every > 0 and (self.ga_engine.generation % show_every == 0):
                if self.ga_engine.best_ind:
                    # show only if no BestDialog is already open
                    open_already = any(isinstance(w, BestDialog) and w.isVisible() for w in QApplication.topLevelWidgets())
                    if not open_already:
                        dialog = BestDialog(self.ga_engine.best_ind, parent=self, title=f"Beste Lösung nach Generation {self.ga_engine.generation}")
                        dialog.exec()

        # Falls Ende erreicht -> zeige Ergebnis
        if self.ga_engine.generation >= self.ga_engine.total_generations:
            if self.ga_engine.best_ind:
                open_already = any(isinstance(w, BestDialog) and w.isVisible() for w in QApplication.topLevelWidgets())
                if not open_already:
                    dialog = BestDialog(self.ga_engine.best_ind, parent=self, title=f"Beste Lösung (Ende Gen {self.ga_engine.generation})")
                    dialog.exec()
            self.next_gen_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            # sicherstellen, dass Stop-Button deaktiviert ist
            try:
                self.stop_btn.setEnabled(False)
            except Exception:
                pass

    def on_entry_exit_changed(self):
        global ENTRY_CELL, EXIT_CELL
        ENTRY_CELL = (int(self.entry_col.value()), int(self.entry_row.value()))
        EXIT_CELL = (int(self.exit_col.value()), int(self.exit_row.value()))
        self.status_label.setText(f"Entry/Exit gesetzt: {ENTRY_CELL} -> {EXIT_CELL}")
        # Aktualisiere Canvas (sowohl große als auch Mini-Canvas)
        try:
            self.pop_canvas.update()
        except Exception:
            pass

    def start_ga(self):
        """Starte kompletten GA-Lauf (blockierend). Initialisiert GAEngine und läuft total_gen Generationen."""
        # Start GA in background thread (nicht-blockierend)
        global POPULATION_SIZE, ELITE_KEEP, MUTATION_PROB, MUTATION_POS_STD, MUTATION_ANGLE_STD, MACHINE_COUNT
        # Werte aus UI übernehmen
        MACHINE_COUNT = int(self.machine_count_spin.value())
        rebuild_machine_sizes(MACHINE_COUNT)
        POPULATION_SIZE = int(self.population_size_spin.value())
        ELITE_KEEP = int(self.elite_keep_spin.value())
        MUTATION_PROB = float(self.mutation_prob_spin.value())
        MUTATION_POS_STD = float(self.mutation_pos_std_spin.value())
        MUTATION_ANGLE_STD = float(self.mutation_angle_std_spin.value())
        gens = int(self.generations_spin.value())

        # UI sperren
        self.start_btn.setEnabled(False)
        self.next_gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("GA läuft (Hintergrund)...")
        QApplication.processEvents()

        # Deaktiviere UI-Elemente, die Lauf-Konfiguration ändern können
        try:
            self.machine_count_spin.setEnabled(False)
            self.size_button.setEnabled(False)
            self.population_size_spin.setEnabled(False)
            self.elite_keep_spin.setEnabled(False)
            self.mutation_prob_spin.setEnabled(False)
            self.mutation_pos_std_spin.setEnabled(False)
            self.mutation_angle_std_spin.setEnabled(False)
            self.generations_spin.setEnabled(False)
            self.advance_spin.setEnabled(False)
            self.show_best_spin.setEnabled(False)
            self.factory_button.setEnabled(False)
            self.entry_col.setEnabled(False)
            self.entry_row.setEnabled(False)
            self.exit_col.setEnabled(False)
            self.exit_row.setEnabled(False)
        except Exception:
            pass

        # Verhindere Mehrfachstart, falls bereits ein Thread läuft
        if hasattr(self, '_ga_thread') and getattr(self, '_ga_thread') is not None and self._ga_thread.isRunning():
            return

        write_crash_log(f"start_ga: starting GA for {gens} gens; MACHINE_COUNT={MACHINE_COUNT}")

        # Synchronous (blocking) GA run in main thread (reverted from threaded mode)
        # progress_callback wird verwendet, um die GUI während des Laufes zu aktualisieren
        def _cb(generation, total_generations, best_score_cb, best_ind_cb, population=None):
            try:
                # Reuse existing progress handler
                self._on_worker_progress(generation, total_generations, best_score_cb, best_ind_cb, population)
            except Exception:
                pass

        # Ensure stop flag reset and connect stop button to set the flag
        global STOP_REQUESTED
        STOP_REQUESTED = False
        try:
            # connect stop button to simple setter (removing previous connections if any)
            try:
                self.stop_btn.clicked.disconnect()
            except Exception:
                pass
            self.stop_btn.clicked.connect(lambda: globals().__setitem__('STOP_REQUESTED', True))
        except Exception:
            pass

        # Run GA synchronously; this will update GUI via callbacks
        try:
            best_ind, best_score = run_ga(gens, progress_callback=_cb)
        except Exception as e:
            try:
                with open('crash_log.txt', 'a', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            best_ind, best_score = None, float('inf')

        # Call finished handler synchronously
        try:
            self._on_worker_finished(best_ind, best_score)
        except Exception:
            pass

    def _ga_progress(self, generation, total_generations, best_score, best_ind, population=None):
        """Backward-compatible Progress-Handler (falls verwendet)."""
        try:
            self.gencounter_label.setText(f"{generation}/{total_generations}")
            self.status_label.setText(f"Gen {generation}/{total_generations} — bester Score: {best_score:.2f}")
            if population is not None:
                self.pop_canvas.set_population(population)
            elif best_ind is not None:
                # fallback: zeige best als single mini-layout
                self.pop_canvas.set_population([best_ind])
            QApplication.processEvents()
        except Exception:
            pass

    def _on_worker_progress(self, generation, total_generations, best_score, best_ind, population):
        write_crash_log(f"_on_worker_progress: gen={generation}/{total_generations} best={best_score}")
        try:
            # population könnte leer sein; nur wenn vorhanden übernehmen
            if population:
                self.pop_canvas.set_population(population)
            elif best_ind is not None:
                # fallback: zeige best als single mini-layout
                self.pop_canvas.set_population([best_ind])
            self.gencounter_label.setText(f"{generation}/{total_generations}")
            self.status_label.setText(f"Gen {generation}/{total_generations} — bester Score: {best_score:.2f}")
            QApplication.processEvents()
        except Exception as e:
            write_crash_log(f"Exception in _on_worker_progress: {e}")
            try:
                with open('crash_log.txt', 'a', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            QMessageBox.critical(self, 'Fehler', f'Fehler während GA-Fortschritt: {e}')

    def _on_worker_finished(self, best_ind, best_score):
        write_crash_log(f"_on_worker_finished: called best_present={best_ind is not None} best_score={best_score}")
        try:
            if best_ind:
                open_already = any(isinstance(w, BestDialog) and w.isVisible() for w in QApplication.topLevelWidgets())
                if not open_already:
                    write_crash_log("_on_worker_finished: opening BestDialog")
                    try:
                        dialog = BestDialog(best_ind, parent=self, title=f"Beste Lösung (Ende)")
                        dialog.exec()
                    except Exception as e:
                        write_crash_log(f"_on_worker_finished: error showing BestDialog: {e}")
            else:
                # Fallback: Zeige Informationsbox mit dem besten bekannten Score
                try:
                    write_crash_log("_on_worker_finished: no best_ind — showing informational message with best score")
                except Exception:
                    pass
                try:
                    QMessageBox.information(self, 'GA beendet', f'GA abgebrochen oder kein Ergebnis. Best score (bekannt): {best_score}')
                except Exception as e:
                    write_crash_log(f"_on_worker_finished: error showing info message: {e}")
        except Exception as e:
            write_crash_log(f"Exception in _on_worker_finished outer: {e}")
            try:
                with open('crash_log.txt', 'a', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass
            try:
                QMessageBox.critical(self, 'Fehler', f'Unerwarteter Fehler nach GA: {e}')
            except Exception:
                pass
        write_crash_log("_on_worker_finished: completed")

        # UI wieder aktivieren
        try:
            self.start_btn.setEnabled(True)
            self.next_gen_btn.setEnabled(True)
            # stop_btn Verbindung entfernen, falls gesetzt
            try:
                self.stop_btn.clicked.disconnect()
            except Exception:
                pass
            self.stop_btn.setEnabled(False)
        except Exception:
            pass

        # Re-enable configuration controls
        try:
            self.machine_count_spin.setEnabled(True)
            self.size_button.setEnabled(True)
            self.population_size_spin.setEnabled(True)
            self.elite_keep_spin.setEnabled(True)
            self.mutation_prob_spin.setEnabled(True)
            self.mutation_pos_std_spin.setEnabled(True)
            self.mutation_angle_std_spin.setEnabled(True)
            self.generations_spin.setEnabled(True)
            self.advance_spin.setEnabled(True)
            self.show_best_spin.setEnabled(True)
            self.factory_button.setEnabled(True)
            self.entry_col.setEnabled(True)
            self.entry_row.setEnabled(True)
            self.exit_col.setEnabled(True)
            self.exit_row.setEnabled(True)
        except Exception:
            pass

        # Thread beenden und aufräumen (falls noch aktiv)
        try:
            if hasattr(self, '_ga_thread') and self._ga_thread is not None:
                try:
                    if self._ga_thread.isRunning():
                        self._ga_thread.quit()
                        self._ga_thread.wait(2000)
                except RuntimeError:
                    pass
                try:
                    self._ga_thread.deleteLater()
                except Exception:
                    pass
                self._ga_thread = None
        except Exception:
            pass

        # Worker referenz entfernen
        try:
            if hasattr(self, '_ga_worker'):
                self._ga_worker = None
        except Exception:
            pass

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = MainWindow()
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('ERROR starting GUI:', e)
        raise
