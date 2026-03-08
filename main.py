print("Test1")
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QDoubleSpinBox, QCheckBox,
    QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QSpinBox,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QGridLayout
)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QColor, QTransform, QPen, QBrush
import sys
import random
import math
import copy

# -------------------------
# Konfiguration / Parameter (Defaultwerte)
# -------------------------
POPULATION_SIZE = 50
ELITE_KEEP = 25               # Anzahl der besten Layouts, die überleben
MACHINE_COUNT = 6             # Default; wird in UI einstellbar

# Raumgröße (Floor) in Einheiten (Meter)
FLOOR_W = 30.0
FLOOR_H = 20.0

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
            if (col + dx, row + dy) in occupied_set:
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
    """Erstellt ein Layout/Individuum mit MACHINE_COUNT Maschinen; initial möglichst ohne Überschneidung."""
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
def random_machine(idx):
    """
    Erzeugt zufällige Maschine i:
    - gx/gy = top-left Zellkoordinate (so dass die Maschine vollständig im Raster liegt)
    - x/y = Mittelpunkt in Metern
    - z = Rotation aus ROTATIONS
    - w_cells/h_cells werden aus MACHINE_SIZES[idx] geladen
    """
    w_cells, h_cells = MACHINE_SIZES[idx]
    max_col = max(0, GRID_COLS - w_cells)
    max_row = max(0, GRID_ROWS - h_cells)
    col = random.randint(0, max_col)
    row = random.randint(0, max_row)
    x, y = cell_center_from_topleft(col, row, w_cells, h_cells)
    z = random.choice(ROTATIONS)
    return {'x': x, 'y': y, 'z': z, 'gx': int(col), 'gy': int(row), 'w_cells': w_cells, 'h_cells': h_cells}

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

    # 1) Distanzkosten entlang der Kette (Mitte zu Mitte)
    for i in range(MACHINE_COUNT - 1):
        a = ind[i]
        b = ind[i+1]
        dx = a['x'] - b['x']
        dy = a['y'] - b['y']
        cost += DIST_SCALE * math.hypot(dx, dy)

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
        cost += 0.5 * math.hypot(first['x'] - entry_x, first['y'] - entry_y)
        cost += 0.5 * math.hypot(last['x'] - exit_x, last['y'] - exit_y)

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
    Tatsächlicher GA:
    - Erzeuge Startpopulation
    - Bewerte, wähle die besten ELITE_KEEP aus
    - Erzeuge neue Population: behalte ELITE_KEEP, erzeuge Kinder durch Crossover+Mutation
      bis POPULATION_SIZE erreicht ist.
    """
    update_grid_counts()
    pop = init_population()
    best_ind = None
    best_score = float('inf')

    global STOP_REQUESTED

    for g in range(1, generations + 1):
        if STOP_REQUESTED:
            if progress_callback:
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

        painter.restore()

class AreaWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flächen-Parameter mit echtem GA (Raster)")
        self.init_ui()

    def init_ui(self):
        # Breite / Höhe (als Double erlaubt, wird intern auf ganze Meter gerundet)
        w_label = QLabel("Breite (m):")
        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(1, 1e6)
        self.w_spin.setDecimals(3)
        self.w_spin.setSingleStep(0.1)
        self.w_spin.setValue(FLOOR_W)

        h_label = QLabel("Höhe (m):")
        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(1, 1e6)
        self.h_spin.setDecimals(3)
        self.h_spin.setSingleStep(0.1)
        self.h_spin.setValue(FLOOR_H)

        wh_layout = QHBoxLayout()
        wh_layout.addWidget(w_label)
        wh_layout.addWidget(self.w_spin)
        wh_layout.addWidget(h_label)
        wh_layout.addWidget(self.h_spin)

        # Maschinenanzahl / Größen bearbeiten
        mc_label = QLabel("Anzahl Maschinen:")
        self.mc_spin = QSpinBox()
        self.mc_spin.setRange(1, 200)
        self.mc_spin.setValue(MACHINE_COUNT)
        self.mc_spin.valueChanged.connect(self.on_machine_count_changed)

        sizes_btn = QPushButton("Maschinengrößen bearbeiten")
        sizes_btn.clicked.connect(self.on_edit_sizes)

        # Entry / Exit Einstellungen
        entry_lbl = QLabel("Entry (col,row):")
        self.entry_col = QSpinBox(); self.entry_col.setRange(0, max(0, GRID_COLS-1)); self.entry_col.setValue(ENTRY_CELL[0])
        self.entry_row = QSpinBox(); self.entry_row.setRange(0, max(0, GRID_ROWS-1)); self.entry_row.setValue(ENTRY_CELL[1])
        exit_lbl = QLabel("Exit (col,row):")
        self.exit_col = QSpinBox(); self.exit_col.setRange(0, max(0, GRID_COLS-1)); self.exit_col.setValue(EXIT_CELL[0])
        self.exit_row = QSpinBox(); self.exit_row.setRange(0, max(0, GRID_ROWS-1)); self.exit_row.setValue(EXIT_CELL[1])

        entry_layout = QHBoxLayout()
        entry_layout.addWidget(mc_label)
        entry_layout.addWidget(self.mc_spin)
        entry_layout.addWidget(sizes_btn)
        entry_layout.addStretch()
        entry_layout.addWidget(entry_lbl)
        entry_layout.addWidget(self.entry_col)
        entry_layout.addWidget(self.entry_row)
        entry_layout.addWidget(exit_lbl)
        entry_layout.addWidget(self.exit_col)
        entry_layout.addWidget(self.exit_row)

        # GA-Steuerung
        gen_label = QLabel("Generationen:")
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 100000)
        self.gen_spin.setValue(200)
        self.start_btn = QPushButton("GA starten")
        self.start_btn.clicked.connect(self.on_start_ga)

        self.cancel_btn = QPushButton("Abbrechen")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel)

        self.status_label = QLabel("Bereit.")
        ga_ctrl_layout = QHBoxLayout()
        ga_ctrl_layout.addWidget(gen_label)
        ga_ctrl_layout.addWidget(self.gen_spin)
        ga_ctrl_layout.addWidget(self.start_btn)
        ga_ctrl_layout.addWidget(self.cancel_btn)
        ga_ctrl_layout.addWidget(self.status_label)

        self.canvas = LayoutCanvas()
        apply_btn = QPushButton("Anwenden / Anzeigen")
        apply_btn.clicked.connect(self.on_apply)

        main_layout = QVBoxLayout()
        main_layout.addLayout(wh_layout)
        main_layout.addLayout(entry_layout)
        main_layout.addLayout(ga_ctrl_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(apply_btn)
        self.setLayout(main_layout)

    def on_machine_count_changed(self, v):
        """Bei Änderung der Maschinenanzahl MACHINE_SIZES und Spinboxen anpassen."""
        global MACHINE_COUNT
        MACHINE_COUNT = int(v)
        rebuild_machine_sizes(MACHINE_COUNT)

    def on_edit_sizes(self):
        """Öffnet Dialog zum Bearbeiten der Maschinen-Größen."""
        dlg = SizesDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            sizes = dlg.get_sizes()
            global MACHINE_SIZES
            MACHINE_SIZES = sizes[:MACHINE_COUNT]
            if len(MACHINE_SIZES) < MACHINE_COUNT:
                MACHINE_SIZES += [(1,1)] * (MACHINE_COUNT - len(MACHINE_SIZES))

    def _exclusive_toggle(self, source: QCheckBox, other: QCheckBox, state: bool):
        if state and other.isChecked():
            other.setChecked(False)

    def on_apply(self):
        # Update Floor + Grid + Entry/Exit
        global FLOOR_W, FLOOR_H, ENTRY_CELL, EXIT_CELL
        # erlaubt Komma-Eingaben, erzwingt aber ganze Meter intern (Runden)
        FLOOR_W = int(round(float(self.w_spin.value())))
        FLOOR_H = int(round(float(self.h_spin.value())))
        # aktualisiere Spinbox-Anzeige auf gerundete Werte
        self.w_spin.setValue(float(FLOOR_W))
        self.h_spin.setValue(float(FLOOR_H))

        update_grid_counts()
        # Entry/Exit Spinbox-Grenzen anpassen
        self.entry_col.setRange(0, max(0, GRID_COLS-1))
        self.entry_row.setRange(0, max(0, GRID_ROWS-1))
        self.exit_col.setRange(0, max(0, GRID_COLS-1))
        self.exit_row.setRange(0, max(0, GRID_ROWS-1))

        ENTRY_CELL = (int(self.entry_col.value()), int(self.entry_row.value()))
        EXIT_CELL = (int(self.exit_col.value()), int(self.exit_row.value()))
        self.status_label.setText("Einstellungen übernommen.")
        self.canvas.update()

    def on_start_ga(self):
        # Übernehme aktuelle Parameter vor Start
        self.on_apply()
        gens = int(self.gen_spin.value())
        global STOP_REQUESTED, MACHINE_COUNT
        STOP_REQUESTED = False
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("GA läuft...")
        QApplication.processEvents()

        def progress_callback(g, generations, best_score, best_ind):
            if STOP_REQUESTED:
                self.status_label.setText(f"Abbruch angefordert — Generation {g}/{generations}")
            else:
                self.status_label.setText(f"Generation {g}/{generations} — bester Score: {best_score:.2f}")
            self.canvas.set_layout(best_ind)
            QApplication.processEvents()

        best_ind, best_score = run_ga(gens, progress_callback)

        if STOP_REQUESTED:
            self.status_label.setText(f"Abgebrochen. Bester Score bisher: {best_score:.2f}")
        else:
            self.status_label.setText(f"Fertig. Bester Score: {best_score:.2f}")
        self.canvas.set_layout(best_ind)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_cancel(self):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        self.status_label.setText("Abbruch angefordert...")
        self.cancel_btn.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AreaWidget()
    window.show()
    sys.exit(app.exec())
