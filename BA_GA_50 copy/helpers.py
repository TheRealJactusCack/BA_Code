# -------------------------
# HELPERS MODULE
# -------------------------
# Enthält alle Hilfsfunktionen für Berechnungen und Kollisionserkennung

import math
import random
import config
from PyQt6.QtGui import QPainter, QColor, QPolygonF, QBrush, QPen
from PyQt6.QtCore import Qt, QPointF




def update_grid_counts():
    """Aktualisiert GRID_COLS und GRID_ROWS basierend auf FLOOR_W/H und GRID_SIZE."""
    import config
    config.GRID_COLS = max(1, int(config.FLOOR_W // config.GRID_SIZE))
    config.GRID_ROWS = max(1, int(config.FLOOR_H // config.GRID_SIZE))
    # adjust entry/exit to remain valid integers within new grid
    try:
        config.ENTRY_CELL = (int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]))
    except Exception:
        config.ENTRY_CELL = (0, 0)
    try:
        config.EXIT_CELL = (min(int(config.EXIT_CELL[0]), config.GRID_COLS - 1),
                            min(int(config.EXIT_CELL[1]), config.GRID_ROWS - 1))
    except Exception:
        config.EXIT_CELL = (config.GRID_COLS - 1, config.GRID_ROWS - 1)
    print(f"GRID_COLS={config.GRID_COLS} GRID_ROWS={config.GRID_ROWS}")

def rebuild_machine_sizes(count):
    """(Re-)initialisiert MACHINE_SIZES-Liste wenn Maschinenanzahl geändert wird."""
    import config
    old = list(config.MACHINE_SIZES)
    config.MACHINE_SIZES = []
    for i in range(count):
        if i < len(old):
            config.MACHINE_SIZES.append(old[i])
        else:
            config.MACHINE_SIZES.append((1.0, 1.0))


def rect_corners(center, w, h, angle_deg):
    """Berechnet die vier Eckpunkte (x,y) eines Rechtecks mit Rotation (w,h in Metern)."""
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
    """Zeichnet ein kleines Dreieck als Pfeilspitze."""
    dx = x2 - x1
    dy = y2 - y1
    ang = math.atan2(dy, dx)
    bx = x2 - length * math.cos(ang)
    by = y2 - length * math.sin(ang)
    side = length * 0.4
    pvx = -math.sin(ang) * side
    pvy = math.cos(ang) * side
    leftx = bx + pvx
    lefty = by + pvy
    rightx = bx - pvx
    righty = by - pvy
    try:
        pts = [QPointF(x2, y2), QPointF(leftx, lefty), QPointF(rightx, righty)]
        poly = QPolygonF(pts)
        painter.save()
        painter.setBrush(painter.pen().color())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(poly)
        painter.restore()
    except Exception:
        pass


def machine_output_point(m):
    """Gibt (x,y) des Output-Anschlusses der Maschine m in Weltkoordinaten zurück."""
    cx = m['x']
    cy = m['y']
    rot = math.radians(int(m.get('z', 0)) % 360)
    w_m = m['w_cells'] * config.GRID_SIZE
    lx, ly = (w_m / 2.0, 0.0)
    ox = cx + math.cos(rot) * lx - math.sin(rot) * ly
    oy = cy + math.sin(rot) * lx + math.cos(rot) * ly
    return ox, oy


def machine_input_point(m):
    """Gibt (x,y) des Input-Anschlusses der Maschine m in Weltkoordinaten zurück."""
    cx = m['x']
    cy = m['y']
    rot = math.radians(int(m.get('z', 0)) % 360)
    w_m = m['w_cells'] * config.GRID_SIZE
    lx, ly = (-w_m / 2.0, 0.0)
    ix = cx + math.cos(rot) * lx - math.sin(rot) * ly
    iy = cy + math.sin(rot) * lx + math.cos(rot) * ly
    return ix, iy


def cell_center_from_topleft(col, row, w_cells, h_cells):
    """Berechnet Zentrum (x,y in m) einer Maschine, die an (col,row) als top-left Zelle verankert ist."""
    x = (col + w_cells / 2.0) * config.GRID_SIZE
    y = (row + h_cells / 2.0) * config.GRID_SIZE
    return x, y


def effective_dims(m_or_w_h, z=None):
    """Gibt (w_eff, h_eff) in Zellen zurück; tauscht bei 90/270° die Dimensionen."""
    if isinstance(m_or_w_h, (tuple, list)):
        w_cells, h_cells = m_or_w_h
    else:
        w_cells = m_or_w_h['w_cells']
        h_cells = m_or_w_h['h_cells']
        if z is None:
            z = m_or_w_h.get('z', 0)

    if z is None:
        z = 0
    if int(z) % 180 == 90:
        return (h_cells, w_cells)
    else:
        return (w_cells, h_cells)


def occupied_cells(m):
    """Gibt Menge der (col,row)-Zellen zurück, die Maschine m belegt."""
    w_eff, h_eff = effective_dims(m)
    cells = set()
    for dx in range(int(w_eff)):
        for dy in range(int(h_eff)):
            cells.add((int(m['gx']) + dx, int(m['gy']) + dy))
    return cells


def normalize_individual(ind):
    """Stellt sicher, dass gx/gy ganzzahlige Zellkoordinaten sind und x/y aus ihnen berechnet werden."""
    for m in ind:
        m['gx'] = int(round(m.get('gx', 0)))
        m['gy'] = int(round(m.get('gy', 0)))
        w_eff, h_eff = effective_dims(m)
        max_col = max(0, config.GRID_COLS - int(w_eff))
        max_row = max(0, config.GRID_ROWS - int(h_eff))
        m['gx'] = max(0, min(max_col, m['gx']))
        m['gy'] = max(0, min(max_row, m['gy']))
        m['x'], m['y'] = cell_center_from_topleft(m['gx'], m['gy'], w_eff, h_eff)
        if m.get('z') not in config.ROTATIONS:
            m['z'] = config.ROTATIONS[0]


def can_place_at(col, row, w_cells, h_cells, occupied_set):
    """Gibt True zurück falls die Zellen Platz haben (keine Überschneidung)."""
    for dx in range(int(w_cells)):
        for dy in range(int(h_cells)):
            cell = (col + dx, row + dy)
            if cell in config.OBSTACLES:
                return False
            if cell in occupied_set:
                return False
    return True


def random_machine_nonoverlap(idx, occupied_set, max_attempts=200):
    """Versucht, Maschine idx so zu platzieren, dass sie nicht mit occupied_set überlappt."""
    # MACHINE_SIZES are stored as meters; convert to grid cell counts
    w_m, h_m = config.MACHINE_SIZES[idx]
    w_cells = max(1, int(round(w_m / config.GRID_SIZE)))
    h_cells = max(1, int(round(h_m / config.GRID_SIZE)))
    z = random.choice(config.ROTATIONS)
    w_eff, h_eff = effective_dims((w_cells, h_cells), z)
    max_col = max(0, config.GRID_COLS - int(w_eff))
    max_row = max(0, config.GRID_ROWS - int(h_eff))

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


def random_individual():
    """Erstellt ein Layout/Individuum mit MACHINE_COUNT Maschinen."""
    occupied = set()
    ind = []
    for i in range(config.MACHINE_COUNT):
        m = random_machine_nonoverlap(i, occupied, max_attempts=250)
        w_eff, h_eff = effective_dims(m)
        for dx in range(int(w_eff)):
            for dy in range(int(h_eff)):
                occupied.add((m['gx'] + dx, m['gy'] + dy))
        ind.append(m)
    normalize_individual(ind)
    return ind
