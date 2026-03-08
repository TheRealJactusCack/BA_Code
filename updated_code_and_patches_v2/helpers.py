# HELPERS MODULE

from __future__ import annotations

import math
import random
import heapq
import hashlib
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import config



def update_grid_counts() -> None:
    """Aktualisiert GRID_COLS und GRID_ROWS basierend auf FLOOR_W H und GRID_SIZE."""
    config.GRID_COLS = max(1, int(config.FLOOR_W // config.GRID_SIZE))
    config.GRID_ROWS = max(1, int(config.FLOOR_H // config.GRID_SIZE))

    try:
        config.ENTRY_CELL = (int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]))
    except Exception:
        print("helpers: 24")
        config.ENTRY_CELL = (0, 0)

    try:
        config.EXIT_CELL = (
            min(int(config.EXIT_CELL[0]), config.GRID_COLS - 1),
            min(int(config.EXIT_CELL[1]), config.GRID_ROWS - 1),
        )
    except Exception:
        print("helpers: 34")
        config.EXIT_CELL = (config.GRID_COLS - 1, config.GRID_ROWS - 1)

    #print(f"GRID_COLS={config.GRID_COLS} GRID_ROWS={config.GRID_ROWS}")


def rect_corners(center: Tuple[float, float], w: float, h: float, angle_deg: int) -> List[Tuple[float, float]]:
    """Vier Eckpunkte eines Rechtecks mit Rotation um center."""
    cx, cy = center
    a = math.radians(int(angle_deg) % 360)
    dx = w / 2.0
    dy = h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    ca = math.cos(a)
    sa = math.sin(a)
    out = []
    for x, y in pts:
        rx = x * ca - y * sa + cx
        ry = x * sa + y * ca + cy
        out.append((rx, ry))
    return out

#=========================================================================================
#=========================== Maschinen gruppieren abhängig von Knopf =====================

def _group_machines(best_ind: List[Dict]) -> List[List[Dict]]:
    """Hier werden die Maschinen abhängig von dem gewünschten Kriterium miteinander gruppiert"""
    #Anfangs immer 2 Gruppieren
    for i, (a, b) in enumerate(zip(best_ind[::2], best_ind[1::2])):
        """
        So lange verschieben bis lokales Minimum
        """  
        #Kostenberechung zwischen beiden Maschinen
        """
        cost = abs(a["gx"] - b["gx"]) + abs(a["gy"] - b["gy"]) #Dummy Kostenfunktion, hier muss die tatsächliche Kostenfunktion rein
        previous_cost = 1e12
        while True:
            previous_cost = cost
            new_cost = abs((a["gx"] + 1) - b["gx"]) + abs(a["gy"] - b["gy"]) #Dummy Kostenfunktion, hier muss die tatsächliche Kostenfunktion rein
            if new_cost < cost:
                a["gx"] += 1
                cost = new_cost
            new_cost = abs((a["gx"] - 1) - b["gx"]) + abs(a["gy"] - b["gy"]) #Dummy Kostenfunktion, hier muss die tatsächliche Kostenfunktion rein
            if new_cost < cost:
                a["gx"] -= 1
                cost = new_cost
            new_cost = abs(a["gx"] - b["gx"]) + abs((a["gy"] + 1) - b["gy"]) #Dummy Kostenfunktion, hier muss die tatsächliche Kostenfunktion rein
            if new_cost < cost:
                a["gy"] += 1
                cost = new_cost
            new_cost = abs(a["gx"] - b["gx"]) + abs((a["gy"] - 1) - b["gy"]) #Dummy Kostenfunktion, hier muss die tatsächliche Kostenfunktion rein
            if new_cost < cost:
                a["gy"] -= 1
                cost = new_cost

            # Snapshot
            old = (a["z"], a["gx"], a["gy"], a.get("x"), a.get("y"))

            # Kandidat berechnen (OHNE sofort zu committen wäre ideal – hier mit Snapshot/Revert)
            cx = float(a["gx"])
            cy = float(a["gy"])

            cand_z = (a["z"] + 90) % 360
            w_eff, h_eff = effective_dims(a, cand_z)

            gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
            cand_gx = int(round((cx / gs) - (float(w_eff) / 2.0)))
            cand_gy = int(round((cy / gs) - (float(h_eff) / 2.0)))

            # Kandidat anwenden
            a["z"] = int(cand_z)
            a["gx"] = cand_gx
            a["gy"] = cand_gy
            a["x"], a["y"] = cell_center_from_topleft(cand_gx, cand_gy, int(w_eff), int(h_eff))

            new_cost = abs(a["gx"] - b["gx"]) + abs(a["gy"] - b["gy"])

            if new_cost < cost:
                cost = new_cost
            else:
                # Revert
                a["z"], a["gx"], a["gy"], a["x"], a["y"] = old

            if cost >= previous_cost:
                break"""


            #Brutal schlechtr Code btw hir drüber

def cell_center_from_topleft(col: int, row: int, w_cells: int, h_cells: int) -> Tuple[float, float]:
    """Zentrum in Metern für eine Maschine deren Top Left Zelle col row ist."""
    x = (col + w_cells / 2.0) * config.GRID_SIZE
    y = (row + h_cells / 2.0) * config.GRID_SIZE
    return x, y


def effective_dims(m_or_w_h, z: Optional[int] = None) -> Tuple[int, int]:
    """Effektive Dimensionen in Zellen für Rotation 0 oder 90."""
    if isinstance(m_or_w_h, (tuple, list)):
        w_cells, h_cells = int(m_or_w_h[0]), int(m_or_w_h[1])
    else:
        w_cells = int(m_or_w_h["w_cells"])
        h_cells = int(m_or_w_h["h_cells"])
        if z is None:
            z = int(m_or_w_h.get("z", 0))

    z = int(z or 0) % 180
    if z == 90 or z == 270:
        return h_cells, w_cells
    return w_cells, h_cells


def clearance_pad_cells() -> int:
    """DEPRECATED: war früher Maschinen-Randpuffer. Wird nicht mehr für Platzierung verwendet."""
    gap_m = float(getattr(config, "MACHINE_CLEARANCE_M", 0.0) or 0.0)
    if gap_m <= 0.0:
        return 0
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    return int(math.ceil((gap_m / 2.0) / gs))


def occupied_cells(m: Dict, *, pad_cells: int | None = None) -> set[Tuple[int, int]]:
    """Raster-Fußabdruck der Maschine.

    Important:
        Standardmäßig wird **kein** zusätzlicher Puffer addiert (pad_cells=None -> 0).
        Damit kollidieren Maschinen nur über ihren echten Footprint. Clearance wird separat
        am Worker-Punkt berechnet.
    """
    w_eff, h_eff = effective_dims(m, int(m.get("z", 0)))
    p = 0 if pad_cells is None else int(pad_cells)

    gx = int(m["gx"]) - p
    gy = int(m["gy"]) - p
    w = int(w_eff) + 2 * p
    h = int(h_eff) + 2 * p

    out: set[Tuple[int, int]] = set()
    for dx in range(w):
        for dy in range(h):
            out.add((gx + dx, gy + dy))
    return out


def worker_clearance_width_m() -> float:
    """Breite (Quadrat) der Clearance am Worker-Punkt in Metern."""
    try:
        return float(getattr(config, "MACHINE_CLEARANCE_M", 0.0) or 0.0)
    except Exception:
        return 0.0


def worker_path_width_m() -> float:
    """Breite der Worker-Wege in Metern (Default 1.0m)."""
    try:
        v = getattr(config, "WORKER_PATH_WIDTH_M", None)
        return float(1.0 if v is None else v)
    except Exception:
        return 1.0


def _norm_rect(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float, float, float]:
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _cells_intersecting_rect(x0: float, y0: float, x1: float, y1: float) -> set[Tuple[int, int]]:
    """Alle Grid-Zellen, deren Zellfläche das Rechteck schneidet (Weltkoordinaten)."""
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    rx0, ry0, rx1, ry1 = _norm_rect(x0, y0, x1, y1)

    c0 = int(math.floor(rx0 / gs)) - 1
    r0 = int(math.floor(ry0 / gs)) - 1
    c1 = int(math.ceil(rx1 / gs)) + 1
    r1 = int(math.ceil(ry1 / gs)) + 1

    c0 = max(0, c0)
    r0 = max(0, r0)
    c1 = min(int(config.GRID_COLS) - 1, c1)
    r1 = min(int(config.GRID_ROWS) - 1, r1)

    out: set[Tuple[int, int]] = set()
    for col in range(c0, c1 + 1):
        for row in range(r0, r1 + 1):
            cx0 = col * gs
            cy0 = row * gs
            cx1 = (col + 1) * gs
            cy1 = (row + 1) * gs
            if not (cx1 <= rx0 or cx0 >= rx1 or cy1 <= ry0 or cy0 >= ry1):
                out.add((col, row))
    return out


def worker_clearance_rect_world(m: Dict, clearance_m: float | None = None) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned Rechteck (Welt) der Worker-Clearance außerhalb der Maschine.

    Das Rechteck liegt *außerhalb* an der Worker-Seite und hat Größe clearance_m x clearance_m.
    Clearance kann sich mit anderer Clearance überlappen, aber nicht mit Maschinenfootprints/Obstacles.
    """
    wpt = machine_worker_point(m)
    if wpt is None:
        return None

    cm = worker_clearance_width_m() if clearance_m is None else float(clearance_m)
    if cm <= 0.0:
        return None

    idx = int(m.get("idx", 0))
    workers = getattr(config, "MACHINE_WORKERS", [])
    wd = workers[idx] if idx < len(workers) else None
    if not wd:
        return None

    side_raw = wd.get("side_worker", None)
    if side_raw in (None, ""):
        return None

    side_eff = _effective_side_for_rotation(str(side_raw), int(m.get("z", 0)))

    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    w_eff, h_eff = effective_dims(m, int(m.get("z", 0)))
    w_m = float(w_eff) * gs
    h_m = float(h_eff) * gs

    cx = float(m["x"])
    cy = float(m["y"])
    left = cx - w_m / 2.0
    right = cx + w_m / 2.0
    top = cy - h_m / 2.0
    bottom = cy + h_m / 2.0

    wx, wy = float(wpt[0]), float(wpt[1])

    half = cm / 2.0
    if side_eff == "left":
        return _norm_rect(left - cm, wy - half, left, wy + half)
    if side_eff == "right":
        return _norm_rect(right, wy - half, right + cm, wy + half)
    if side_eff == "top":
        return _norm_rect(wx - half, top - cm, wx + half, top)
    # bottom
    return _norm_rect(wx - half, bottom, wx + half, bottom + cm)


def worker_clearance_cells(m: Dict, clearance_m: float | None = None) -> set[Tuple[int, int]]:
    """Worker-Clearance Zellen (außerhalb Maschine)."""
    rect = worker_clearance_rect_world(m, clearance_m=clearance_m)
    if rect is None:
        return set()
    x0, y0, x1, y1 = rect
    cells = _cells_intersecting_rect(x0, y0, x1, y1)
    # sicherheitshalber Maschinenfläche ausnehmen
    cells -= occupied_cells(m, pad_cells=0)
    return cells


def worker_clearance_cells_for_ind(ind: List[Dict]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for m in ind:
        out |= worker_clearance_cells(m)
    return out


def corridor_cells_for_path_cells(path_cells: Sequence[Tuple[int, int]], width_m: float) -> set[Tuple[int, int]]:
    """Markiert alle Zellen, die vom Worker-Weg (Zellenpfad) mit Breite width_m belegt werden."""
    if not path_cells:
        return set()

    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    half = float(width_m) / 2.0

    def center(col: int, row: int) -> Tuple[float, float]:
        return ((col + 0.5) * gs, (row + 0.5) * gs)

    out: set[Tuple[int, int]] = set()

    if len(path_cells) == 1:
        col, row = path_cells[0]
        cx, cy = center(col, row)
        out |= _cells_intersecting_rect(cx - gs / 2.0, cy - half, cx + gs / 2.0, cy + half)
        out |= _cells_intersecting_rect(cx - half, cy - gs / 2.0, cx + half, cy + gs / 2.0)
        return out

    for (c0, r0), (c1, r1) in zip(path_cells[:-1], path_cells[1:]):
        x0, y0 = center(c0, r0)
        x1, y1 = center(c1, r1)

        if c0 == c1:
            # vertikaler Schritt: Breite in x, Länge in y (+gs/2 für Zellflächen)
            out |= _cells_intersecting_rect(x0 - half, min(y0, y1) - gs / 2.0, x0 + half, max(y0, y1) + gs / 2.0)
        elif r0 == r1:
            # horizontaler Schritt: Breite in y
            out |= _cells_intersecting_rect(min(x0, x1) - gs / 2.0, y0 - half, max(x0, x1) + gs / 2.0, y0 + half)
        else:
            # sollte bei 4-neighborhood nicht passieren, fallback: beide Achsen
            out |= _cells_intersecting_rect(min(x0, x1) - half, min(y0, y1) - half, max(x0, x1) + half, max(y0, y1) + half)

    return out

def normalize_individual(ind: List[Dict]) -> None:
    """Normalisiert gx gy x y z."""
    for m in ind:
        m["gx"] = int(round(m.get("gx", 0)))
        m["gy"] = int(round(m.get("gy", 0)))
        z = int(m.get("z", 0)) % 360
        if z not in config.ROTATIONS:
            z = config.ROTATIONS[0]
        m["z"] = z
        w_eff, h_eff = effective_dims(m, z)
        max_col = max(0, config.GRID_COLS - int(w_eff))
        max_row = max(0, config.GRID_ROWS - int(h_eff))
        m["gx"] = max(0, min(max_col, m["gx"]))
        m["gy"] = max(0, min(max_row, m["gy"]))
        m["x"], m["y"] = cell_center_from_topleft(m["gx"], m["gy"], w_eff, h_eff)


def can_place_at(col: int, row: int, w_cells: int, h_cells: int, occupied_set: set[Tuple[int, int]]) -> bool:
    #True wenn keine Kollision mit obstacles oder occupied_set
    for dx in range(int(w_cells)):
        for dy in range(int(h_cells)):
            cell = (int(col) + dx, int(row) + dy)
            if cell in config.OBSTACLES:
                return False
            if cell in occupied_set:
                return False
    return True

#===========================================================================
#===================Hier wird die Dict für Maschinen definiert==============
#===========================================================================

def random_machine_nonoverlap(
    idx: int,
    *,
    machine_cells: set[Tuple[int, int]],
    clearance_no_machines: set[Tuple[int, int]],
    obstacle_cells: set[Tuple[int, int]],
    max_attempts: int = 200,
) -> Dict:
    """Platziert Maschine idx ohne Überlappung.

    Constraints:
        - Maschinen-Footprints dürfen sich nicht überlappen.
        - Maschinen-Footprints dürfen nicht in Obstacles oder in Worker-Clearance liegen.
        - Worker-Clearance darf andere Worker-Clearance überlappen, aber nicht Maschinen/Obstacles.
    """
    w_m, d_m = config.MACHINE_SIZES[idx]
    w_cells = max(1, int(round(float(w_m) / config.GRID_SIZE)))
    h_cells = max(1, int(round(float(d_m) / config.GRID_SIZE)))

    fixed_list = getattr(config, "MACHINE_FIXED", [])
    fixed = fixed_list[idx] if idx < len(fixed_list) else None

    def footprint_cells_for(col: int, row: int, w_eff: int, h_eff: int) -> set[Tuple[int, int]]:
        out: set[Tuple[int, int]] = set()
        for dx in range(int(w_eff)):
            for dy in range(int(h_eff)):
                out.add((int(col) + dx, int(row) + dy))
        return out

    def footprint_ok(fp: set[Tuple[int, int]]) -> bool:
        # bounds + obstacles + clearance + other machines
        for c in fp:
            if c[0] < 0 or c[1] < 0 or c[0] >= int(config.GRID_COLS) or c[1] >= int(config.GRID_ROWS):
                return False
            if c in obstacle_cells:
                return False
            if c in clearance_no_machines:
                return False
            if c in machine_cells:
                return False
        return True

    def clearance_ok(cl: set[Tuple[int, int]]) -> bool:
        # clearance darf nicht in obstacles oder Maschinen liegen; overlap mit anderer clearance ist erlaubt
        for c in cl:
            if c[0] < 0 or c[1] < 0 or c[0] >= int(config.GRID_COLS) or c[1] >= int(config.GRID_ROWS):
                return False
            if c in obstacle_cells:
                return False
            if c in machine_cells:
                return False
        return True

    if fixed is not None:
        z_fixed = fixed.get("z", None)
        z = int(z_fixed) if z_fixed is not None else 0

        w_eff, h_eff = effective_dims((w_cells, h_cells), z)
        x = float(fixed["x"])
        y = float(fixed["y"])

        gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
        gx = int(round((x - 0.5 * float(w_eff) * gs) / gs))
        gy = int(round((y - 0.5 * float(h_eff) * gs) / gs))

        cand = {
            "idx": int(idx),
            "x": x,
            "y": y,
            "z": int(z),
            "gx": int(gx),
            "gy": int(gy),
            "w_cells": int(w_cells),
            "h_cells": int(h_cells),
        }

        fp = occupied_cells(cand, pad_cells=0)
        cl = worker_clearance_cells(cand)

        if not footprint_ok(fp) or not clearance_ok(cl):
            # Fixed kann aus Excel kommen; wir crashen nicht, aber Layout wird in fitness hart bestraft.
            print(f"[WARN] Fixed machine idx={idx} verletzt Constraints (Footprint/Clearance).")

        machine_cells |= fp
        clearance_no_machines |= cl
        return cand

    for _ in range(int(max_attempts)):
        z = random.choice(config.ROTATIONS)
        w_eff, h_eff = effective_dims((w_cells, h_cells), z)

        max_col = max(0, int(config.GRID_COLS) - int(w_eff))
        max_row = max(0, int(config.GRID_ROWS) - int(h_eff))

        gx = random.randint(0, max_col)
        gy = random.randint(0, max_row)

        x, y = cell_center_from_topleft(gx, gy, int(w_eff), int(h_eff))
        cand = {
            "idx": int(idx),
            "x": float(x),
            "y": float(y),
            "z": int(z),
            "gx": int(gx),
            "gy": int(gy),
            "w_cells": int(w_cells),
            "h_cells": int(h_cells),
        }

        fp = occupied_cells(cand, pad_cells=0)
        if not footprint_ok(fp):
            continue

        cl = worker_clearance_cells(cand)
        if not clearance_ok(cl):
            continue

        machine_cells |= fp
        clearance_no_machines |= cl
        return cand

    # Fallback: random (kann invalid sein -> fitness straft)
    z = random.choice(config.ROTATIONS)
    w_eff, h_eff = effective_dims((w_cells, h_cells), z)
    max_col = max(0, int(config.GRID_COLS) - int(w_eff))
    max_row = max(0, int(config.GRID_ROWS) - int(h_eff))
    gx = random.randint(0, max_col)
    gy = random.randint(0, max_row)
    x, y = cell_center_from_topleft(gx, gy, int(w_eff), int(h_eff))
    cand = {
        "idx": int(idx),
        "x": float(x),
        "y": float(y),
        "z": int(z),
        "gx": int(gx),
        "gy": int(gy),
        "w_cells": int(w_cells),
        "h_cells": int(h_cells),
    }
    machine_cells |= occupied_cells(cand, pad_cells=0)
    clearance_no_machines |= worker_clearance_cells(cand)
    return cand


def random_individual() -> List[Dict]:
    """Start Individuum mit MACHINE_COUNT Maschinen.

    Fixed machines (aus Excel) werden zuerst gesetzt, damit Zufallsmaschinen sie nicht blockieren.
    """
    obstacle_cells: set[Tuple[int, int]] = set(getattr(config, "OBSTACLES", set()) or set())
    machine_cells: set[Tuple[int, int]] = set()
    clearance_no_machines: set[Tuple[int, int]] = set()

    n = int(config.MACHINE_COUNT)
    ind: List[Optional[Dict]] = [None] * n

    fixed_list = getattr(config, "MACHINE_FIXED", []) or []
    fixed_indices = [i for i in range(n) if i < len(fixed_list) and fixed_list[i] is not None]
    free_indices = [i for i in range(n) if i not in set(fixed_indices)]

    for i in fixed_indices:
        ind[i] = random_machine_nonoverlap(
            i,
            machine_cells=machine_cells,
            clearance_no_machines=clearance_no_machines,
            obstacle_cells=obstacle_cells,
            max_attempts=1,
        )

    for i in free_indices:
        ind[i] = random_machine_nonoverlap(
            i,
            machine_cells=machine_cells,
            clearance_no_machines=clearance_no_machines,
            obstacle_cells=obstacle_cells,
            max_attempts=250,
        )

    out: List[Dict] = [m for m in ind if m is not None]
    normalize_individual(out)
    return out

def _parse_side(side: str) -> str:
    s = str(side or "").strip().lower()
    if s in {"top", "bottom", "left", "right"}:
        return s
    else:
        s = "top"
        return s
    raise ValueError(f"Ungültige side {side}. Erlaubt Top Bottom Left Right")


def port_world_xy(
    *,
    center_x: float,
    center_y: float,
    w_m: float,
    d_m: float,
    side: str,
    offset_m: float,
    rotation_deg: int,
) -> Tuple[float, float]:
    """
    Port Koordinate aus w d side offset rotation
    Rotation nur 0, 90, 180, 270 Grad
    Offset ist immer für Rotation 0 definiert
    """
    side_n = _parse_side(side)
    rot = int(rotation_deg) % 360
    if rot not in (0, 90, 180, 270):
        raise ValueError(f"Rotation {rotation_deg} ist nicht erlaubt. Nur 0, 90, 180 oder 270")

    w0 = float(w_m)
    d0 = float(d_m)
    off = float(offset_m)

    if side_n in {"left", "right"}:
        if off < 0.0 or off > d0:
            raise ValueError(f"Offset {off} ist außerhalb 0 bis {d0} für side {side}")
        y_local = (d0 / 2.0) - off
        x_local = (-w0 / 2.0) if side_n == "left" else (w0 / 2.0)
    else:
        if off < 0.0 or off > w0:
            raise ValueError(f"Offset {off} ist außerhalb 0 bis {w0} für side {side}")
        x_local = (-w0 / 2.0) + off
        y_local = (-d0 / 2.0) if side_n == "top" else (d0 / 2.0)

    a = math.radians(rot)
    ca = math.cos(a)
    sa = math.sin(a)
    x_rot = x_local * ca - y_local * sa
    y_rot = x_local * sa + y_local * ca

    return center_x + x_rot, center_y + y_rot


def _effective_side_for_rotation(side: str, rotation_deg: int) -> str:
    side_n = _parse_side(side)
    rot = int(rotation_deg) % 360
    if rot == 0:
        return side_n
    if rot == 90:
        mapping = {"top": "right", "right": "bottom", "bottom": "left", "left": "top"}
        return mapping[side_n]
    if rot == 180:
        mapping = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}
        return mapping[side_n]
    if rot == 270:
        mapping = {"top": "left", "right": "top", "bottom": "right", "left": "bottom"}
        return mapping[side_n]
    raise ValueError(f"Rotation {rotation_deg} ist nicht erlaubt. Nur 0, 90, 180 oder 270")

#=============================================================================================
#=========================0= Anfang der Port berechnung ======================================
#5 Funktionen für 6 Ports (dumm und entstanden durch das nacheinander hinzufügen der Ports) ==

def machine_water_point(m: Dict) -> Optional[Tuple[float, float]]:
    """Weltkoordinate des Wasseranschlusses einer Maschine. None, wenn kein Wasseranschluss existiert (Damit keine Fehler wegen paralellen Listen entstehen)"""
    idx = int(m.get("idx", 0))
    w_m, d_m = config.MACHINE_SIZES[idx]

    water = getattr(config, "MACHINE_WATER", [])
    wd = water[idx] if idx < len(water) else None
    if not wd:
        return None

    side_raw = wd.get("side_water", None)
    off_raw = wd.get("offset_water", None)
    if side_raw in (None, "") or off_raw is None:
        return None

    side = str(side_raw).strip().lower()
    offset = float(off_raw)

    return port_world_xy(
        center_x=float(m["x"]),
        center_y=float(m["y"]),
        w_m=float(w_m),
        d_m=float(d_m),
        side=side,
        offset_m=offset,
        rotation_deg=int(m.get("z", 0)),
    )

def machine_gas_point(m: Dict) -> Optional[Tuple[float, float]]:
    """Weltkoordinate des Gasanschlusses einer Maschine. None, wenn kein Gasanschluss existiert (Damit keine Fehler wegen paralellen Listen entstehen)"""
    idx = int(m.get("idx", 0))
    w_m, d_m = config.MACHINE_SIZES[idx]

    gas = getattr(config, "MACHINE_GAS", [])
    wd = gas[idx] if idx < len(gas) else None
    if not wd:
        return None

    side_raw = wd.get("side_gas", None)
    off_raw = wd.get("offset_gas", None)
    if side_raw in (None, "") or off_raw is None:
        return None

    side = str(side_raw).strip().lower()
    offset = float(off_raw)

    return port_world_xy(
        center_x=float(m["x"]),
        center_y=float(m["y"]),
        w_m=float(w_m),
        d_m=float(d_m),
        side=side,
        offset_m=offset,
        rotation_deg=int(m.get("z", 0)),
    )

def machine_other_point(m: Dict) -> Optional[Tuple[float, float]]:
    """Weltkoordinate der zusätzlichen Anschlüsse einer Maschine. None, wenn kein Anschluss existiert (Damit keine Fehler wegen paralellen Listen entstehen)"""
    idx = int(m.get("idx", 0))
    w_m, d_m = config.MACHINE_SIZES[idx]

    other = getattr(config, "MACHINE_OTHER", [])
    wd = other[idx] if idx < len(other) else None
    if not wd:
        return None

    side_raw = wd.get("side_other", None)
    off_raw = wd.get("offset_other", None)
    if side_raw in (None, "") or off_raw is None:
        return None

    side = str(side_raw).strip().lower()
    offset = float(off_raw)

    return port_world_xy(
        center_x=float(m["x"]),
        center_y=float(m["y"]),
        w_m=float(w_m),
        d_m=float(d_m),
        side=side,
        offset_m=offset,
        rotation_deg=int(m.get("z", 0)),
    )

def machine_worker_point(m: Dict) -> Optional[Tuple[float, float]]:
    """Weltkoordinate der Arbeiterstation einer Maschine. None, wenn kein Worker existiert (genau wie bei anschlüssen zum vermeiden von Fehlern wegen paralellen Listen)"""
    idx = int(m.get("idx", 0))
    w_m, d_m = config.MACHINE_SIZES[idx]

    workers = getattr(config, "MACHINE_WORKERS", [])
    wd = workers[idx] if idx < len(workers) else None
    if not wd:
        return None

    side_raw = wd.get("side_worker", None)
    off_raw = wd.get("offset_worker", None)
    if side_raw in (None, "") or off_raw is None:
        return None

    side = str(side_raw).strip().lower()
    offset = float(off_raw)

    return port_world_xy(
        center_x=float(m["x"]),
        center_y=float(m["y"]),
        w_m=float(w_m),
        d_m=float(d_m),
        side=side,
        offset_m=offset,
        rotation_deg=int(m.get("z", 0)),
    )

def machine_port_point(m: Dict, kind: str) -> Tuple[float, float]:
    """Weltkoordinate des Ports einer Maschine."""
    idx = int(m.get("idx", 0))
    w_m, d_m = config.MACHINE_SIZES[idx]
    ports = getattr(config, "MACHINE_PORTS", [])
    pd = ports[idx] if idx < len(ports) else {}

    if kind == "in":
        side = pd.get("side_in", "left")
        offset = pd.get("offset_in", float(d_m) / 2.0)
    else:
        side = pd.get("side_out", "right")
        offset = pd.get("offset_out", float(d_m) / 2.0)

    return port_world_xy(
        center_x=float(m["x"]),
        center_y=float(m["y"]),
        w_m=float(w_m),
        d_m=float(d_m),
        side=str(side),
        offset_m=float(offset),
        rotation_deg=int(m.get("z", 0)),
    )


def machine_input_point(m: Dict) -> Tuple[float, float]:
    return machine_port_point(m, "in")


def machine_output_point(m: Dict) -> Tuple[float, float]:
    return machine_port_point(m, "out")

#=============================================================================================
#=========================0= Ende Port berechnung ============================================
#=============================================================================================

def flow_footprint_cells(world_pts: Sequence[Tuple[float, float]], width_m: float) -> set[Tuple[int, int]]:
    """Rasterzellen die vom Flow Band mit konstanter Breite belegt werden."""
    if len(world_pts) < 2:
        return set()

    half = float(width_m) / 2.0
    gs = float(config.GRID_SIZE)

    out: set[Tuple[int, int]] = set()

    for i in range(len(world_pts) - 1):
        x1, y1 = world_pts[i]
        x2, y2 = world_pts[i + 1]

        if abs(x2 - x1) >= abs(y2 - y1):
            y = float(y1)
            x0 = min(x1, x2) - half
            x1b = max(x1, x2) + half
            y0 = y - half
            y1c = y + half
        else:
            x = float(x1)
            x0 = x - half
            x1b = x + half
            y0 = min(y1, y2) - half
            y1c = max(y1, y2) + half

        c0 = int(math.floor(x0 / gs))
        c1 = int(math.floor(x1b / gs))
        r0 = int(math.floor(y0 / gs))
        r1 = int(math.floor(y1c / gs))

        for col in range(max(0, c0), min(config.GRID_COLS - 1, c1) + 1):
            for row in range(max(0, r0), min(config.GRID_ROWS - 1, r1) + 1):
                cx = (col + 0.5) * gs
                cy = (row + 0.5) * gs
                if cx < x0 or cx > x1b or cy < y0 or cy > y1c:
                    continue
                out.add((col, row))

    return out

def _cell_of_world(x: float, y: float) -> Tuple[int, int]:
    """Map world meters to a grid cell (col,row), clamped into bounds."""
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    col = int(math.floor(float(x) / gs))
    row = int(math.floor(float(y) / gs))
    col = max(0, min(int(config.GRID_COLS) - 1, col))
    row = max(0, min(int(config.GRID_ROWS) - 1, row))
    return col, row


def _world_of_cell_center(col: int, row: int) -> Tuple[float, float]:
    """Center of a grid cell in world meters."""
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    return (float(col) + 0.5) * gs, (float(row) + 0.5) * gs


def _neighbors4(col: int, row: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if col > 0:
        out.append((col - 1, row))
    if col + 1 < int(config.GRID_COLS):
        out.append((col + 1, row))
    if row > 0:
        out.append((col, row - 1))
    if row + 1 < int(config.GRID_ROWS):
        out.append((col, row + 1))
    return out


def _blocked_signature(blocked: set[Tuple[int, int]]) -> int:
    """Deterministic signature for blocked cells (used as cache key)"""
    h = hashlib.blake2b(digest_size=8)
    for c, r in sorted(blocked):
        h.update(int(c).to_bytes(2, "little", signed=False))
        h.update(int(r).to_bytes(2, "little", signed=False))
    return int.from_bytes(h.digest(), "little", signed=False)


_ROUTE_CACHE: "OrderedDict[Tuple[int, Tuple[int,int], Tuple[int,int]], Optional[List[Tuple[int,int]]]]" = OrderedDict()


def _cache_get(key):
    try:
        val = _ROUTE_CACHE.pop(key)
    except KeyError:
        return None, False
    _ROUTE_CACHE[key] = val
    return val, True


def _cache_put(key, val) -> None:
    _ROUTE_CACHE[key] = val
    max_items = int(getattr(config, "ROUTE_CACHE_SIZE", 5000))
    while len(_ROUTE_CACHE) > max_items:
        _ROUTE_CACHE.popitem(last=False)


def _pick_free_cell_near(cell: Tuple[int, int], blocked: set[Tuple[int, int]], *, max_radius: int = 3) -> Tuple[int, int]:
    """If a cell is blocked, pick a nearby free cell (BFS within radius)."""
    if cell not in blocked:
        return cell

    sc, sr = cell
    q = deque([(sc, sr, 0)])
    seen = {(sc, sr)}
    while q:
        c, r, d = q.popleft()
        if d > max_radius:
            break
        for nc, nr in _neighbors4(c, r):
            if (nc, nr) in seen:
                continue
            seen.add((nc, nr))
            if (nc, nr) not in blocked:
                return (nc, nr)
            q.append((nc, nr, d + 1))
    return cell



def _cell_envelope_cells_for_width(cell: Tuple[int, int], width_m: float) -> set[Tuple[int, int]]:
    """Zellen, die durch einen *Punkt* mit Breite (Worker-Weg) belegt wären.

    Wir verwenden pad = width/2 + GRID_SIZE/2, damit das Ergebnis mit der Segment-Korridor-Logik konsistent ist.
    """
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    half = float(width_m) / 2.0
    pad = half + gs / 2.0
    x, y = _world_of_cell_center(int(cell[0]), int(cell[1]))
    return _cells_intersecting_rect(x - pad, y - pad, x + pad, y + pad)


def _is_cell_safe_for_width(cell: Tuple[int, int], blocked: set[Tuple[int, int]], width_m: float) -> bool:
    if cell in blocked:
        return False
    return not (_cell_envelope_cells_for_width(cell, width_m) & blocked)


def _pick_free_cell_near_wide(
    cell: Tuple[int, int],
    blocked: set[Tuple[int, int]],
    *,
    width_m: float,
    max_radius: int = 8,
) -> Tuple[int, int]:
    """Wie _pick_free_cell_near(), aber zusätzlich: Zelle muss auch für die gewünschte Wegbreite frei sein."""
    if _is_cell_safe_for_width(cell, blocked, width_m):
        return cell

    sc, sr = cell
    q = deque([(sc, sr, 0)])
    seen = {(sc, sr)}
    while q:
        c, r, d = q.popleft()
        if d > max_radius:
            break
        for nc, nr in _neighbors4(c, r):
            if (nc, nr) in seen:
                continue
            seen.add((nc, nr))
            if _is_cell_safe_for_width((nc, nr), blocked, width_m):
                return (nc, nr)
            q.append((nc, nr, d + 1))
    return cell


def _segment_corridor_cells(
    a: Tuple[int, int],
    b: Tuple[int, int],
    *,
    width_m: float,
) -> set[Tuple[int, int]]:
    """Zellen, die der Korridor (Breite width_m) für ein einzelnes 4-Nachbar-Segment belegt."""
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    half = float(width_m) / 2.0

    c0, r0 = int(a[0]), int(a[1])
    c1, r1 = int(b[0]), int(b[1])
    x0, y0 = _world_of_cell_center(c0, r0)
    x1, y1 = _world_of_cell_center(c1, r1)

    if c0 == c1:
        return _cells_intersecting_rect(x0 - half, min(y0, y1) - gs / 2.0, x0 + half, max(y0, y1) + gs / 2.0)
    if r0 == r1:
        return _cells_intersecting_rect(min(x0, x1) - gs / 2.0, y0 - half, max(x0, x1) + gs / 2.0, y0 + half)

    return _cells_intersecting_rect(min(x0, x1) - half, min(y0, y1) - half, max(x0, x1) + half, max(y0, y1) + half)


def _astar_path_wide(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: set[Tuple[int, int]],
    *,
    width_m: float,
) -> Optional[List[Tuple[int, int]]]:
    """A* (4-Nachbar), aber mit Wegbreite: jedes Segment darf mit seinem Korridor keine Blockaden schneiden."""
    if start == goal:
        return [start]

    blocked_local = set(blocked)
    blocked_local.discard(start)
    blocked_local.discard(goal)

    seg_ok_cache: dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}

    def seg_ok(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        k = (a, b) if a <= b else (b, a)
        v = seg_ok_cache.get(k, None)
        if v is not None:
            return v
        corridor = _segment_corridor_cells(a, b, width_m=width_m)
        ok = not (corridor & blocked_local)
        seg_ok_cache[k] = ok
        return ok

    def h(c: Tuple[int, int]) -> int:
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start), 0, start))

    gscore: dict[Tuple[int, int], int] = {start: 0}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        if g != gscore.get(cur, 10**12):
            continue

        for nb in _neighbors4(cur[0], cur[1]):
            if nb in blocked_local:
                continue
            if not seg_ok(cur, nb):
                continue
            ng = g + 1
            if ng < gscore.get(nb, 10**12):
                gscore[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))

    return None


def _astar_path_wide_cached(
    blocked_sig: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: set[Tuple[int, int]],
    *,
    width_m: float,
) -> Optional[List[Tuple[int, int]]]:
    """Wide-A* mit Cache: cache-key enthält width_m."""
    width_key = int(round(float(width_m) * 1000.0))
    key = ("wide", int(blocked_sig), int(width_key), tuple(start), tuple(goal))
    val, hit = _cache_get(key)
    if hit:
        return val
    val = _astar_path_wide(start, goal, blocked, width_m=width_m)
    _cache_put(key, val)
    return val


def _astar_path(start: Tuple[int, int], goal: Tuple[int, int], blocked: set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """Kürzester 4 Nachbar weg mit beachtung von Hindernissen und verwendung der Maschinen laufwege"""
    if start == goal:
        return [start]

    # Start/Ziel dürfen in "blocked" liegen (Ports liegen häufig in der Maschine).
    blocked_local = set(blocked)
    blocked_local.discard(start)
    blocked_local.discard(goal)

    def h(c: Tuple[int, int]) -> int:
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start), 0, start))

    gscore: dict[Tuple[int, int], int] = {start: 0}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        if g != gscore.get(cur, 10**12):
            continue

        for nb in _neighbors4(cur[0], cur[1]):
            if nb in blocked_local:
                continue
            ng = g + 1
            if ng < gscore.get(nb, 10**12):
                gscore[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))

    return None


#================ A* mit Cache =======================
# Checken ob sinn macht da gleicher input und gleicher output nicht heißt dass keine maschine dazwischen steht

def _astar_path_cached(blocked_sig: int, start: Tuple[int, int], goal: Tuple[int, int], blocked: set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """A* mit LRU-Cache: gleiche Inputs -> gleicher Output, ohne Neuberechnung"""
    key = (int(blocked_sig), tuple(start), tuple(goal))
    val, hit = _cache_get(key)
    if hit:
        return val
    val = _astar_path(start, goal, blocked)
    _cache_put(key, val)
    return val


def _compress_polyline(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Remove intermediate collinear points from a polyline."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        x0, y0 = out[-1]
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if (abs(x0 - x1) < 1e-9 and abs(x1 - x2) < 1e-9) or (abs(y0 - y1) < 1e-9 and abs(y1 - y2) < 1e-9):
            continue
        out.append((x1, y1))
    out.append(pts[-1])
    return out


def _blocked_cells_for_routing(ind: List[Dict]) -> set[Tuple[int, int]]:
    """Routing-Blockaden: OBSTACLES + Maschinen-Fußabdruck (ohne Clearance)."""
    blocked = set(getattr(config, "OBSTACLES", []) or [])

    # Entry/Exit sind immer begehbar, selbst wenn sie in OBSTACLES liegen.
    try:
        blocked.discard(tuple(config.ENTRY_CELL))
        blocked.discard(tuple(config.EXIT_CELL))
    except Exception:
        print("helpers: 539")
        pass

    for m in ind:
        blocked |= occupied_cells(m, pad_cells=0)  # <-- Clearance bleibt begehbar
    return blocked

def route_line_to_world(
    start_world: Tuple[float, float],
    end_world: Tuple[float, float],
    machine_world : Tuple[float, float],
    *,
    blocked: set[Tuple[int, int]],
    blocked_sig: int,
) -> Tuple[Optional[List[Tuple[float, float]]], float]:
    
    diff_x = end_world[0] - start_world[0]
    diff_y = end_world[1] - start_world[1]

    wx = machine_world[0] - start_world[0]
    wy = machine_world[1] - start_world[1]

    denom = diff_x * diff_x + diff_y * diff_y

    #Sonderfall: A und B sind identisch (keine Linie, nur ein Punkt)
    if denom == 0:
        dx = machine_world[0] - start_world[0]
        dy = machine_world[1] - start_world[1]
        d = math.sqrt(dx * dx + dy * dy)
        return (d, start_world)

    #Projektion: t gibt an, wo F relativ zu A->B liegt
    t = (wx * diff_x + wy * diff_y) / denom
    if t < 0: t = 0
    if t > 1: t = 1
    final_x = start_world[0] + t * diff_x
    final_y = start_world[1] + t * diff_y

    pts: List[Tuple[float, float]] = [machine_world]
    pts.append((final_x, final_y))

    #Abstand berechnen
    dx = machine_world[0] - final_x
    dy = machine_world[1] - final_y
    length = math.sqrt(dx * dx + dy * dy)

    return (pts, length)

def route_world_to_world(
    start_world: Tuple[float, float],
    end_world: Tuple[float, float],
    *,
    blocked: set[Tuple[int, int]],
    blocked_sig: int,
) -> Tuple[Optional[List[Tuple[float, float]]], float]:
    """Route on the grid from start_world to end_world, return (polyline_world, length_m)."""
    s_cell = _cell_of_world(*start_world)
    e_cell = _cell_of_world(*end_world)

    # Start/Ziel aus Maschinenfläche/Obstacle heraus auf freie Nachbarzelle schieben.
    s_cell = _pick_free_cell_near(s_cell, blocked, max_radius=3)
    e_cell = _pick_free_cell_near(e_cell, blocked, max_radius=3)

    cells = _astar_path_cached(blocked_sig, s_cell, e_cell, blocked)
    
    #Debugging output
    #print(f"[ROUTE] start_world={start_world} end_world={end_world} " f"start_cell={s_cell} goal_cell={e_cell} "f"path_len_cells={(len(cells) if cells else None)} "f"path_head={(cells[:8] if cells else None)}")
    if not cells:
        return None, float("inf")

    centers = [_world_of_cell_center(c, r) for (c, r) in cells]

    pts: List[Tuple[float, float]] = [start_world]
    pts.extend(centers)
    pts.append(end_world)

    pts = _compress_polyline(pts)

    length_m = (len(cells) - 1) * float(config.GRID_SIZE)
    return pts, float(length_m)


def route_world_to_world_cells(
    start_world: Tuple[float, float],
    end_world: Tuple[float, float],
    *,
    blocked: set[Tuple[int, int]],
    blocked_sig: int,
    width_m: float | None = None,
) -> Tuple[Optional[List[Tuple[float, float]]], float, Optional[List[Tuple[int, int]]]]:
    """Route on the grid from start_world to end_world.

    Returns:
        (polyline_world, length_m, path_cells)

    Wenn width_m gesetzt ist (>0), wird der A*-Weg *direkt* mit dieser Breite berechnet:
    jedes Segment wird nur zugelassen, wenn der Korridor (Breite width_m) keine Blockaden schneidet.
    """
    s_cell = _cell_of_world(*start_world)
    e_cell = _cell_of_world(*end_world)

    wm = None
    try:
        wm = None if width_m is None else float(width_m)
    except Exception:
        wm = None

    if wm is not None and wm > 0.0:
        gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
        # Worker-Ports liegen oft direkt an Maschinen; Radius daher etwas großzügiger.
        max_r = int(max(8, math.ceil((wm / gs) + 4)))
        s_cell = _pick_free_cell_near_wide(s_cell, blocked, width_m=wm, max_radius=max_r)
        e_cell = _pick_free_cell_near_wide(e_cell, blocked, width_m=wm, max_radius=max_r)
        cells = _astar_path_wide_cached(blocked_sig, s_cell, e_cell, blocked, width_m=wm)
    else:
        s_cell = _pick_free_cell_near(s_cell, blocked, max_radius=3)
        e_cell = _pick_free_cell_near(e_cell, blocked, max_radius=3)
        cells = _astar_path_cached(blocked_sig, s_cell, e_cell, blocked)

    if not cells:
        return None, float("inf"), None

    centers = [_world_of_cell_center(c, r) for (c, r) in cells]

    pts: List[Tuple[float, float]] = [start_world]
    pts.extend(centers)
    pts.append(end_world)

    pts = _compress_polyline(pts)

    length_m = (len(cells) - 1) * float(config.GRID_SIZE)
    return pts, float(length_m), list(cells)

def compute_routed_edges(ind: List[Dict]) -> Dict[str, List[Dict]]:
    """Berechne geroutete Polylinien für alle Connection-Typen.

    Zusätzlich für Worker:
        - width_m (default 1.0m via config.WORKER_PATH_WIDTH_M)
        - cells (A*-Zellenpfad)
        - clearance_cells (Zellen, die durch den Weg mit Breite belegt werden)
    """
    n = len(ind)
    blocked = _blocked_cells_for_routing(ind)
    blocked_sig = _blocked_signature(blocked)

    entry_world = cell_center_from_topleft(int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]), 1, 1)
    exit_world = cell_center_from_topleft(int(config.EXIT_CELL[0]), int(config.EXIT_CELL[1]), 1, 1)

    out: Dict[str, List[Dict]] = {"material": [], "worker": [], "water": [], "gas": [], "other": []}

    # ---------------- Material ----------------
    edges = list(getattr(config, "MATERIAL_CONNECTIONS", []) or [])
    for e in edges:
        if not e or len(e) < 2:
            continue
        a = e[0]
        b = e[1]
        w = 1.0
        if len(e) >= 3 and e[2] is not None:
            try:
                w = float(e[2])
            except Exception:
                w = 1.0

        a_idx: Optional[int] = None if a is None else int(a)
        b_idx: Optional[int] = None if b is None else int(b)

        if a_idx is None:
            p1 = entry_world
        else:
            if not (0 <= a_idx < n):
                continue
            p1 = machine_output_point(ind[a_idx])

        if b_idx is None:
            p2 = exit_world
        else:
            if not (0 <= b_idx < n):
                continue
            p2 = machine_input_point(ind[b_idx])

        pts, length_m = route_world_to_world(p1, p2, blocked=blocked, blocked_sig=blocked_sig)
        out["material"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m), "weight": float(w)})

    # ---------------- Worker ----------------
    width_m = float(worker_path_width_m())
    w_edges = list(getattr(config, "WORKER_CONNECTIONS", []) or [])
    for a, b in w_edges:
        a_idx: Optional[int] = None if a is None else int(a)
        b_idx: Optional[int] = None if b is None else int(b)
        if a_idx is None or b_idx is None:
            continue
        if not (0 <= a_idx < n and 0 <= b_idx < n):
            continue

        p1 = machine_worker_point(ind[a_idx])
        p2 = machine_worker_point(ind[b_idx])
        if p1 is None or p2 is None:
            continue

        pts, length_m, cells = route_world_to_world_cells(p1, p2, blocked=blocked, blocked_sig=blocked_sig, width_m=width_m)
        clearance_cells = corridor_cells_for_path_cells(cells or [], width_m)
        out["worker"].append(
            {
                "a": a_idx,
                "b": b_idx,
                "pts": pts,
                "length_m": float(length_m),
                "width_m": width_m,
                "cells": cells,
                "clearance_cells": clearance_cells,
            }
        )

    # ---------------- Optional Water/Gas/Other ----------------
    if getattr(config, "WATER_CELL", [None, None])[0] is not None and getattr(config, "WATER_CELL", [None, None])[1] is not None:
        water_edges = list(getattr(config, "WATER_CONNECTIONS", []) or [])
        w1 = config.WATER_CELL[0]
        w2 = config.WATER_CELL[1]
        for a, b in water_edges:
            a_idx: Optional[int] = None if a is None else int(a)
            b_idx: Optional[int] = None if b is None else int(b)
            if a_idx is None or b_idx is None:
                continue
            if not (0 <= b_idx < n):
                continue
            p2 = machine_water_point(ind[b_idx])
            if p2 is None:
                continue
            pts, length_m = route_line_to_world(w1, w2, p2, blocked=blocked, blocked_sig=blocked_sig)
            out["water"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    if getattr(config, "GAS_CELL", [None, None])[0] is not None and getattr(config, "GAS_CELL", [None, None])[1] is not None:
        gas_edges = list(getattr(config, "GAS_CONNECTIONS", []) or [])
        g1 = config.GAS_CELL[0]
        g2 = config.GAS_CELL[1]
        for a, b in gas_edges:
            a_idx: Optional[int] = None if a is None else int(a)
            b_idx: Optional[int] = None if b is None else int(b)
            if a_idx is None or b_idx is None:
                continue
            if not (0 <= b_idx < n):
                continue
            p2 = machine_gas_point(ind[b_idx])
            if p2 is None:
                continue
            pts, length_m = route_line_to_world(g1, g2, p2, blocked=blocked, blocked_sig=blocked_sig)
            out["gas"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    if getattr(config, "OTHER_CELL", [None, None])[0] is not None and getattr(config, "OTHER_CELL", [None, None])[1] is not None:
        other_edges = list(getattr(config, "OTHER_CONNECTIONS", []) or [])
        o1 = config.OTHER_CELL[0]
        o2 = config.OTHER_CELL[1]
        for a, b in other_edges:
            a_idx: Optional[int] = None if a is None else int(a)
            b_idx: Optional[int] = None if b is None else int(b)
            if a_idx is None or b_idx is None:
                continue
            if not (0 <= b_idx < n):
                continue
            p2 = machine_other_point(ind[b_idx])
            if p2 is None:
                continue
            pts, length_m = route_line_to_world(o1, o2, p2, blocked=blocked, blocked_sig=blocked_sig)
            out["other"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    return out

def distance_cost(ind: List[Dict], config: any, *, routed: Optional[Dict] = None) -> float:
    """Summe aller Flusslängen (Material + Arbeiter) im Individuum.

    routed kann vorab berechnet werden (compute_routed_edges), um doppelte Routing-Arbeit
    in fitness() zu vermeiden.
    """
    if routed is None:
        routed = compute_routed_edges(ind)

    no_path_penalty = float(getattr(config, "NO_PATH_PENALTY", 1e6))

    cost = 0.0
    for m in routed.get("material", []):
        w = float(m.get("weight", 1.0) or 1.0)
        length = float(m.get("length_m", float("inf")))
        cost += no_path_penalty if not math.isfinite(length) else length * w

    for e in routed.get("worker", []):
        length = float(e.get("length_m", float("inf")))
        cost += no_path_penalty if not math.isfinite(length) else length * float(getattr(config, "WORKER_WEIGHT", 1.0))

    for w in routed.get("water", []):
        length = float(w.get("length_m", float("inf")))
        cost += no_path_penalty if not math.isfinite(length) else length * float(getattr(config, "WATER_WEIGHT", 1.0))

    for g in routed.get("gas", []):
        length = float(g.get("length_m", float("inf")))
        cost += no_path_penalty if not math.isfinite(length) else length * float(getattr(config, "GAS_WEIGHT", 1.0))

    for o in routed.get("other", []):
        length = float(o.get("length_m", float("inf")))
        cost += no_path_penalty if not math.isfinite(length) else length * float(getattr(config, "OTHER_WEIGHT", 1.0))

    return float(cost)

