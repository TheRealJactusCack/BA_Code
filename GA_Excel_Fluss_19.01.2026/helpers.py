# HELPERS MODULE

from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import config

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QTransform, QPolygonF, QFont



def update_grid_counts() -> None:
    """Aktualisiert GRID_COLS und GRID_ROWS basierend auf FLOOR_W H und GRID_SIZE."""
    config.GRID_COLS = max(1, int(config.FLOOR_W // config.GRID_SIZE))
    config.GRID_ROWS = max(1, int(config.FLOOR_H // config.GRID_SIZE))

    try:
        config.ENTRY_CELL = (int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]))
    except Exception:
        config.ENTRY_CELL = (0, 0)

    try:
        config.EXIT_CELL = (
            min(int(config.EXIT_CELL[0]), config.GRID_COLS - 1),
            min(int(config.EXIT_CELL[1]), config.GRID_ROWS - 1),
        )
    except Exception:
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
    """Padding pro Seite in Zellen. 1m Kantenabstand => 0.5m pro Maschine."""
    gap_m = float(getattr(config, "MACHINE_CLEARANCE_M", 0.0) or 0.0)
    if gap_m <= 0.0:
        return 0
    gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
    return int(math.ceil((gap_m / 2.0) / gs))


def occupied_cells(m: Dict, *, pad_cells: int | None = None) -> set[Tuple[int, int]]:
    """
    Raster-Fußabdruck der Maschine.
    pad_cells erweitert den Fußabdruck um pad_cells auf jeder Seite (für Laufwege).
    """
    w_eff, h_eff = effective_dims(m)  # Rotation: w/h tauschen :contentReference[oaicite:2]{index=2}
    p = clearance_pad_cells() if pad_cells is None else int(pad_cells)

    gx = int(m["gx"]) - p
    gy = int(m["gy"]) - p
    w = int(w_eff) + 2 * p
    h = int(h_eff) + 2 * p

    out: set[Tuple[int, int]] = set()
    for dx in range(w):
        for dy in range(h):
            out.add((gx + dx, gy + dy))
    return out


def normalize_individual(ind: List[Dict]) -> None:
    """Normalisiert gx gy x y z."""
    for m in ind:
        m["gx"] = int(round(m.get("gx", 0)))
        m["gy"] = int(round(m.get("gy", 0)))
        z = int(m.get("z", 0))
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

def random_machine_nonoverlap(idx: int, occupied_set: set[Tuple[int, int]], max_attempts: int = 200) -> Dict:
    """Platziert Maschine idx ohne Überlappung."""
    w_m, d_m = config.MACHINE_SIZES[idx]
    w_cells = max(1, int(round(float(w_m) / config.GRID_SIZE)))
    h_cells = max(1, int(round(float(d_m) / config.GRID_SIZE)))

    z = random.choice(config.ROTATIONS)
    # helpers.py (innerhalb random_machine_nonoverlap)
    p = clearance_pad_cells()  # <- Laufweg-Puffer in Zellen pro Seite

    w_eff, h_eff = effective_dims((w_cells, h_cells), z)

    # Für die Randbegrenzung müssen wir die gepufferte Fläche berücksichtigen,
    # sonst würde die echte Maschine zwar im Grid liegen, aber der "Laufweg" ragt raus.
    w_eff_p = int(w_eff) + 2 * p
    h_eff_p = int(h_eff) + 2 * p
    max_col = max(0, config.GRID_COLS - w_eff_p)
    max_row = max(0, config.GRID_ROWS - h_eff_p)

    for _ in range(max_attempts):
        col = random.randint(0, max_col)
        row = random.randint(0, max_row)

        # Testet die gepufferte Fläche: top-left um p nach außen verschieben.
        # Wichtig: gx/gy bleiben die echten Maschinen-zellen (ohne Puffer),
        # damit Center/Ports/Zeichnung korrekt bleiben.
        if can_place_at(col, row, w_eff_p, h_eff_p, occupied_set):
            gx = int(col + p)
            gy = int(row + p)
            x, y = cell_center_from_topleft(gx, gy, w_eff, h_eff)
            return {
                "idx": int(idx),
                "x": x,
                "y": y,
                "z": int(z),
                "gx": gx,
                "gy": gy,
                "w_cells": int(w_cells),
                "h_cells": int(h_cells),
            }

    # Fallback (wenn keine Platzierung gefunden)
    col = random.randint(0, max_col)
    row = random.randint(0, max_row)
    gx = int(col + p)
    gy = int(row + p)
    x, y = cell_center_from_topleft(gx, gy, w_eff, h_eff)
    return {
        "idx": int(idx),
        "x": x,
        "y": y,
        "z": int(z),
        "gx": gx,
        "gy": gy,
        "w_cells": int(w_cells),
        "h_cells": int(h_cells),
    }

def random_individual() -> List[Dict]:
    """Start Individuum mit MACHINE_COUNT Maschinen."""
    occupied: set[Tuple[int, int]] = set()
    ind: List[Dict] = []
    for i in range(int(config.MACHINE_COUNT)):
        m = random_machine_nonoverlap(i, occupied, max_attempts=250)
        occupied |= occupied_cells(m)
        ind.append(m)
    normalize_individual(ind)
    return ind


def _parse_side(side: str) -> str:
    s = str(side or "").strip().lower()
    if s in {"top", "bottom", "left", "right"}:
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
    rot = int(rotation_deg)
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
    rot = int(rotation_deg)
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

def machine_worker_point(m: Dict) -> Tuple[float, float]:
    """Weltkoordinate der Arbeiterstation einer Maschine."""
    idx = int(m.get("idx", 0))
    w_m, d_m = config.MACHINE_SIZES[idx]
    workers = getattr(config, "MACHINE_WORKERS", [])
    wd = workers[idx] if idx < len(workers) else {} 

    side = str(wd.get("side_worker", "top")).strip().lower()
    default_off = (float(d_m) / 2.0) if side in {"left", "right"} else (float(w_m) / 2.0)
    offset = float(wd.get("offset_worker", default_off))

    return port_world_xy(
        center_x = float(m["x"]),
        center_y = float(m["y"]),
        w_m = float(w_m),
        d_m = float(d_m),
        side = side,
        offset_m = offset,
        rotation_deg = int(m.get("z", 0)),
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

def distance_cost(ind: List[Dict], config: any) -> float:
    cost = 0.0
    n = len(ind)

    w_edges = list(getattr(config, "WORKER_CONNECTIONS", []))
    edges = list(getattr(config, "MATERIAL_CONNECTIONS", []))
    entry_world = cell_center_from_topleft(int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]), 1, 1)
    exit_world = cell_center_from_topleft(int(config.EXIT_CELL[0]), int(config.EXIT_CELL[1]), 1, 1)

    # Hier werden die Kosten berechnet basierend auf der Distanz zwischen den Maschinen
    for a,b in edges:
        a_idx: Optional[int] = None if a is None else int(a)
        b_idx: Optional[int] = None if b is None else int(b)
        # Startpunkt
        if a_idx is None:
            ax, ay = entry_world
        else:
            if not (0 <= a_idx < n):
                continue
            ax, ay = machine_output_point(ind[a_idx])
        # Zielpunkt
        if b_idx is None:
            bx, by = exit_world
        else:
            if not (0 <= b_idx < n):
                continue
            bx, by = machine_input_point(ind[b_idx])        
        cost += abs(ax - bx) + abs(ay - by)

    # Hier werden die Kosten berechnet basierend auf der Distanz zwischen den Maschinen (Worker)
    for a,b in w_edges:
        a_idx: Optional[int] = None if a is None else int(a)
        b_idx: Optional[int] = None if b is None else int(b)

        # Startpunkt
        if not (0 <= a_idx < n) or a_idx is None:
            continue
        wax, way = machine_worker_point(ind[a_idx])
        # Zielpunkt
        if not (0 <= b_idx < n) or b_idx is None:
            continue
        wbx, wby = machine_worker_point(ind[b_idx])        
        cost += abs(wax - wbx) + abs(way - wby)
    return cost

Edge = Tuple[Optional[int], Optional[int]]