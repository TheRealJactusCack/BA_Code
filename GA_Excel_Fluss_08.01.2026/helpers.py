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

    print(f"GRID_COLS={config.GRID_COLS} GRID_ROWS={config.GRID_ROWS}")


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
    if z == 90:
        return h_cells, w_cells
    return w_cells, h_cells


def occupied_cells(m: Dict) -> set[Tuple[int, int]]:
    """Raster Fußabdruck der Maschine."""
    w_eff, h_eff = effective_dims(m)
    out: set[Tuple[int, int]] = set()
    gx = int(m["gx"])
    gy = int(m["gy"])
    for dx in range(int(w_eff)):
        for dy in range(int(h_eff)):
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
    w_eff, h_eff = effective_dims((w_cells, h_cells), z)
    max_col = max(0, config.GRID_COLS - int(w_eff))
    max_row = max(0, config.GRID_ROWS - int(h_eff))

    for _ in range(max_attempts):
        col = random.randint(0, max_col)
        row = random.randint(0, max_row)
        if can_place_at(col, row, w_eff, h_eff, occupied_set):
            x, y = cell_center_from_topleft(col, row, w_eff, h_eff)
            return {
                "idx": int(idx),
                "x": x,
                "y": y,
                "z": int(z),
                "gx": int(col),
                "gy": int(row),
                "w_cells": int(w_cells),
                "h_cells": int(h_cells),
            }

    col = random.randint(0, max_col)
    row = random.randint(0, max_row)
    x, y = cell_center_from_topleft(col, row, w_eff, h_eff)
    return {
        "idx": int(idx),
        "x": x,
        "y": y,
        "z": int(z),
        "gx": int(col),
        "gy": int(row),
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
    Port Koordinate aus w d side offset rotation.
    Rotation nur 0 oder 90.
    Offset ist immer für Rotation 0 definiert.
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


def _clamp_cell(col: int, row: int) -> Tuple[int, int]:
    c = max(0, min(config.GRID_COLS - 1, int(col)))
    r = max(0, min(config.GRID_ROWS - 1, int(row)))
    return c, r


def get_terminal_cell(m: Dict, kind: str = "out") -> Tuple[int, int]:
    """Zelle zum Routing Start oder Ziel."""
    px, py = machine_output_point(m) if kind == "out" else machine_input_point(m)

    col = int(px // config.GRID_SIZE)
    row = int(py // config.GRID_SIZE)
    col, row = _clamp_cell(col, row)

    occ = occupied_cells(m)
    if (col, row) not in occ:
        return col, row

    idx = int(m.get("idx", 0))
    ports = getattr(config, "MACHINE_PORTS", [])
    pd = ports[idx] if idx < len(ports) else {}

    if kind == "out":
        side0 = str(pd.get("side_out", "right"))
    else:
        side0 = str(pd.get("side_in", "left"))

    side_eff = _effective_side_for_rotation(side0, int(m.get("z", 0)))

    dc, dr = 0, 0
    if side_eff == "right":
        dc = 1
    elif side_eff == "left":
        dc = -1
    elif side_eff == "bottom":
        dr = 1
    elif side_eff == "top":
        dr = -1

    return _clamp_cell(col + dc, row + dr)

#===========================================================================
#===================Pfad mit Überlappungen==================================
#===========================================================================

def find_manhattan_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked_cells: Optional[set[Tuple[int, int]]] = None,) -> Optional[List[Tuple[int, int]]]:
    """BFS auf Raster mit 4 Nachbarn."""
    blocked = blocked_cells or set()
    s = (int(start[0]), int(start[1]))
    g = (int(goal[0]), int(goal[1]))
    if s == g:
        return [s]

    cols = int(config.GRID_COLS)
    rows = int(config.GRID_ROWS)

    q = deque([s])
    came = {s: None}
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        cur = q.popleft()
        for dc, dr in neigh:
            nc = cur[0] + dc
            nr = cur[1] + dr
            if nc < 0 or nr < 0 or nc >= cols or nr >= rows:
                continue
            nxt = (nc, nr)
            if nxt in came:
                continue
            if nxt in blocked and nxt != g:
                continue
            came[nxt] = cur
            if nxt == g:
                path = [g]
                curp = cur
                while curp is not None:
                    path.append(curp)
                    curp = came[curp]
                path.reverse()
                return path
            q.append(nxt)

    return None

#===========================================================================
#===================Pfad ohne Überlappungen==================================
#===========================================================================

def find_manhattan_path_nooverlap(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked_cells: Optional[set[Tuple[int, int]]] = None,) -> Optional[List[Tuple[int, int]]]:
    
    blocked = blocked_cells or set()
    s = (int(start[0]), int(start[1]))
    g = (int(goal[0]), int(goal[1]))
    if s == g:
        return [s]
    
    cols = int(config.GRID_COLS)
    rows = int(config.GRID_ROWS)

    q = deque([s])
    came = {s: None}
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        cur = q.popleft()
        for dc, dr in neigh:
            nc = cur[0] + dc
            nr = cur[1] + dr
            if nc < 0 or nr < 0 or nc >= cols or nr >= rows:
                continue
            nxt = (nc, nr)
            if nxt in came:
                continue
            if nxt in blocked and nxt != g:
                continue
            came[nxt] = cur
            if nxt == g:
                path = [g]
                curp = cur
                while curp is not None:
                    path.append(curp)
                    curp = came[curp]
                path.reverse()
                return path
            q.append(nxt)

    return None



def cells_to_world(path_cells: Sequence[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """Zell Zentren in Weltkoordinaten."""
    return [cell_center_from_topleft(c, r, 1, 1) for (c, r) in path_cells]


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


def _build_sources_sinks(n: int, edges: Sequence[Edge]) -> Tuple[List[int], List[int]]:
    """
    Sources/Sinks nur aus Maschinen->Maschinen Kanten ableiten.
    Entry/Exit (None-Endpunkte) werden ignoriert.
    """
    incoming = [0] * n
    outgoing = [0] * n

    for a, b in edges:
        if a is None or b is None:
            continue
        if 0 <= int(a) < n and 0 <= int(b) < n:
            outgoing[int(a)] += 1
            incoming[int(b)] += 1

    sources = [i for i in range(n) if incoming[i] == 0]
    sinks = [i for i in range(n) if outgoing[i] == 0]
    return sources, sinks


Edge = Tuple[Optional[int], Optional[int]]

def route_all_flows(ind: List[Dict]) -> Tuple[List[Dict], float]:
    """Routed alle Materialflüsse mit Breite FLOW_WIDTH_M."""
    n = len(ind)

    raw_edges = list(getattr(config, "MATERIAL_CONNECTIONS", []))

    # Robust: akzeptiert sowohl (int,int) als auch (None,int)/(int,None)
    edges: List[Edge] = []
    for a, b in raw_edges:
        a2 = None if a is None else int(a)
        b2 = None if b is None else int(b)
        edges.append((a2, b2))

    has_explicit_entry_exit = any(a is None or b is None for a, b in edges)

    routed_edges: List[Tuple[Optional[int], Optional[int], str]] = []

    if has_explicit_entry_exit:
        # Kanten genau so routen, wie sie gespeichert wurden (keine Auto-Erweiterung!)
        for a, b in edges:
            if a is None and b is None:
                # unsinnig, aber defensiv: ignorieren
                continue
            if a is None:
                routed_edges.append((None, b, "entry"))
            elif b is None:
                routed_edges.append((a, None, "exit"))
            else:
                routed_edges.append((a, b, "mm"))
    else:
        # Backward-compatible: alte Speicherung (nur mm) -> Entry/Exit ableiten
        sources, sinks = _build_sources_sinks(n, edges)
        for s in sources:
            routed_edges.append((None, s, "entry"))
        for a, b in edges:
            routed_edges.append((a, b, "mm"))
        for t in sinks:
            routed_edges.append((t, None, "exit"))

    blocked_base: set[Tuple[int, int]] = set(config.OBSTACLES)
    for m in ind:
        blocked_base |= occupied_cells(m)

    flow_used: set[Tuple[int, int]] = set()
    flows: List[Dict] = []
    penalty = 0.0

    entry_cell = _clamp_cell(
        int(round(float(config.ENTRY_CELL[0]))),
        int(round(float(config.ENTRY_CELL[1]))),
    )
    exit_cell = _clamp_cell(
        int(round(float(config.EXIT_CELL[0]))),
        int(round(float(config.EXIT_CELL[1]))),
    )

    for a_idx, b_idx, kind in routed_edges:
        if a_idx is None:
            start = entry_cell
        else:
            start = get_terminal_cell(ind[int(a_idx)], kind="out")

        if b_idx is None:
            goal = exit_cell
        else:
            goal = get_terminal_cell(ind[int(b_idx)], kind="in")

        local_blocked = set(blocked_base) | set(flow_used)
        local_blocked.discard(start)
        local_blocked.discard(goal)

        #Pfad Suchen abhängig von überlappung oder nicht
        if config.OVERLAPPED:
            path = find_manhattan_path(start, goal, blocked_cells=local_blocked)
        else:
            path = find_manhattan_path_nooverlap(start, goal)

        if path is None:
            penalty += float(config.OUT_OF_BOUNDS_PENALTY) * 5.0
            flows.append({"kind": kind, "path": None, "world": None, "cells": set()})
            continue

        world = cells_to_world(path)
        footprint = flow_footprint_cells(world, float(config.FLOW_WIDTH_M))

        if footprint & (blocked_base - {start, goal}):
            penalty += float(config.FLOW_PENALTY)
        if footprint & flow_used:
            penalty += float(config.FLOW_PENALTY)

        flow_used |= footprint
        flows.append({"kind": kind, "path": path, "world": world, "cells": footprint})

    return flows, penalty

