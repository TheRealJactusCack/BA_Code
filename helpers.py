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
    """Platziert Maschine idx ohne Überlappung"""
    w_m, d_m = config.MACHINE_SIZES[idx]
    w_cells = max(1, int(round(float(w_m) / config.GRID_SIZE)))
    h_cells = max(1, int(round(float(d_m) / config.GRID_SIZE)))

    fixed_list = getattr(config, "MACHINE_FIXED", [])
    fixed = fixed_list[idx] if idx < len(fixed_list) else None

    if fixed is not None:
        #z: entweder fix oder fallback (aber wenn "fixed" dann lieber deterministisch)
        z_fixed = fixed.get("z", None)
        z = int(z_fixed) if z_fixed is not None else 0  # oder random.choice(config.ROTATIONS), wenn du willst

        p = clearance_pad_cells()
        w_eff, h_eff = effective_dims((w_cells, h_cells), z)

        #Fixed world center
        x = float(fixed["x"])
        y = float(fixed["y"])

        #top-left (gx/gy) berechnen, damit occupied_set korrekt ist
        grid = float(config.GRID_SIZE)

        #Umrechnung Center->TopLeft in Zellen:
        #top-left (in Zellen) = round((center - 0.5*eff_dim*grid)/grid)
        gx = int(round((x - 0.5 * float(w_eff) * grid) / grid))
        gy = int(round((y - 0.5 * float(h_eff) * grid) / grid))

        #Rand + Puffer prüfen: gepufferte Fläche muss im Grid liegen & frei sein
        w_eff_p = int(w_eff) + 2 * p
        h_eff_p = int(h_eff) + 2 * p
        col = gx - p
        row = gy - p

        """if col < 0 or row < 0 or (col + w_eff_p) > config.GRID_COLS or (row + h_eff_p) > config.GRID_ROWS:
            raise ValueError(f"Fixed machine idx={idx} liegt (inkl. Puffer) außerhalb des Grids")

        if not can_place_at(col, row, w_eff_p, h_eff_p, occupied_set):
            raise ValueError(f"Fixed machine idx={idx} überlappt (inkl. Puffer) mit anderer Maschine")"""

        # belege occupied_set (gepufferte Fläche!)
        for c in range(col, col + w_eff_p):
            for r in range(row, row + h_eff_p):
                occupied_set.add((c, r))

        return {
            "idx": int(idx),
            "x": x,
            "y": y,
            "z": int(z),
            "gx": int(gx),
            "gy": int(gy),
            "w_cells": int(w_cells),
            "h_cells": int(h_cells),
        }


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
    """Start Individuum mit MACHINE_COUNT Maschinen"""
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


def compute_routed_edges(ind: List[Dict]) -> Dict[str, List[Dict]]:
    """Berechne geroutete Polylinien für MATERIAL_CONNECTIONS und WORKER_CONNECTIONS.
    Ergebnisst:
    {"material": [{"a":..,"b":..,"pts":[(x,y),...],"length_m":..}, ...],
    "worker": [{"a":..,"b":..,"pts":[(x,y),...],"length_m":..}, ...]}
    """
    n = len(ind)
    blocked = _blocked_cells_for_routing(ind)
    blocked_sig = _blocked_signature(blocked)

    entry_world = cell_center_from_topleft(int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]), 1, 1)
    exit_world = cell_center_from_topleft(int(config.EXIT_CELL[0]), int(config.EXIT_CELL[1]), 1, 1)

    out: Dict[str, List[Dict]] = {"material": [], "worker": [], "water": [], "gas": [], "other": []}

    edges = list(getattr(config, "MATERIAL_CONNECTIONS", []))
    for a, b in edges:
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
        out["material"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    w_edges = list(getattr(config, "WORKER_CONNECTIONS", []))
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

        pts, length_m = route_world_to_world(p1, p2, blocked=blocked, blocked_sig=blocked_sig)
        out["worker"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})


    #=========================================================================================================
    #================ Hier werden die optionalen Wasser, Gas und Sonstiges Verbindungen definiert ============
    #=========================================================================================================

    if config.WATER_CELL[0] is not None and config.WATER_CELL[1] is not None:
        water_world = cell_center_from_topleft(int(config.WATER_CELL[0]), int(config.WATER_CELL[1]), 1, 1)
        water_edges = list(getattr(config, "WATER_CONNECTIONS", []))
        #print(f"Water edges: {water_edges}")
        for a, b in water_edges:
            a_idx: Optional[int] = None if a is None else int(a)
            b_idx: Optional[int] = None if b is None else int(b)

            if a_idx is None:
                print(f"Water connection with None a_idx, using water_world as p1")
                continue
            else:
                p1 = water_world

            if b_idx is None:
                continue
            else:
                if not (0 <= b_idx < n):
                    print(f"Water connection with invalid b_idx={b_idx}, skipping")
                    continue
                p2 = machine_water_point(ind[b_idx])

            pts, length_m = route_world_to_world(p1, p2, blocked=blocked, blocked_sig=blocked_sig)
            out["water"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})
            #print(f"Water connection: a={a_idx} b={b_idx} length_m={length_m} pts={pts}")
            #print(f"a: {p1} b: {p2}")

    if config.GAS_CELL[0] is not None and config.GAS_CELL[1] is not None:
        gas_world = cell_center_from_topleft(int(config.GAS_CELL[0]), int(config.GAS_CELL[1]), 1, 1)
        gas_edges = list(getattr(config, "GAS_CONNECTIONS", []))
        for a, b in gas_edges:
            a_idx: Optional[int] = None if a is None else int(a)
            b_idx: Optional[int] = None if b is None else int(b)

            if a_idx is None:
                print(f"Gas connection with None a_idx, using gas_world as p1")
                continue
            else:
                p1 = gas_world

            if b_idx is None:
                continue
            else:
                if not (0 <= b_idx < n):
                    continue
                p2 = machine_gas_point(ind[b_idx])

            pts, length_m = route_world_to_world(p1, p2, blocked=blocked, blocked_sig=blocked_sig)
            out["gas"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    if config.OTHER_CELL[0] is not None and config.OTHER_CELL[1] is not None:
        other_world = cell_center_from_topleft(int(config.OTHER_CELL[0]), int(config.OTHER_CELL[1]), 1, 1)
        other_edges = list(getattr(config, "OTHER_CONNECTIONS", []))
        for a, b in other_edges:
            a_idx: Optional[int] = None if a is None else int(a)
            b_idx: Optional[int] = None if b is None else int(b)

            if a_idx is None:
                continue
            else:
                p1 = other_world

            if b_idx is None:
                continue
            else:
                if not (0 <= b_idx < n):
                    continue
                p2 = machine_other_point(ind[b_idx])

            pts, length_m = route_world_to_world(p1, p2, blocked=blocked, blocked_sig=blocked_sig)
            out["other"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    return out

    #Wege zu allen Anschlüssen!!!!==========================================================================

def distance_cost(ind: List[Dict], config: any) -> float:
    """Summer aller Flusslängen (Material + Arbeiter) im Individuum

    - Kollision mit maschinen und Hindernissen werden mit hoher Strafe belegt
    - Laufweg Zellen bleiben verwendbar für Fluss und laufweg
    - Entry/Exit remain walkable even if they are in OBSTACLES
    - keine kollision zwischen flows 
    - Uses a small LRU cache for A* results (speed-up during GA)?????
    """
    routed = compute_routed_edges(ind)
    no_path_penalty = float(getattr(config, "NO_PATH_PENALTY", 1e6))

    cost = 0.0
    for m in routed["material"]:
        if m["a"] is None:
            length = float(m["length_m"])
            cost += no_path_penalty if not math.isfinite(length) else length * config.FACILITY_ENTRY_WEIGHT
        elif m["b"] is None:
            length = float(m["length_m"])
            cost += no_path_penalty if not math.isfinite(length) else length * config.FACILITY_EXIT_WEIGHT
        else:
            length = float(m["length_m"])
            cost += no_path_penalty if not math.isfinite(length) else length * 3  # Materialflüsse höher gewichtet

    for e in routed["worker"]:
        length = float(e["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length

    for w in routed["water"]:
        length = float(w["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length * 10

    for g in routed["gas"]:
        length = float(g["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length 

    for o in routed["other"]:
        length = float(o["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length 

    return float(cost)


Edge = Tuple[Optional[int], Optional[int]]