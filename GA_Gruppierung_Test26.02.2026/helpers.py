# HELPERS MODULE

from __future__ import annotations

import copy
import math
import random
import heapq
import hashlib
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Set

import config

def Create_Groups() -> List[Optional[List[Optional[int]]]]:
    """Hier werden die Maschinen abhängig vom Arbeiter miteinander gruppiert
    Ist anpassbar ==> bessere Gruppierungslogik
    """
    Gruppen = [None] * config.MACHINE_COUNT
    workers = getattr(config, "MACHINE_WORKERS", [])
    for Index, Maschine in enumerate (workers): 
        if Maschine is None:
            continue
        WorkerIdxFromZero = int(Maschine["worker"]) - 1
        if Gruppen[WorkerIdxFromZero] is None:
            Gruppen[WorkerIdxFromZero] = [Index, None]
        else:
            Gruppen[WorkerIdxFromZero][1] = Index
    Gruppen[:] = [x for x in Gruppen if x is not None]
    return Gruppen

def GetGroupCost(MachineFixed: Dict, MachineTest: Dict) -> int:
    """Gruppenkosten berechnen ==> Wenn verbunden: Material + Workerkosten sonst nur Arbeiterkosten"""
    GroupDicts: List[Dict] = []
    ThisGroupCost = 0.0
    no_path_penalty = float(getattr(config, "NO_PATH_PENALTY", 1e6))
    FixedGrid = get_worker_clearance(MachineFixed)
    TestGrid = get_worker_clearance(MachineTest)
    GroupDicts.append(MachineFixed)
    GroupDicts.append(MachineTest)
    blockedForGroups = _blocked_cells_for_routing(GroupDicts)
    WorkerResult = AStar_Worker_Path(FixedGrid, TestGrid, blockedForGroups)
    #Arbeiter kosten und wenn in einer Linie materialfluss mit einberechnen
    if WorkerResult is not None:
        _, ThisGroupCost, _= WorkerResult
    else: 
        ThisGroupCost = 1e12


    if any(MachineFixed["idx"] == Out and MachineTest["idx"] == In for (Out, In, Weight) in config.MATERIAL_CONNECTIONS):
        # print("Wird beachtet")
        edges = list(getattr(config, "MATERIAL_CONNECTIONS", []))
        EdgeWeights = {(Out, In): Weight for (Out, In, Weight) in edges}
        Out, In = (MachineFixed["idx"], MachineTest["idx"])
        Weight = EdgeWeights.get((Out, In))
        MaterialPoint1 = machine_output_point(MachineFixed)
        MaterialPoint2 = machine_input_point(MachineTest)
        blocked_sig = _blocked_signature(blockedForGroups)
        _, MaterialLength = route_world_to_world(MaterialPoint1, MaterialPoint2, blocked = blockedForGroups, blocked_sig = blocked_sig)
        if not math.isfinite(MaterialLength):
            ThisGroupCost += no_path_penalty
            # print(f"falsch: {ThisGroupCost}") 
        else: 
            ThisGroupCost += MaterialLength * Weight * config.MATERIAL_WEIGHT
            # print(f"richtig: {ThisGroupCost}") 
    return ThisGroupCost

def Optimize_Groups(BestIndCopy: List[Dict]) -> List[List[Dict]]:
    # 4. bei der Gruppe, die Position relativ zueinander optimieren
    # 5. Gruppe fest machen und im Ga als eine maschine Behandeln 
    # print("OBSTACLES:", len(config.OBSTACLES))
    def MoveTestMachine(TestMachine: Dict, FixedMachine: Dict, dx: int, dy: int, dz: int) -> Dict:
        TestMachine["gx"] = int(TestMachine.get("gx", 0)) + int(dx)
        TestMachine["gy"] = int(TestMachine.get("gy", 0)) + int(dy)
        TestMachine["z"] = (int(TestMachine.get("z", 0)) % 360 + int(dz)) % 360
        w_eff, h_eff = effective_dims(TestMachine, TestMachine["z"])

        max_col = max(0, int(config.GRID_COLS) - int(w_eff))
        max_row = max(0, int(config.GRID_ROWS) - int(h_eff))
        if max_col < int(TestMachine["gx"]) or max_row < int(TestMachine["gy"]):
            return None
        if int(TestMachine["gx"]) < 0 or int(TestMachine["gy"]) < 0:
            return None
        if occupied_cells(TestMachine, True) & occupied_cells(FixedMachine, True):
            return None
        TestMachine["x"], TestMachine["y"] = cell_center_from_topleft(int(TestMachine["gx"]), int(TestMachine["gy"]), int(w_eff), int(h_eff))
        return TestMachine

    MachineGroups = Create_Groups()
    ConnectedGroups: List[Dict] =[]
    for (MachineFixed, MachineTest) in MachineGroups:
        FixedMachine = BestIndCopy[MachineFixed]
        GroupCost = GetGroupCost(FixedMachine, BestIndCopy[MachineTest])
        # print(f"Group Cost vorher: {GroupCost}")
        TestMachineCopy = copy.deepcopy(BestIndCopy[MachineTest])
        previous_cost = 1e12
        # print("before", FixedMachine["gx"], BestIndCopy[MachineTest]["gx"], FixedMachine["gy"], BestIndCopy[MachineTest]["gy"])

        for i in range(500): #Maximallänge
            if previous_cost <= GroupCost:
                # print(f"Group Cost Nachher: {GroupCost}")
                # print("======================================================")
                break
            previous_cost = GroupCost
            TestXPlus = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = 1, dy = 0, dz = 0)
            if TestXPlus is not None:
                new_cost = GetGroupCost(FixedMachine, TestXPlus)
                if new_cost < GroupCost:
                    TestMachineCopy = TestXPlus
                    # print(f"TestXPlus")
                    GroupCost = new_cost
                    continue
            TestXMinus = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = -1, dy = 0, dz = 0)
            if TestXMinus is not None:
                new_cost = GetGroupCost(FixedMachine, TestXMinus)
                if new_cost < GroupCost:
                    TestMachineCopy = TestXMinus
                    # print(f"TestXMinus")
                    GroupCost = new_cost
                    continue
            TestYPlus = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = 0, dy = 1, dz = 0)
            if TestYPlus is not None:
                new_cost = GetGroupCost(FixedMachine, TestYPlus)
                if new_cost < GroupCost:
                    TestMachineCopy = TestYPlus
                    # print(f"TestYPlus")
                    GroupCost = new_cost
                    continue
            TestYMinus = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = 0, dy = -1, dz = 0)
            if TestYMinus is not None:
                new_cost = GetGroupCost(FixedMachine, TestYMinus)
                if new_cost < GroupCost:
                    TestMachineCopy = TestYMinus
                    # print(f"TestYMinus")
                    GroupCost = new_cost
                    continue
            TestZ90 = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = 0, dy = 0, dz = 90)
            if TestZ90 is not None:
                new_cost = GetGroupCost(FixedMachine, TestZ90)
                if new_cost < GroupCost:
                    TestMachineCopy = TestZ90
                    # print(f"TestZ90")
                    GroupCost = new_cost
                    continue
            TestZ180 = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = 0, dy = 0, dz = 180)
            if TestZ180 is not None:
                new_cost = GetGroupCost(FixedMachine, TestZ180)
                if new_cost < GroupCost:
                    TestMachineCopy = TestZ180
                    # print(f"TestZ180")
                    GroupCost = new_cost
                    continue
            TestZ270 = MoveTestMachine(copy.deepcopy(TestMachineCopy), FixedMachine, dx = 0, dy = 0, dz = 270)
            if TestZ270 is not None:
                new_cost = GetGroupCost(FixedMachine, TestZ270) 
                if new_cost < GroupCost:
                    TestMachineCopy = TestZ270
                    # print(f"TestZ270")
                    GroupCost = new_cost
                    continue
    # ===================================== Feste Maschine nur rotieren ==================================
            TestFixedZ90 = MoveTestMachine(copy.deepcopy(FixedMachine), TestMachineCopy, dx = 0, dy = 0, dz = 90)
            if TestFixedZ90 is not None:
                new_cost = GetGroupCost(TestFixedZ90, TestMachineCopy)
                if new_cost < GroupCost:
                    FixedMachine = TestFixedZ90
                    # print(f"TestFixedZ90")
                    GroupCost = new_cost
                    continue
            TestFixedZ180 = MoveTestMachine(copy.deepcopy(FixedMachine), TestMachineCopy, dx = 0, dy = 0, dz = 180)
            if TestFixedZ180 is not None:
                new_cost = GetGroupCost(TestFixedZ180, TestMachineCopy)
                if new_cost < GroupCost:
                    FixedMachine = TestFixedZ180
                    # print(f"TestFixedZ180")
                    GroupCost = new_cost
                    continue
            TestFixedZ270 = MoveTestMachine(copy.deepcopy(FixedMachine), TestMachineCopy, dx = 0, dy = 0, dz = 270)
            if TestFixedZ270 is not None:
                new_cost = GetGroupCost(TestFixedZ270, TestMachineCopy) 
                if new_cost < GroupCost:
                    FixedMachine = TestFixedZ270
                    # print(f"TestFixedZ270")
                    GroupCost = new_cost
                    continue
        # ===================================== Feste Maschine nur rotieren ==================================

        # print("copy ", TestMachineCopy["gx"], TestMachineCopy["gy"], TestMachineCopy["z"])
        # print("ind  ", BestIndCopy[MachineTest]["gx"], BestIndCopy[MachineTest]["gy"], BestIndCopy[MachineTest]["z"])
        BestIndCopy[MachineTest] = TestMachineCopy  # commit
        # ... hillclimb ...
        # print("ind' ", BestIndCopy[MachineTest]["gx"], BestIndCopy[MachineTest]["gy"], BestIndCopy[MachineTest]["z"])

        LeaderMemberX = int(round(2 * (TestMachineCopy["x"] - FixedMachine["x"]) / config.GRID_SIZE))
        LeaderMemberY = int(round(2 * (TestMachineCopy["y"] - FixedMachine["y"]) / config.GRID_SIZE))
        LeaderMemberZ = (TestMachineCopy["z"] - FixedMachine["z"]) % 360
        ConnectedGroups.append({
            "leader": MachineFixed,
            "members": [MachineFixed, MachineTest],
            "local": {
                MachineTest: {"MemberX": LeaderMemberX, "MemberY": LeaderMemberY, "MemberZ": LeaderMemberZ}
            }
        })

    return ConnectedGroups

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

def get_worker_clearance(m: Dict):
    """Erstellt das 1*1 meter feld für den Worker"""
    worker_point = machine_worker_point(m)  #Hier haben wir den Punkt
    rotation = m.get("z", 0)
    clearance: set[Tuple[int, int]] = set()
    if worker_point is None:
        return clearance
    col = int(worker_point[0] / config.GRID_SIZE)
    row = int(worker_point[1] / config.GRID_SIZE)
    side_raw = ((getattr(config, "MACHINE_WORKERS", []) or [None])[int(m.get("idx", -1))] or {}).get("side_worker") if 0 <= int(m.get("idx", -1)) < len(getattr(config, "MACHINE_WORKERS", []) or []) else None
    side = str(side_raw).strip().lower()
    actual_side = _rotated_side(side, rotation)
    
    OneMeterInGrid = int(round(1 / config.GRID_SIZE))
    half = OneMeterInGrid // 2
    if actual_side == "left":
        x_top_left = col - OneMeterInGrid
        y_top_left = row - half
    elif actual_side == "top":
        x_top_left = col - half
        y_top_left = row - OneMeterInGrid
    elif actual_side == "right":
        x_top_left = col
        y_top_left = row - half
    elif actual_side == "bottom":
        x_top_left = col - half
        y_top_left = row
    else:
        print("funktioniert nicht du lappen (Zeile 524 in helpers)")
    for dwx in range(OneMeterInGrid):
        for dwy in range(OneMeterInGrid):
            col = x_top_left + dwx
            row = y_top_left + dwy
            if 0 <= col < config.GRID_COLS and 0 <= row < config.GRID_ROWS:
                clearance.add((col, row))
    return clearance

def occupied_cells(m: Dict, Clearance) -> set[Tuple[int, int]]:
    """
    Raster-Fußabdruck der Maschine & bei bedarf:
    1*1 Meter Viereck hinter dem Worker
    """
    w_eff, h_eff = effective_dims(m)  #Bei Rotation w & h tauschen
    idx = int(m.get("idx", -1))
    out: set[Tuple[int, int]] = set()
    if Clearance:
        if 0 <= idx < len(config.MACHINE_WORKERS) and config.MACHINE_WORKERS[idx] is not None:
            out |= get_worker_clearance(m)

    gx = int(m["gx"])
    gy = int(m["gy"])
    w = int(w_eff)
    h = int(h_eff)
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
    label = config.MACHINE_LABELS[idx]
    w_m, d_m = config.MACHINE_SIZES[idx]
    w_cells = max(1, int(round(float(w_m) / config.GRID_SIZE)))
    h_cells = max(1, int(round(float(d_m) / config.GRID_SIZE)))

    fixed_list = getattr(config, "MACHINE_FIXED", [])
    fixed = fixed_list[idx] if idx < len(fixed_list) else None

    if fixed is not None:
        #z: entweder fix oder fallback (aber wenn "fixed" dann lieber deterministisch)
        z_fixed = fixed.get("z", None)
        z = int(z_fixed) if z_fixed is not None else 0  # oder random.choice(config.ROTATIONS), wenn du willst

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
        col = gx
        row = gy

        # belege occupied_set (gepufferte Fläche!)
        for c in range(col, col + w_eff):
            for r in range(row, row + h_eff):
                occupied_set.add((c, r))

        return {
            "idx": int(idx),
            "label": str(label),
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

    w_eff, h_eff = effective_dims((w_cells, h_cells), z)

    # Für die Randbegrenzung müssen wir die gepufferte Fläche berücksichtigen,
    # sonst würde die echte Maschine zwar im Grid liegen, aber der "Laufweg" ragt raus.

    max_col = max(0, config.GRID_COLS - int(w_eff))
    max_row = max(0, config.GRID_ROWS - int(h_eff))

    for _ in range(max_attempts):
        col = random.randint(0, max_col)
        row = random.randint(0, max_row)

        # Testet die gepufferte Fläche: top-left um p nach außen verschieben
        # Wichtig: gx/gy bleiben die echten Maschinen-zellen (ohne Puffer),
        # damit Center/Ports/Zeichnung korrekt bleiben.
        if can_place_at(col, row, w_eff, h_eff, occupied_set):
            gx = int(col)
            gy = int(row)
            x, y = cell_center_from_topleft(gx, gy, w_eff, h_eff)
            return {
                "idx": int(idx),
                "label": str(label),
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

    x, y = cell_center_from_topleft(int(col), int(row), w_eff, h_eff)
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
        occupied |= occupied_cells(m, Clearance = True)
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

def _rotated_side(side: str, rotation_deg: int) -> str:
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
#=========================== Ende Port berechnung ============================================
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
    """If a cell is blocked, pick a nearby free cell (BFS within radius)"""
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

    #Start/Ziel dürfen in "blocked" liegen (Ports liegen häufig in der Maschine)
    blocked_local = set(blocked)
    blocked_local.discard(start)
    blocked_local.discard(goal)

    def heuristik_Astar(c: Tuple[int, int]) -> int:
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristik_Astar(start), 0, start))

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
                heapq.heappush(open_heap, (ng + heuristik_Astar(nb), ng, nb))

    return None

Cell = Tuple[int, int]
GridBlock = Set[Cell]
GridBlockKey = frozenset[Cell]

def WorkerPathNeighbors(CurrentHeapGrid: GridBlock) -> List[GridBlock]:
    NextHeapGrid: set[Tuple[int,int]]
    WorkerNeighbors: list[set[Tuple[int,int]]] =[]

    AllXCords = [x for x, _ in CurrentHeapGrid]
    AllYCords = [y for _, y in CurrentHeapGrid]
    MaxX, MinX = max(AllXCords), min(AllXCords)
    MaxY, MinY = max(AllYCords), min(AllYCords)

    if MinX > 0:
        NextHeapGrid =  {(col - 1, row) for (col, row) in CurrentHeapGrid} #Nach links um 1 verschieben
        WorkerNeighbors.append(NextHeapGrid)
    if MaxX < int(config.GRID_COLS) - 1:
        NextHeapGrid =  {(col + 1, row) for (col, row) in CurrentHeapGrid} #Nach rechts verschieben
        WorkerNeighbors.append(NextHeapGrid)
    if MinY > 0:
        NextHeapGrid =  {(col, row - 1) for (col, row) in CurrentHeapGrid} #Nach unten verschieben
        WorkerNeighbors.append(NextHeapGrid)
    if MaxY < int(config.GRID_ROWS) - 1:
        NextHeapGrid =  {(col, row + 1) for (col, row) in CurrentHeapGrid} #Nach oben verschieben
        WorkerNeighbors.append(NextHeapGrid)
    return WorkerNeighbors

def AStar_Grid_Center(Grid: GridBlock) -> Tuple[float, float]:
    if not Grid:
        return None
    AllXCords = [x for x, _ in Grid]
    AllYCords = [y for _, y in Grid]
    MaxX, MinX = max(AllXCords), min(AllXCords)
    MaxY, MinY = max(AllYCords), min(AllYCords)
    CenterX = (MaxX + MinX +1) / 2
    CenterY = (MaxY + MinY +1) / 2
    Grid_Center: Tuple[float, float]
    Grid_Center = (CenterX, CenterY)
    return Grid_Center

def AStar_Worker_Path(
    StartGrid: GridBlock, 
    EndGrid: GridBlock, 
    blocked: GridBlock
    ) -> Optional[Tuple[List[GridBlock], float]]:

    if not StartGrid or not EndGrid:
        return None

    StartGridCenter = AStar_Grid_Center(StartGrid)
    EndGridCenter = AStar_Grid_Center(EndGrid)
    blocked_local = set(blocked)

    if EndGridCenter is None:
        return None
    if StartGridCenter == EndGridCenter and StartGridCenter is not None:
        TracedPath = [(StartGridCenter[0] * float(config.GRID_SIZE), StartGridCenter[1] * float(config.GRID_SIZE))]
        return set(StartGrid), 0.0, TracedPath

    def heuristik_Astar_Worker(CurrentGrid: GridBlock) -> int: #Astar heuristik Worker (Abstand aktueller Grids zum WorkerGrid) 
        CurrentGridCenter = AStar_Grid_Center(CurrentGrid)
        if CurrentGridCenter is None:
            return 1e12
        return config.GRID_SIZE * (abs(CurrentGridCenter[0] - EndGridCenter[0]) + abs(CurrentGridCenter[1] - EndGridCenter[1]))   #anpassen weil Abstandberechnung
    
    StartKey: GridBlockKey = frozenset(StartGrid)
    open_worker_heap: List[Tuple[float, float, int, GridBlockKey]] = []          #Liste (Heuristische Kosten g + h , g, GridSet) (f = h + g)
    HeapCounter = 0
    heapq.heappush(open_worker_heap, (heuristik_Astar_Worker(StartGrid), 0.0, HeapCounter, StartKey))

    Score_G: dict[GridBlockKey, float] = {StartKey: 0.0}
    CameFromGrid: dict[GridBlockKey, GridBlockKey] = {}
    EndGridSet = set(EndGrid)
    BasisWorkerQuadrat = (1 / config.GRID_SIZE)
    RequiredOverlap =  BasisWorkerQuadrat * BasisWorkerQuadrat

    while open_worker_heap:
        _, CurrentCost, _, CurrentHeapKey = heapq.heappop(open_worker_heap)
        CurrentHeapGrid: GridBlock = set(CurrentHeapKey)
        if len(CurrentHeapGrid & EndGridSet) == RequiredOverlap:
            path: List[GridBlockKey] = [CurrentHeapKey]
            while CurrentHeapKey in CameFromGrid:
                CurrentHeapKey = CameFromGrid[CurrentHeapKey]
                path.append(CurrentHeapKey)
            path.reverse()

            UniquePath: GridBlock = set()
            TracedPath: List[Tuple[float, float]] = []
            for k in path:
                UniquePath.update(k)  #fügt nur neue Zellen zum Pfad hinzu
                GridCenter = AStar_Grid_Center(set(k)) #erstellt aus allen grids die hinzugefügt wurden eine Liste aus den Centern (Zum zeichnen)
                if GridCenter is None:
                    continue
                TracedPath.append((float(GridCenter[0]) * config.GRID_SIZE,float(GridCenter[1]) * config.GRID_SIZE))
            return UniquePath, float(CurrentCost), TracedPath

        if CurrentCost != Score_G.get(CurrentHeapKey, 10**12):
            continue

        for Neighbor in WorkerPathNeighbors(CurrentHeapGrid):
            if Neighbor & blocked_local:
                continue
            NeighborKey: GridBlockKey = frozenset(Neighbor)
            NewScore_G = CurrentCost + config.GRID_SIZE
            if NewScore_G < Score_G.get(NeighborKey, 10**12):
                Score_G[NeighborKey] = NewScore_G
                CameFromGrid[NeighborKey] = CurrentHeapKey
                HeapCounter += 1
                heapq.heappush(open_worker_heap, (NewScore_G + heuristik_Astar_Worker(Neighbor), NewScore_G, HeapCounter, NeighborKey))

    return None

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

def CompressCollinearPoints(Points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(Points) <= 2:
        return Points
    out = [Points[0]]
    for i in range(1, len(Points) - 1):
        PrevPointX, PrevPointY = out[-1]
        CurrentX, CurrentY = Points[i]
        NextPointX, NextPoint = Points[i + 1]
        if (abs(PrevPointX - CurrentX) < 1e-9 and abs(CurrentX - NextPointX) < 1e-9) or (abs(PrevPointY - CurrentY) < 1e-9 and abs(CurrentY - NextPoint) < 1e-9):
            continue
        out.append((CurrentX, CurrentY))
    out.append(Points[-1])
    return out

def _blocked_cells_for_routing(ind: List[Dict]) -> set[Tuple[int, int]]:
    """Routing-Blockaden: OBSTACLES + Maschinen-Fußabdruck (ohne Clearance)."""
    blocked = set(getattr(config, "OBSTACLES", []) or [])
    #print(f"Das sind die in config geblockten Zellen: {blocked}")

    #Entry/Exit sind immer begehbar, selbst wenn sie in OBSTACLES liegen
    try:
        blocked.discard(tuple(config.ENTRY_CELL))
        blocked.discard(tuple(config.EXIT_CELL))
    except Exception:
        print("helpers: 539")
        pass

    for m in ind:
        blocked |= occupied_cells(m, Clearance = False)  #<-- Clearance bleibt begehbar
    return blocked

def route_port_to_point(
    start_world: Tuple[float, float],
    machine_world : Tuple[float, float]
) -> Tuple[Optional[List[Tuple[float, float]]], float]:
    pts: List[Tuple[float, float]] = [start_world, machine_world] 
    length = math.dist(start_world, machine_world)
    return (pts, length)

def route_line_to_world(
    start_world: Tuple[float, float],
    end_world: Tuple[float, float],
    machine_world : Tuple[float, float]
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

    s_cell = _pick_free_cell_near(s_cell, blocked, max_radius=3)
    e_cell = _pick_free_cell_near(e_cell, blocked, max_radius=3)
    cells = _astar_path_cached(blocked_sig, s_cell, e_cell, blocked)
    
    #print(f"[ROUTE] start_world={start_world} end_world={end_world} " f"start_cell={s_cell} goal_cell={e_cell} "f"path_len_cells={(len(cells) if cells else None)} "f"path_head={(cells[:8] if cells else None)}")
    if not cells:
        return None, float("inf")

    centers = [_world_of_cell_center(c, r) for (c, r) in cells]

    Points: List[Tuple[float, float]] = [start_world]
    Points.extend(centers)
    Points.append(end_world)

    pts = CompressCollinearPoints(Points)

    length_m = (len(cells) - 1) * float(config.GRID_SIZE)
    return pts, float(length_m)

def route_worker(
    StartGrid: set[Tuple[int, int]] = set(),
    EndGrid: set[Tuple[int, int]] = set(),
    *,
    blocked: set[Tuple[int, int]],
    blocked_sig: int,
    ) -> Tuple[Optional[List[Tuple[float, float]]], float]:

    WorkerResult = AStar_Worker_Path(StartGrid, EndGrid, blocked)
    if WorkerResult is None:
        return None, float("inf"), None
    UniquePath, WorkerCost, Points = WorkerResult

    if not Points:
        return (None, WorkerCost, None) if math.isfinite(WorkerCost) else (None, float("inf"), None)
    TracedPath = CompressCollinearPoints(Points)
    return TracedPath, float(WorkerCost), UniquePath

def compute_routed_edges(ind: List[Dict]) -> Dict[str, List[Dict]]:
    """Berechne geroutete Polylinien für MATERIAL_CONNECTIONS, WORKER_CONNECTIONS und Anschlüssen
    Ergebnisst:
    {"material": [{"a":..,"b":..,"pts":[(x,y),...],"length_m":..}, ...],
    "worker": [{"a":..,"b":..,"pts":[(x,y),...],"length_m":..}, ...]} etc.
    """
    n = len(ind)
    blocked = _blocked_cells_for_routing(ind)
    blocked_sig = _blocked_signature(blocked)

    entry_world = cell_center_from_topleft(int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]), 1, 1)
    exit_world = cell_center_from_topleft(int(config.EXIT_CELL[0]), int(config.EXIT_CELL[1]), 1, 1)

    out: Dict[str, List[Dict]] = {"material": [], "worker": [], "water": [], "gas": [], "other": []}

    edges = list(getattr(config, "MATERIAL_CONNECTIONS", []))
    for e in edges:
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

    w_edges = list(getattr(config, "WORKER_CONNECTIONS", []))
    Previous_a_idx = None #placeholder
    Previous_b_idx = None
    pts = None
    length_m = float("inf") #Placeholder
    for a, b in w_edges:
        a_idx: Optional[int] = None if a is None else int(a)
        b_idx: Optional[int] = None if b is None else int(b)

        if a_idx is None or b_idx is None:
                continue
        if not (0 <= a_idx < n and 0 <= b_idx < n):
                continue
        
        if  Previous_a_idx == b_idx and  Previous_b_idx == a_idx:
            ptsLoop = list(reversed(pts)) if pts else None
            out["worker"].append({"a": a_idx, "b": b_idx, "pts": ptsLoop, "length_m": float(length_m), "WorkerPathCells": None})
            Previous_a_idx, Previous_b_idx=  a_idx, b_idx
            continue

        StartGrid = get_worker_clearance(ind[a_idx])
        EndGrid = get_worker_clearance(ind[b_idx])

        pts, length_m, WorkerPathCells = route_worker(StartGrid, EndGrid, blocked=blocked, blocked_sig=blocked_sig)
        out["worker"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m), "WorkerPathCells": WorkerPathCells})
        Previous_a_idx = a_idx
        Previous_b_idx = b_idx

    #=========================================================================================================
    #================ Hier werden die optionalen Wasser, Gas und Sonstiges Verbindungen definiert ============
    #=========================================================================================================

    if config.WATER_CELL[0] is not None:
        water_edges = list(getattr(config, "WATER_CONNECTIONS", []))
        #print(f"Water edges: {water_edges}")
        for _, b in water_edges:
            b_idx: Optional[int] = None if b is None else int(b)
            if b_idx is None:
                continue
            else:
                if not (0 <= b_idx < n):
                    print(f"Water connection with invalid b_idx={b_idx}, skipping")
                    continue
                p2 = machine_water_point(ind[b_idx])
                w1 = config.WATER_CELL[0]
            if config.WATER_CELL[1] == None:
                pts, length_m = route_port_to_point(w1, p2)
            else:
                w2 = config.WATER_CELL[1]
                pts, length_m = route_line_to_world(w1, w2, p2)
            out["water"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    if config.GAS_CELL[0] is not None:
        gas_edges = list(getattr(config, "GAS_CONNECTIONS", []))
        for _, b in gas_edges:
            b_idx: Optional[int] = None if b is None else int(b)
            if b_idx is None:
                continue
            else:
                if not (0 <= b_idx < n):
                    continue
                p2 = machine_gas_point(ind[b_idx])
                g1 = config.GAS_CELL[0]
            if config.GAS_CELL[1] == None:
                pts, length_m = route_port_to_point(g1, p2)
            else:
                g2 = config.GAS_CELL[1]
                pts, length_m = route_line_to_world(g1, g2, p2) 
            out["gas"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    if config.OTHER_CELL[0] is not None:
        other_edges = list(getattr(config, "OTHER_CONNECTIONS", []))
        for _, b in other_edges:
            b_idx: Optional[int] = None if b is None else int(b)
            if b_idx is None:
                continue
            else:
                if not (0 <= b_idx < n):
                    continue
                p2 = machine_other_point(ind[b_idx])
                o1 = config.OTHER_CELL[0]
            if config.OTHER_CELL[1] == None:
                pts, length_m = route_port_to_point(o1, p2)
            else:
                o2 = config.OTHER_CELL[1]
                pts, length_m = route_line_to_world(o1, o2, p2)
            out["other"].append({"a": a_idx, "b": b_idx, "pts": pts, "length_m": float(length_m)})

    return out

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
        w = float(m.get("weight", 1.0) or 1.0)
        if m["a"] is None:
            length = float(m["length_m"])
            cost += no_path_penalty if not math.isfinite(length) else length * w * config.MATERIAL_WEIGHT
        elif m["b"] is None:
            length = float(m["length_m"])
            cost += no_path_penalty if not math.isfinite(length) else length * w * config.MATERIAL_WEIGHT
        else:
            length = float(m["length_m"])
            cost += no_path_penalty if not math.isfinite(length) else length * w * config.MATERIAL_WEIGHT

    for e in routed["worker"]:
        length = float(e["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length * config.WORKER_WEIGHT
    for w in routed["water"]:
        length = float(w["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length * config.WATER_WEIGHT
    for g in routed["gas"]:
        length = float(g["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length * config.GAS_WEIGHT
    for o in routed["other"]:
        length = float(o["length_m"])
        cost += no_path_penalty if not math.isfinite(length) else length * config.OTHER_WEIGHT

    return float(cost)