from __future__ import annotations

import copy
import math
from typing import Any, Optional

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QSizeF
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF, QTransform
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget, QMessageBox, QCheckBox
)

import ui_main
import config
from helpers import (
    cell_center_from_topleft,
    compute_routed_edges,
    effective_dims,
    machine_input_point,
    machine_output_point,
    machine_worker_point,
    machine_water_point,
    machine_gas_point,
    machine_other_point,
    normalize_individual,
    occupied_cells
)

TauschModus = []  # (idx1, idx2) oder None

def _draw_arrowhead(
    painter: QPainter,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    length: float = 0.25,
) -> None:
    """Pfeilspitzen am Strichende (x1,y1)->(x2,y2) in Welt koordinaten"""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return
    theta = math.atan2(dy, dx)
    ang = math.radians(28.0)

    x3 = x2 - length * math.cos(theta - ang)
    y3 = y2 - length * math.sin(theta - ang)
    x4 = x2 - length * math.cos(theta + ang)
    y4 = y2 - length * math.sin(theta + ang)

    poly = QPolygonF([QPointF(x2, y2), QPointF(x3, y3), QPointF(x4, y4)])
    painter.save()
    painter.setBrush(QBrush(painter.pen().color()))
    painter.drawPolygon(poly)
    painter.restore()


class LayoutCanvas(QWidget):
    """Große Ansicht: 1 Layout + geroutete Wege (Material oder Fußweg) + Entry/Exit + Labels
    Achsen:
      - X: 0..FLOOR_W (Meter), Strich alle 5m
      - Y: 0..FLOOR_H (Meter), Strich alle 5m
      - Orientierung wie im Code: y wächst nach unten (Qt-Standard)

    Interaktion:
        - Linksklick auf Maschine: Rotation +90° (Uhrzeigersinn) und Fitness-Neuberechnung
        - Modus "Tauschen": zwei Linksklicks auf Maschinen tauschen Positionen + Fitness-Neuberechnung
        - Modus "Bewegen": per pfeil-tasten verschiebens
    """

    layout_changed = pyqtSignal(object, float)  # (layout_data, score)

    # Margin (Pixel) für Achsenbeschriftung
    margin_left = 55
    margin_right = 15
    margin_top = 15
    margin_bottom = 45

    tick_step_m = 5.0  # Achsenticks alle 5 Meter

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_data = None
        self.flow_mode = "Material"  # "material" oder "foot"

        # Viewport / Transform (Device -> World)
        self._sx = 1.0
        self._sy = 1.0
        self._origin_x = 0.0  # Pixel
        self._origin_y = 0.0  # Pixel

        self._selected_idx: int | None = None
        self._routed_cache: dict | None = None

        self.setFixedSize(800, 580)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_layout(self, layout_data):
        """Input: layout_data (List[Dict]); Output: None; Nutzen: setzt Layout + triggert Repaint"""
        self.layout_data = layout_data
        self._selected_idx = None
        self._routed_cache = None
        self.update()

    def set_flow_mode(self, mode: str) -> None:
        """Input: 'material'/'foot'; Output: None; Nutzen: schaltet Ansicht um"""
        if mode == "Worker": flow_mode="Worker"
        elif mode == "Anschlüsse": flow_mode="Anschlüsse"
        elif mode == "Material": flow_mode="Material"
        else: flow_mode="Material"  # fallback
        self.flow_mode = flow_mode
        self.update()

    def _device_to_world(self, pos: QPointF) -> tuple[float, float]:
        """Input: Mausposition in Pixel; Output: (x,y) in Metern; Nutzen: korrektes Hit-Testing"""
        sx = float(self._sx) if float(self._sx) > 0 else 1.0
        sy = float(self._sy) if float(self._sy) > 0 else 1.0
        wx = (float(pos.x()) - float(self._origin_x)) / sx
        wy = (float(pos.y()) - float(self._origin_y)) / sy
        return wx, wy

    def _hit_test_machine(self, wx: float, wy: float) -> int | None:
        """Input: Weltpunkt; Output: Maschinenindex oder None; Nutzen: Klick-Auswahl"""
        if not self.layout_data:
            return None
        for mi, m in enumerate(self.layout_data):
            cx = float(m.get("x", 0.0))
            cy = float(m.get("y", 0.0))
            rot = int(m.get("z", 0)) % 360
            w_m = float(m.get("w_cells", 0)) * float(config.GRID_SIZE)
            h_m = float(m.get("h_cells", 0)) * float(config.GRID_SIZE)

            dx = wx - cx
            dy = wy - cy

            a = math.radians(-rot)
            ca = math.cos(a)
            sa = math.sin(a)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca

            if abs(lx) <= (w_m / 2.0) and abs(ly) <= (h_m / 2.0):
                return mi
        return None

    def _rotate_clockwise(self, mi: int) -> float:
        """Input: index; Output: new fitness; Nutzen: interaktives Drehen + Score update"""
        if not self.layout_data or not (0 <= int(mi) < len(self.layout_data)):
            return float("nan")

        m = self.layout_data[int(mi)]

        fixed_list = getattr(config, "MACHINE_FIXED", [])
        idx = int(m.get("idx", -1))
        if 0 <= idx < len(fixed_list) and fixed_list[idx] is not None:
            return float("nan") # feste Maschine, keine Rotation

        cx = float(m.get("x", 0.0))
        cy = float(m.get("y", 0.0))

        new_z = (int(m.get("z", 0)) + 90) % 360
        w_eff, h_eff = effective_dims(m, new_z)

        gs = float(config.GRID_SIZE) if float(config.GRID_SIZE) > 0 else 1.0
        gx = int(round((cx / gs) - (float(w_eff) / 2.0)))
        gy = int(round((cy / gs) - (float(h_eff) / 2.0)))

        m["z"] = int(new_z)
        m["gx"] = int(gx)
        m["gy"] = int(gy)
        m["x"], m["y"] = cell_center_from_topleft(int(gx), int(gy), int(w_eff), int(h_eff))

        normalize_individual(self.layout_data)
        self._routed_cache = None

        from ga_engine import fitness

        score = float(fitness(self.layout_data))
        self.layout_changed.emit(self.layout_data, score)
        self.update()
        return score

    def _tauschen(self, mi1: int, mi2: int) -> float:
        """
        Input: zwei Indizes; Output: new fitness; 
        Nutzen: interaktives Tauschen + Score update
        """
        if not self.layout_data or not (0 <= int(mi1) < len(self.layout_data)) or not (0 <= int(mi2) < len(self.layout_data)):
            return float("nan")
        
        m1, m2 = self.layout_data[int(mi1)], self.layout_data[int(mi2)]

        fixed_list = getattr(config, "MACHINE_FIXED", [])
        idx1 = int(m1.get("idx", -1))
        if 0 <= idx1 < len(fixed_list) and fixed_list[idx1] is not None:
            return float("nan") # feste Maschine, kein Tausch
        
        idx2 = int(m2.get("idx", -1))
        if 0 <= idx2 < len(fixed_list) and fixed_list[idx2] is not None:
            return float("nan") # feste Maschine, kein Tausch

        m1_gx, m1_gy = m1["gx"], m1["gy"]
        m2_gx, m2_gy = m2["gx"], m2["gy"]

        m1["gx"], m2["gx"] = m2_gx, m1_gx
        m1["gy"], m2["gy"] = m2_gy, m1_gy

        w1, h1 = effective_dims(m1, int(m1.get("z", 0)))
        w2, h2 = effective_dims(m2, int(m2.get("z", 0)))

        m1["x"], m1["y"] = cell_center_from_topleft(int(m1["gx"]), int(m1["gy"]), int(w1), int(h1))
        m2["x"], m2["y"] = cell_center_from_topleft(int(m2["gx"]), int(m2["gy"]), int(w2), int(h2))

        normalize_individual(self.layout_data)
        self._routed_cache = None

        from ga_engine import fitness

        score = float(fitness(self.layout_data))
        self.layout_changed.emit(self.layout_data, score)
        self.update()
        return score

    def _bewegen(self, m_id: int, dx: int, dy: int) -> None:
        """
        Bewegt die aktuell selektierte Maschine um (dx,dy) Grid-Zellen
        Input:
        dx, dy: Verschiebung in Grid-Zellen (z.B. Pfeil rechts => dx=+1)
        Output:
        None (mutiert layout_data in-place)
        Nutzen:
        Pfeiltasten-Bewegung im "Bewegen"-Modus, ohne Maschine/Dict zu tauschen.
        """
        if not self.layout_data:
            return
        if self._selected_idx is None:
            return
        if not (0 <= self._selected_idx < len(self.layout_data)):
            return

        m = self.layout_data[self._selected_idx]

        fixed_list = getattr(config, "MACHINE_FIXED", [])
        idx = int(m.get("idx", -1))
        if 0 <= idx < len(fixed_list) and fixed_list[idx] is not None:
            return # feste Maschine, keine Bewegung

        new_gx = int(m["gx"]) + int(dx)
        new_gy = int(m["gy"]) + int(dy)

        m["gx"] = int(new_gx)
        m["gy"] = int(new_gy)
        w_eff, h_eff = effective_dims(m, int(m.get("z", 0)))
        m["x"], m["y"] = cell_center_from_topleft(int(new_gx), int(new_gy), int(w_eff), int(h_eff))

        self._routed_cache = None

        from ga_engine import fitness
        score = float(fitness(self.layout_data))
        self.layout_changed.emit(self.layout_data, score)
        self.update()
        return score


    def keyPressEvent(self, event):
        if config.MODUS != "Bewegen":
            return super().keyPressEvent(event)

        key = event.key()
        if key == Qt.Key.Key_Left:
            self._bewegen(self._selected_idx, -1, 0)
            return
        if key == Qt.Key.Key_Right:
            self._bewegen(self._selected_idx, 1, 0)
            return
        if key == Qt.Key.Key_Up:
            self._bewegen(self._selected_idx, 0, -1)
            return
        if key == Qt.Key.Key_Down:
            self._bewegen(self._selected_idx, 0, 1)
            return

        return super().keyPressEvent(event)

    def mousePressEvent(self, event):   
        """Klick -> Maschinenkoordinaten -> Hit-Test"""
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        wx, wy = self._device_to_world(event.position())
        mi = self._hit_test_machine(wx, wy)
        self.setFocus()
        if mi is None:
            self._selected_idx = None
            self.update()
            return
        if config.MODUS == "Rotation":
            self._selected_idx = int(mi)
            self._rotate_clockwise(int(mi))
        elif config.MODUS == "Tauschen":
            self._selected_idx = int(mi)
            self.layout_changed.emit(self.layout_data, float('nan'))
            self.update()
            TauschModus.append(int(mi))
            if len(TauschModus) == 2:
                self._tauschen(TauschModus[0], TauschModus[1])
                TauschModus.clear()
        else: # Bewegen
            self._selected_idx = int(mi)
            self.layout_changed.emit(self.layout_data, float('nan'))
            self.update()


    def _draw_axes(
        self,
        painter: QPainter,
        *,
        left: int,
        top: int,
        inner_w: int,
        inner_h: int,
        sx: float,
        sy: float,
    ) -> None:
        """Zeichnet Achsen im Pixelraum (nach restore), Strich alle 5m"""
        right = left + inner_w
        bottom = top + inner_h

        axis_pen = QPen(QColor(0, 0, 0))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)

        # Achsenlinien
        painter.drawLine(left, bottom, right, bottom)  # X unten
        painter.drawLine(left, top, left, bottom)      # Y links

        tick_len = 6
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)

        step = float(self.tick_step_m)

        # X: 0..FLOOR_W (Meter)
        x_m = 0.0
        while x_m <= float(config.FLOOR_W) + 1e-9:
            x_px = left + int(round(x_m * sx))
            painter.drawLine(x_px, bottom, x_px, bottom + tick_len)
            painter.drawText(x_px - 12, bottom + 22, f"{x_m:g}m")
            x_m += step

        # Y: 0..FLOOR_H (Meter)
        y_m = 0.0
        while y_m <= float(config.FLOOR_H) + 1e-9:
            y_px = bottom - int(round(y_m * sy))
            painter.drawLine(left - tick_len, y_px, left, y_px)
            painter.drawText( left - 40, y_px + 4, f"{y_m:g}m")
            y_m += step

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background (Widget)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        BackroundWidth = int(self.width())
        backroundHeight = int(self.height())

        MarginLeft = int(self.margin_left)
        MarginRight = int(self.margin_right)
        MarginTop = int(self.margin_top)
        MarginBottom = int(self.margin_bottom)

        InnerWidth = max(1, BackroundWidth - MarginLeft - MarginRight)
        InnerHeight = max(1, backroundHeight - MarginTop - MarginBottom)

        ScaledX = InnerWidth / float(config.FLOOR_W) if float(config.FLOOR_W) > 0 else 1.0
        ScaledY = InnerHeight / float(config.FLOOR_H) if float(config.FLOOR_H) > 0 else 1.0
        FacilityScale = min(ScaledX, ScaledY)
        FacilityWidth = float(config.FLOOR_W) * FacilityScale
        FacilityHeight = float(config.FLOOR_H) * FacilityScale
        FacilityCenterX = float(MarginLeft) + (float(InnerWidth) - FacilityWidth) / 2
        FacilityCenterY = float(MarginTop) + (float(InnerHeight) - FacilityHeight) / 2

        # Store viewport for hit-testing
        self._sx = float(FacilityScale)
        self._sy = float(FacilityScale)
        self._origin_x = float(FacilityCenterX)
        self._origin_y = float(FacilityCenterY)

        # ---------------- WORLD DRAW (translated+scaled into inner area) ----------------
        painter.save()
        painter.translate(FacilityCenterX, FacilityCenterY)
        painter.scale(FacilityScale, FacilityScale)

        labels: list[tuple[float, float, str]] = []

        #Floor
        pen = QPen(QColor(180, 180, 180))
        pen.setWidthF(2.0 / ScaledX)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

        #Grid
        grid_pen = QPen(QColor(200, 200, 200))
        grid_pen.setWidthF(0.5 / ScaledX)
        painter.setPen(grid_pen)
        for c in range(int(config.GRID_COLS) + 1):
            x = c * float(config.GRID_SIZE)
            painter.drawLine(QPointF(x, 0.0), QPointF(x, float(config.FLOOR_H)))
        for r in range(int(config.GRID_ROWS) + 1):
            y = r * float(config.GRID_SIZE)
            painter.drawLine(QPointF(0.0, y), QPointF(float(config.FLOOR_W), y))

        #Obstacles
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(90, 90, 90)))
        for (col, row) in getattr(config, "OBSTACLES", []):
            rx = float(col) * float(config.GRID_SIZE)
            ry = float(row) * float(config.GRID_SIZE)
            painter.drawRect(QRectF(rx, ry, float(config.GRID_SIZE), float(config.GRID_SIZE)))

        #Entry, Exit, Water, Gas, Other markers
        ex_x, ex_y = cell_center_from_topleft(int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]), 1, 1)
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.setPen(QPen(QColor(0, 160, 0)))
        painter.drawEllipse(QPointF(ex_x, ex_y), 0.1, 0.1)

        ox, oy = cell_center_from_topleft(int(config.EXIT_CELL[0]), int(config.EXIT_CELL[1]), 1, 1)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.setPen(QPen(QColor(160, 0, 0)))
        painter.drawEllipse(QPointF(ox, oy), 0.1, 0.1)

        #Machines + clearance + alle input points
        if self.layout_data:
            for Machine, m in enumerate(self.layout_data):
                #print(f"Maschine {Machine}: x={m['x']}, y={m['y']}, z={m.get('z', 0)}, w_cells={m['w_cells']}, h_cells={m['h_cells']} und worker={m.get('worker')}")
                cx = float(m["x"])
                cy = float(m["y"])
                rot = int(m.get("z", 0))
                w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
                h_m = float(m["h_cells"]) * float(config.GRID_SIZE)

                #==========================================================================================
                #===================== Clearance Zellen ===================================================
                worker_clearance = set()
                worker_clearance |= occupied_cells(m, True) - occupied_cells(m, False) #Nur gesperrte durch Worker hinzufügen
                if worker_clearance is not None:
                    gs = float(config.GRID_SIZE)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(232,97,0)))
                    for col, row in worker_clearance:
                        painter.drawRect(QRectF(col * gs, row * gs, gs, gs))
                #=============================================================================================
                #========================================================================================

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)

                rect = QRectF(-w_m / 2, -h_m / 2, w_m, h_m)

                #Maschinen Zeichnen
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(172, 213, 230)))
                painter.drawRect(rect)

                #Umrandung ausgewählter Maschine
                if self._selected_idx == Machine:
                    sel_pen = QPen(QColor(0, 120, 255))
                    sel_pen.setWidthF(2.5 / ScaledX)
                    painter.setPen(sel_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(rect)


                MaschineLabel = str(m.get("label",None))
                if MaschineLabel is None or MaschineLabel == "":
                        MaschineLabel = str(int(m.get("idx", None)) + 1)
                labels.append((float(m["x"]), float(m["y"]), MaschineLabel))

                painter.restore()

                #Worker dot (world coords)
                work_r = 0.2
                work_pen = QPen(QColor(0, 0, 0))
                work_pen.setWidthF(1.0 / ScaledX)
                painter.setPen(work_pen)
                painter.setBrush(QBrush(QColor(232,97,0)))
                wpt = machine_worker_point(m)
                if wpt is not None:
                    wx, wy = wpt
                    painter.drawEllipse(QPointF(wx, wy), work_r, work_r)

                #=================================================================================
                #===============Hier werden anschlüsse gezeichnet=================================
                wpt = machine_water_point(m)
                if wpt is not None:
                    anschluss_r = 0.1
                    water_pen = QPen(QColor(0, 0, 0))
                    water_pen.setWidthF(1.0 / ScaledX)
                    painter.setPen(water_pen)
                    painter.setBrush(QBrush(QColor(0, 0, 255)))
                    wx, wy = wpt
                    painter.drawEllipse(QPointF(wx, wy), anschluss_r, anschluss_r)

                gpt = machine_gas_point(m)
                if gpt is not None:
                    anschluss_r = 0.1
                    gas_pen = QPen(QColor(0, 0, 0))
                    gas_pen.setWidthF(1.0 / ScaledX)
                    painter.setPen(gas_pen)
                    painter.setBrush(QBrush(QColor(139,69,19)))
                    gx, gy = gpt
                    painter.drawEllipse(QPointF(gx, gy), anschluss_r, anschluss_r)

                opt = machine_other_point(m)
                if opt is not None:
                    anschluss_r = 0.1
                    other_pen = QPen(QColor(0, 0, 0))
                    other_pen.setWidthF(1.0 / ScaledX)
                    painter.setPen(other_pen)
                    painter.setBrush(QBrush(QColor(238,238,0)))
                    ox, oy = opt
                    painter.drawEllipse(QPointF(ox, oy), anschluss_r, anschluss_r)
                #=================================================================================
                #=================================================================================

        if config.WATER_CELL[0] is not None:
            pen = QPen(QColor(0, 0, 255))
            pen.setWidthF(0.1)
            if config.WATER_CELL[1] is not None:
                painter.setBrush(QBrush(QColor(0, 0, 255)))
                painter.setPen(pen)
                painter.drawLine(QPointF(config.WATER_CELL[0][0], config.WATER_CELL[0][1]),QPointF(config.WATER_CELL[1][0], config.WATER_CELL[1][1]))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(pen)
                painter.drawEllipse(QPointF(config.WATER_CELL[0][0], config.WATER_CELL[0][1]), 0.2, 0.2)

        if config.GAS_CELL[0] is not None:
            pen = QPen(QColor(139,69,19))
            pen.setWidthF(0.1)
            if config.GAS_CELL[1] is not None:
                painter.setBrush(QBrush(QColor(139,69,19)))
                painter.setPen(pen)
                painter.drawLine(QPointF(config.GAS_CELL[0][0], config.GAS_CELL[0][1]), QPointF(config.GAS_CELL[1][0], config.GAS_CELL[1][1]))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(pen)
                painter.drawEllipse(QPointF(config.GAS_CELL[0][0], config.GAS_CELL[0][1]), 0.2, 0.2)

        if config.OTHER_CELL[0] is not None:      
            pen = QPen(QColor(238,238,0))
            pen.setWidthF(0.1)
            if config.OTHER_CELL[1] is not None:
                painter.setBrush(QBrush(QColor(238,238,0)))
                painter.setPen(pen)
                painter.drawLine(QPointF(config.OTHER_CELL[0][0], config.OTHER_CELL[0][1]), QPointF(config.OTHER_CELL[1][0], config.OTHER_CELL[1][1]))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(pen)
                painter.drawEllipse(QPointF(config.OTHER_CELL[0][0], config.OTHER_CELL[0][1]), 0.2, 0.2)

        #Routed paths (A*)
        if self.layout_data:
            if self._routed_cache is None:
                self._routed_cache = compute_routed_edges(self.layout_data)
            routed = self._routed_cache


            WorkerBrush = QBrush(QColor(0, 255, 0, 60))
            painter.setBrush(WorkerBrush)
            painter.setPen(Qt.PenStyle.NoPen)
            UniqueWorkerPathCells: set[tuple[int, int]] = set()
            for Worker in routed.get("worker", []):
                if Worker.get("WorkerPathCells") is None:
                    continue
                WorkerPathCells = Worker.get("WorkerPathCells")
                UniqueWorkerPathCells.update(WorkerPathCells)
            for Cell in UniqueWorkerPathCells:
                painter.drawRect(QRectF(Cell[0] * gs, Cell[1] * gs, gs, gs))

            if self.flow_mode == "Worker":
                pen = QPen(QColor(232,97,0))
                pen.setWidthF(1.5 / ScaledX)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                for e in routed.get("worker", []):                          
                    pts = e.get("pts") or []
                    if len(pts) < 2:
                        continue
                    for i in range(len(pts) - 1):
                        xa, ya = pts[i]
                        xb, yb = pts[i + 1]
                        painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))

            elif self.flow_mode == "Anschlüsse":

                pen_w = QPen(QColor(0, 0, 255))
                pen_w.setWidthF(1.5 / ScaledX)
                painter.setPen(pen_w)
                
                #Wasseranschluss Zeichnen
                for w in routed.get("water", []):
                    pts = w.get("pts") or []
                    if len(pts) < 2:
                        continue
                    w1_x, w1_y = pts[0]
                    w2_x, w2_y = pts[1]
                    painter.drawLine(QPointF(w1_x, w1_y), QPointF(w2_x, w2_y))

                #Gasanschluss Zeichnen
                pen_g = QPen(QColor(139,69,19))
                pen_g.setWidthF(1.5 / ScaledX)
                painter.setPen(pen_g)
                for g in routed.get("gas", []):
                    pts = g.get("pts") or []
                    if len(pts) < 2:
                        continue
                    g1_x, g1_y = pts[0]
                    g2_x, g2_y = pts[1]
                    painter.drawLine(QPointF(g1_x, g1_y), QPointF(g2_x, g2_y))

                #Sonstiger Anschluss Zeichnen
                pen_o = QPen(QColor(238,238,0))
                pen_o.setWidthF(1.5 / ScaledX)
                painter.setPen(pen_o)
                for o in routed.get("other", []):
                    pts = o.get("pts") or []
                    if len(pts) < 2:
                        continue
                    o1_x, o1_y = pts[0]
                    o2_x, o2_y = pts[1]
                    painter.drawLine(QPointF(o1_x, o1_y), QPointF(o2_x, o2_y))
                
            else:
                pen = QPen(QColor(30, 30, 30))
                pen.setWidthF(1.5 / ScaledX)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                for e in routed.get("material", []):
                    pts = e.get("pts") or []
                    if len(pts) < 2:
                        continue
                    for i in range(len(pts) - 1):
                        xa, ya = pts[i]
                        xb, yb = pts[i + 1]
                        painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))
                    _draw_arrowhead(painter, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], length=0.25)

            # Ports (above lines)
            port_r = 0.12
            port_pen = QPen(QColor(0, 0, 0))
            port_pen.setWidthF(1.0 / ScaledX)
            painter.setPen(port_pen)

            painter.setBrush(QBrush(QColor(0, 200, 0)))
            for m in self.layout_data:
                x, y = machine_input_point(m)
                painter.drawEllipse(QPointF(x, y), port_r, port_r)

            painter.setBrush(QBrush(QColor(220, 0, 0)))
            for m in self.layout_data:
                x, y = machine_output_point(m)
                painter.drawEllipse(QPointF(x, y), port_r, port_r)

        # END world draw
        painter.restore()

        # ---------------- DEVICE DRAW (pixel space) ----------------

        # Labels (pixel space, include margins!)
        if labels:
            painter.setPen(QPen(QColor(0, 0, 0)))
            font = QFont()
            font.setPixelSize(14)
            painter.setFont(font)

            for xw, yw, text in labels:
                px = FacilityCenterX + xw * FacilityScale
                py = FacilityCenterY + yw * FacilityScale
                r = QRectF(px - 18, py - 10, 36, 20)
                painter.drawText(r, int(Qt.AlignmentFlag.AlignCenter), text)

        # Axes (diagram style)
        self._draw_axes(
            painter,
            left = int(round(FacilityCenterX)),
            top = int(round(FacilityCenterY)),
            inner_w = int(round(FacilityWidth)),
            inner_h = int(round(FacilityHeight)),
            sx = float(FacilityScale),
            sy = float(FacilityScale),
        )


class PopulationCanvas(QWidget):
    """Übersicht: viele Individuen als Miniaturen (ohne Flows, ohne Entry/Exit)"""

    def __init__(self, parent=None, cols=10, rows=1):
        super().__init__(parent)
        self.population = None
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(900, 100)

    def set_population(self, population):
        """Input: List[List[Dict]]; Output: None; Nutzen: setzt Population und repainted"""
        self.population = population
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
            scale = min(avail_w / config.FLOOR_W, avail_h / config.FLOOR_H)

            painter.translate(cell_x + margin, cell_y + margin)
            painter.scale(scale, scale)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(245, 245, 245)))
            painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

            painter.setBrush(QBrush(QColor(90, 90, 90)))
            for (col_o, row_o) in config.OBSTACLES:
                ox = col_o * config.GRID_SIZE
                oy = row_o * config.GRID_SIZE
                painter.drawRect(QRectF(ox, oy, config.GRID_SIZE, config.GRID_SIZE))

            for m in ind:
                cx = float(m["x"])
                cy = float(m["y"])
                rot = int(m.get("z", 0))
                w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
                h_m = float(m["h_cells"]) * float(config.GRID_SIZE)

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)
                painter.setBrush(QBrush(QColor(172, 213, 230)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(-w_m / 2, -h_m / 2, w_m, h_m))
                painter.restore()

            painter.restore()


class BestDialog(QDialog):
    """Dialog: zeigt die beste Lösung (LayoutCanvas) im großen Format

    Features:
        - Toggle: Materialfluss anzeigen <-> Fußweg anzeigen (Worker-Routen)
        - Modus wechseln: Rotation <-> Tauschen + Fitness-neuberechnung
        - Klick auf Maschine: Rotation + Fitness-Neuberechnung
    """

    def __init__(self, layout_data, parent=None, title="Beste Lösung"):
        super().__init__(parent)
        self.setWindowTitle(title)

        from ga_engine import fitness

        self._ga_layout = copy.deepcopy(layout_data)
        self._current_layout = copy.deepcopy(self._ga_layout)
        self._saved_layout = None
        self._showing_saved = False

        if parent is not None and hasattr(parent, "_saved_best_layout") and parent._saved_best_layout is not None:
            self._saved_layout = copy.deepcopy(parent._saved_best_layout)


        root = QVBoxLayout(self)
        top = QHBoxLayout()
        Modes = QVBoxLayout()
        Saves = QVBoxLayout()     

        self.score_label = QLabel(f"Fitness: {float(fitness(self._ga_layout)):.2f}")
        
        #============================================GRUPPEN BUTTON===========================
        
        self.group_btn = QPushButton("Gruppen ansehen")
        self.group_btn.setFixedSize(200, 30)
        self.group_btn.clicked.connect(self._open_group_viewer)
        Modes.addWidget(self.group_btn)

        #=====================================================================================
        #=====================================================================================

        self.layout_button = QPushButton("Gespeichertes Ergebniss")
        self.layout_button.setFixedSize(200, 30)
        self.layout_button.clicked.connect(self._toggle_layout)

        self.save_button = QPushButton("Layout Speichern")
        self.save_button.setFixedSize(200, 30)
        self.save_button.clicked.connect(self._save_layout)

        self.toggle_btn = QPushButton("Materialwege")
        self.toggle_btn.setFixedSize(200, 30)
        self.toggle_btn.clicked.connect(self._toggle_mode)
        
        self.tausch_btn = QPushButton(config.MODUS)
        self.tausch_btn.setFixedSize(200, 30)
        self.tausch_btn.clicked.connect(self._change_mode)

        top.addWidget(self.score_label, 1)
        Saves.addWidget(self.layout_button)
        Saves.addWidget(self.save_button)
        top.addLayout(Saves)
        Modes.addWidget(self.toggle_btn)
        Modes.addWidget(self.tausch_btn)
        top.addLayout(Modes)
        root.addLayout(top)

        self.canvas = LayoutCanvas(self)
        self.canvas.set_layout(self._current_layout)
        root.addWidget(self.canvas, 1)

        def _on_changed(_layout, score: float):
            self.score_label.setText(f"Fitness: {float(score):.2f}")

        self.canvas.layout_changed.connect(_on_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    def _toggle_layout(self) -> None:
        from ga_engine import fitness

        if self._saved_layout is None:
            QMessageBox.information(self, "Kein gespeichertes Layout", "Bitte zuerst 'Layout Speichern' klicken.")
            return

        if self._showing_saved:
            self._current_layout = copy.deepcopy(self._ga_layout)
            self._showing_saved = False
            self.layout_button.setText("Gespeichertes Ergebnis")
        else:
            self._current_layout = copy.deepcopy(self._saved_layout)
            self._showing_saved = True
            self.layout_button.setText("GA Ergebnis")

        self.canvas.set_layout(self._current_layout)
        self.score_label.setText(f"Fitness: {float(fitness(self._current_layout)):.2f}")


    def _save_layout(self):
        self._saved_layout = copy.deepcopy(self.canvas.layout_data)
        if self.parent() is not None:
            self.parent()._saved_best_layout = copy.deepcopy(self._saved_layout)
        self.layout_button.setEnabled(True)

    def _toggle_mode(self) -> None:
        """Wechselt Canvas-Ansicht: Materialfluss <-> Fußweg"""
        if self.canvas.flow_mode == "Material":
            self.canvas.set_flow_mode("Worker")
            self.toggle_btn.setText("Fußwege")

        elif self.canvas.flow_mode == "Worker":
            self.canvas.set_flow_mode("Anschlüsse")
            self.toggle_btn.setText("Anschlusswege")
        else:
            self.canvas.set_flow_mode("Material")
            self.toggle_btn.setText("Materialwege")

    def _change_mode(self, mode: str) -> None:
        """Wechselt von Rotation zu tausch modus"""
        if config.MODUS == "Rotation":
            config.MODUS = "Tauschen"
        elif config.MODUS == "Tauschen":
            config.MODUS = "Bewegen"
        else:
            config.MODUS = "Rotation"
        self.tausch_btn.setText(config.MODUS)

    #=============================================================================================================
    #GRuppenknopf==============================================================================================
    def _open_group_viewer(self) -> None:
        links = getattr(config, "GROUPS_FOR_GA", None)
        if not links:
            QMessageBox.information(self, "Keine Gruppen", "Es wurden keine Gruppenlinks gespeichert. Erst GA laufen lassen.")
            return
        # zeigt die aktuell im BestDialog angezeigte Layout-Version (GA oder Saved, je nach Toggle)
        show_group_debug_viewer(self._current_layout, links, parent=self)



#=============================================================================================================
#GRUPPENTEST==============================================================================================

def _parse_group_entry(entry: Any) -> Optional[dict]:
    """
    Supports either:
      - {"leader": int, "members":[a,b], "local":{...}}
      - [a,b] or (a,b)
    Returns normalized:
      {"leader": int, "member": int, "raw": entry, "local": dict|None}
    """
    try:
        if isinstance(entry, dict):
            leader = int(entry.get("leader"))
            members = entry.get("members") or []
            if len(members) < 2:
                return None
            members_i = [int(x) for x in members]
            # pick first member that's not leader (fallback: second)
            member = next((m for m in members_i if m != leader), members_i[1])
            local = entry.get("local") or None
            return {"leader": leader, "member": int(member), "raw": entry, "local": local}

        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            leader = int(entry[0])
            member = int(entry[1])
            return {"leader": leader, "member": member, "raw": entry, "local": None}
    except Exception:
        return None
    return None

def _member_from_local(
    leader: dict,
    member_template: dict,
    local_map: Optional[dict],
    member_idx: int,
) -> dict:
    """
    Baut ein Member-Dict nur für die Anzeige, basierend auf local Offsets aus Optimize_Groups.
    Erwartet bei dir: local_map[member_idx] = {"MemberX": dx2, "MemberY": dy2, "MemberZ": dz}
    wobei dx2/dy2 Halbzellen sind (2 * Δx/GRID_SIZE).
    """
    import copy
    from helpers import effective_dims, cell_center_from_topleft

    m = copy.deepcopy(member_template)
    if not local_map:
        return m

    d = local_map.get(member_idx) or local_map.get(str(member_idx))
    if not isinstance(d, dict):
        return m

    # akzeptiere auch dx2/dy2/dz, falls du später umbenennst
    dx2 = d.get("dx2", d.get("MemberX"))
    dy2 = d.get("dy2", d.get("MemberY"))
    dz  = d.get("dz",  d.get("MemberZ"))

    if dx2 is None or dy2 is None or dz is None:
        return m

    dx2 = int(dx2)
    dy2 = int(dy2)
    dz = int(dz)

    # Pose rekonstruieren (World Center)
    gs = float(config.GRID_SIZE)
    leader_x = float(leader.get("x", 0.0))
    leader_y = float(leader.get("y", 0.0))
    leader_z = int(leader.get("z", 0)) % 360

    # MemberX/Y sind Halbzellen => /2 * GRID_SIZE = Meter
    x = leader_x + (dx2 / 2.0) * gs
    y = leader_y + (dy2 / 2.0) * gs
    z = (leader_z + dz) % 360

    m["z"] = int(z)

    # gx/gy aus center + eff dims (für occupied_cells / routing)
    w_eff, h_eff = effective_dims(m, int(z))
    gx = int(round((x / gs) - (float(w_eff) / 2.0)))
    gy = int(round((y / gs) - (float(h_eff) / 2.0)))

    max_col = max(0, int(config.GRID_COLS) - int(w_eff))
    max_row = max(0, int(config.GRID_ROWS) - int(h_eff))
    gx = max(0, min(max_col, gx))
    gy = max(0, min(max_row, gy))

    m["gx"] = int(gx)
    m["gy"] = int(gy)
    m["x"], m["y"] = cell_center_from_topleft(int(gx), int(gy), int(w_eff), int(h_eff))
    return m
class GroupCanvas(QWidget):
    """
    Zeichnet NUR zwei Maschinen + (optional) Obstacles + Worker/Material Wege.
    Kein Floor, kein Grid, keine anderen Maschinen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._leader: Optional[dict] = None
        self._member: Optional[dict] = None
        self._worker_pts: list[tuple[float, float]] = []
        self._worker_cost: float = float("nan")
        self._mat_edges: list[dict] = []  # [{"out":i,"in":j,"pts":[...],"len":float,"w":float}]
        self._bbox: Optional[tuple[float, float, float, float]] = None  # (minx,miny,maxx,maxy)

        self.show_worker = True
        self.show_material = True
        self.show_obstacles = True

        self.setMinimumSize(700, 520)

    def set_group(self, leader: dict, member: dict) -> None:
        self._leader = copy.deepcopy(leader)
        self._member = copy.deepcopy(member)
        self._recompute_routes_and_bbox()
        self.update()

    def _recompute_routes_and_bbox(self) -> None:
        if not self._leader or not self._member:
            self._worker_pts = []
            self._mat_edges = []
            self._bbox = None
            return

        from helpers import (
            AStar_Worker_Path,
            get_worker_clearance,
            machine_input_point,
            machine_output_point,
            occupied_cells,
            rect_corners,
            route_world_to_world,
            _blocked_cells_for_routing,
            _blocked_signature,
        )

        leader = self._leader
        member = self._member

        # group-only blocked: OBSTACLES + footprints (no clearance) for these two
        blocked = _blocked_cells_for_routing([leader, member])

        # Worker route
        wr = AStar_Worker_Path(get_worker_clearance(leader), get_worker_clearance(member), blocked)
        if wr is None:
            self._worker_cost = float("inf")
            self._worker_pts = []
        else:
            _, cost, traced = wr
            self._worker_cost = float(cost)
            self._worker_pts = list(traced or [])

        # Material routes (both directions if edges exist)
        edges = list(getattr(config, "MATERIAL_CONNECTIONS", []) or [])
        weights = {}
        for e in edges:
            if not e or len(e) < 3:
                continue
            o, i, w = e[0], e[1], e[2]
            if o is None or i is None or w is None:
                continue
            try:
                weights[(int(o), int(i))] = float(w)
            except (TypeError, ValueError):
                continue

        self._mat_edges = []
        blocked_sig = _blocked_signature(blocked)

        def _add_edge(out_idx: int, in_idx: int) -> None:
            w = weights.get((int(out_idx), int(in_idx)))
            if w is None:
                return
            out_m = leader if int(leader.get("idx")) == int(out_idx) else member
            in_m = leader if int(leader.get("idx")) == int(in_idx) else member
            p1 = machine_output_point(out_m)
            p2 = machine_input_point(in_m)
            pts, length = route_world_to_world(p1, p2, blocked=blocked, blocked_sig=blocked_sig)
            self._mat_edges.append(
                {"out": int(out_idx), "in": int(in_idx), "pts": pts or [], "len": float(length), "w": float(w)}
            )

        a = int(leader.get("idx", -1))
        b = int(member.get("idx", -1))
        _add_edge(a, b)
        _add_edge(b, a)

        # --- bbox in world coords (machines + route points) ---
        pts_world: list[tuple[float, float]] = []

        def _add_machine_bbox(m: dict) -> None:
            w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
            h_m = float(m["h_cells"]) * float(config.GRID_SIZE)
            corners = rect_corners((float(m["x"]), float(m["y"])), w_m, h_m, int(m.get("z", 0)))
            pts_world.extend(corners)

        _add_machine_bbox(leader)
        _add_machine_bbox(member)

        pts_world.extend(self._worker_pts)
        for e in self._mat_edges:
            pts_world.extend(e.get("pts") or [])

        if not pts_world:
            self._bbox = (0.0, 0.0, 1.0, 1.0)
            return

        xs = [p[0] for p in pts_world]
        ys = [p[1] for p in pts_world]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        pad = 0.8  # meters padding
        self._bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(245, 245, 245))

        if not self._leader or not self._member or not self._bbox:
            painter.setPen(QPen(QColor(30, 30, 30)))
            painter.drawText(20, 30, "Keine Gruppe gesetzt.")
            return

        from helpers import (
            machine_input_point,
            machine_output_point,
            machine_worker_point,
        )

        minx, miny, maxx, maxy = self._bbox
        bbox_w = max(1e-6, maxx - minx)
        bbox_h = max(1e-6, maxy - miny)

        W = float(self.width())
        H = float(self.height())
        margin = 20.0
        inner_w = max(1.0, W - 2 * margin)
        inner_h = max(1.0, H - 2 * margin)

        scale = min(inner_w / bbox_w, inner_h / bbox_h)
        scale = max(1e-6, scale)

        # world transform: fit bbox into widget
        offset_world_x = (inner_w / scale - bbox_w) / 2.0
        offset_world_y = (inner_h / scale - bbox_h) / 2.0

        painter.save()
        painter.translate(margin, margin)
        painter.scale(scale, scale)
        painter.translate(-minx + offset_world_x, -miny + offset_world_y)

        # draw nearby obstacles (only those inside bbox area)
        if self.show_obstacles:
            obs_pen = Qt.PenStyle.NoPen
            painter.setPen(obs_pen)
            painter.setBrush(QBrush(QColor(120, 120, 120)))
            gs = float(config.GRID_SIZE)
            c0 = int(max(0, math.floor(minx / gs) - 2))
            c1 = int(min(int(config.GRID_COLS) - 1, math.ceil(maxx / gs) + 2))
            r0 = int(max(0, math.floor(miny / gs) - 2))
            r1 = int(min(int(config.GRID_ROWS) - 1, math.ceil(maxy / gs) + 2))
            obs = set(getattr(config, "OBSTACLES", []) or [])
            for c in range(c0, c1 + 1):
                for r in range(r0, r1 + 1):
                    if (c, r) in obs:
                        painter.drawRect(QRectF(c * gs, r * gs, gs, gs))

        # helper to draw a machine
        def _draw_machine(m: dict, *, outline: Optional[QColor] = None) -> None:
            cx = float(m["x"])
            cy = float(m["y"])
            rot = int(m.get("z", 0))
            w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
            h_m = float(m["h_cells"]) * float(config.GRID_SIZE)

            painter.save()
            t = QTransform()
            t.translate(cx, cy)
            t.rotate(rot)
            painter.setTransform(t, True)

            rect = QRectF(-w_m / 2.0, -h_m / 2.0, w_m, h_m)
            painter.setBrush(QBrush(QColor(172, 213, 230)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)

            if outline is not None:
                pen = QPen(outline)
                pen.setWidthF(2.2 / scale)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)

            painter.restore()

        # leader highlighted
        _draw_machine(self._leader, outline=QColor(0, 120, 255))
        _draw_machine(self._member, outline=QColor(60, 60, 60))

        # Worker points
        work_pen = QPen(QColor(0, 0, 0))
        work_pen.setWidthF(1.0 / scale)
        painter.setPen(work_pen)
        painter.setBrush(QBrush(QColor(232, 97, 0)))
        for m in (self._leader, self._member):
            wpt = machine_worker_point(m)
            if wpt is None:
                continue
            wx, wy = wpt
            painter.drawEllipse(QPointF(wx, wy), 0.18, 0.18)

        # Paths
        if self.show_worker and self._worker_pts:
            pen_w = QPen(QColor(232, 97, 0))
            pen_w.setWidthF(2.0 / scale)
            painter.setPen(pen_w)
            for i in range(len(self._worker_pts) - 1):
                x1, y1 = self._worker_pts[i]
                x2, y2 = self._worker_pts[i + 1]
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        if self.show_material:
            pen_m = QPen(QColor(30, 30, 30))
            pen_m.setWidthF(2.0 / scale)
            painter.setPen(pen_m)

            for e in self._mat_edges:
                pts = e.get("pts") or []
                if len(pts) < 2:
                    continue
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                _draw_arrowhead(painter, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], length=0.25)

        # Ports
        port_r = 0.12
        port_pen = QPen(QColor(0, 0, 0))
        port_pen.setWidthF(1.0 / scale)
        painter.setPen(port_pen)

        painter.setBrush(QBrush(QColor(0, 200, 0)))
        for m in (self._leader, self._member):
            x, y = machine_input_point(m)
            painter.drawEllipse(QPointF(x, y), port_r, port_r)

        painter.setBrush(QBrush(QColor(220, 0, 0)))
        for m in (self._leader, self._member):
            x, y = machine_output_point(m)
            painter.drawEllipse(QPointF(x, y), port_r, port_r)

        painter.restore()

        # overlay text (pixel space)
        painter.setPen(QPen(QColor(20, 20, 20)))
        font = QFont()
        font.setPixelSize(12)
        painter.setFont(font)

        leader_idx = int(self._leader.get("idx", -1))
        member_idx = int(self._member.get("idx", -1))
        txt = f"Leader idx={leader_idx}  Member idx={member_idx}  WorkerCost={self._worker_cost:.3f}"
        painter.drawText(16, 18, txt)


class GroupDebugDialog(QDialog):
    """Dialog: Blättern durch Gruppen und nur Gruppe + Wege anzeigen."""

    def __init__(self, best_ind: list[dict], group_links: list[Any], parent=None, title: str = "Gruppen Debug Viewer"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(820, 650)

        self._ind = best_ind
        self._groups = [g for g in (_parse_group_entry(x) for x in (group_links or [])) if g is not None]
        self._i = 0

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.btn_prev = QPushButton("◀ Vorherige")
        self.btn_next = QPushButton("Nächste ▶")

        self.cb_worker = QCheckBox("Workerweg")
        self.cb_worker.setChecked(True)
        self.cb_material = QCheckBox("Materialweg")
        self.cb_material.setChecked(True)
        self.cb_obs = QCheckBox("Obstacles")
        self.cb_obs.setChecked(True)

        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        top.addSpacing(10)
        top.addWidget(self.cb_worker)
        top.addWidget(self.cb_material)
        top.addWidget(self.cb_obs)
        top.addStretch(1)
        top.addWidget(self.info)

        root.addLayout(top)

        self.canvas = GroupCanvas(self)
        root.addWidget(self.canvas, 1)

        close_row = QHBoxLayout()
        btn_close = QPushButton("Schließen")
        btn_close.clicked.connect(self.accept)
        close_row.addStretch(1)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

        self.btn_prev.clicked.connect(self._prev)
        self.btn_next.clicked.connect(self._next)

        self.cb_worker.stateChanged.connect(self._apply_toggles)
        self.cb_material.stateChanged.connect(self._apply_toggles)
        self.cb_obs.stateChanged.connect(self._apply_toggles)

        self._show_current()

    def _apply_toggles(self) -> None:
        self.canvas.show_worker = bool(self.cb_worker.isChecked())
        self.canvas.show_material = bool(self.cb_material.isChecked())
        self.canvas.show_obstacles = bool(self.cb_obs.isChecked())
        self.canvas.update()

    def _prev(self) -> None:
        if not self._groups:
            return
        self._i = (self._i - 1) % len(self._groups)
        self._show_current()

    def _next(self) -> None:
        if not self._groups:
            return
        self._i = (self._i + 1) % len(self._groups)
        self._show_current()

    def _show_current(self) -> None:
        if not self._groups:
            self.info.setText("Keine Gruppen vorhanden.")
            return

        g = self._groups[self._i]
        leader = int(g["leader"])
        member = int(g["member"])

        if leader < 0 or leader >= len(self._ind) or member < 0 or member >= len(self._ind):
            self.info.setText(f"Gruppe {self._i+1}/{len(self._groups)}: Index out of range (leader={leader}, member={member})")
            return

        m_leader = self._ind[leader]
        local = g.get("local") or {}
        m_member = _member_from_local(m_leader, self._ind[member], local, member)
        self.canvas.set_group(m_leader, m_member)
        self._apply_toggles()

        local_txt = ""
        local = g.get("local") or {}
        try:
            d = local.get(member) or local.get(str(member)) or {}
            if d:
                local_txt = f" | local={d}"
        except Exception:
            pass

        self.info.setText(f"Gruppe {self._i+1}/{len(self._groups)}  leader={leader} member={member}{local_txt}")


def show_group_debug_viewer(best_ind: list[dict], group_links: list[Any], *, parent=None) -> None:
    """
    Convenience function:
      show_group_debug_viewer(best_ind, group_links, parent=some_qt_widget)
    """
    dlg = GroupDebugDialog(best_ind, group_links, parent=parent)
    dlg.exec()