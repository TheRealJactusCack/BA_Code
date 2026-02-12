from __future__ import annotations

import copy
import math
from typing import Optional

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF, QTransform
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
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

        self.setMinimumSize(800, 580)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_layout(self, layout_data):
        """Input: layout_data (List[Dict]); Output: None; Nutzen: setzt Layout + triggert Repaint."""
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
        """Zeichnet Achsen im Pixelraum (nach restore), Tick alle 5m."""
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

        # Y: 0..FLOOR_H (Meter) (y wächst nach unten)
        y_m = 0.0
        while y_m <= float(config.FLOOR_H) + 1e-9:
            y_px = top + int(round(y_m * sy))
            painter.drawLine(left - tick_len, y_px, left, y_px)
            painter.drawText(5, y_px + 4, f"{y_m:g}m")
            y_m += step

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background (Widget)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        W = int(self.width())
        H = int(self.height())

        ml = int(self.margin_left)
        mr = int(self.margin_right)
        mt = int(self.margin_top)
        mb = int(self.margin_bottom)

        inner_w = max(1, W - ml - mr)
        inner_h = max(1, H - mt - mb)

        sx = inner_w / float(config.FLOOR_W) if float(config.FLOOR_W) > 0 else 1.0
        sy = inner_h / float(config.FLOOR_H) if float(config.FLOOR_H) > 0 else 1.0

        # Store viewport for hit-testing
        self._sx = float(sx)
        self._sy = float(sy)
        self._origin_x = float(ml)
        self._origin_y = float(mt)

        # ---------------- WORLD DRAW (translated+scaled into inner area) ----------------
        painter.save()
        painter.translate(ml, mt)
        painter.scale(sx, sy)

        labels: list[tuple[float, float, str]] = []

        # Floor
        pen = QPen(QColor(180, 180, 180))
        pen.setWidthF(2.0 / sx)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

        # Grid
        grid_pen = QPen(QColor(200, 200, 200))
        grid_pen.setWidthF(0.5 / sx)
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

        if config.WATER_CELL[0] is not None and config.WATER_CELL[1] is not None:
            wx, wy = cell_center_from_topleft(int(config.WATER_CELL[0]), int(config.WATER_CELL[1]), 1, 1)
            painter.setBrush(QBrush(QColor(0, 0, 255)))
            painter.setPen(QPen(QColor(0, 0, 255)))
            painter.drawEllipse(QPointF(wx, wy), 0.05, 0.05)

        if config.GAS_CELL[0] is not None and config.GAS_CELL[1] is not None:
            gx, gy = cell_center_from_topleft(int(config.GAS_CELL[0]), int(config.GAS_CELL[1]), 1, 1)
            painter.setBrush(QBrush(QColor(139,69,19)))
            painter.setPen(QPen(QColor(139,69,19)))
            painter.drawEllipse(QPointF(gx, gy), 0.05, 0.05)

        if config.OTHER_CELL[0] is not None and config.OTHER_CELL[1] is not None:
            ox, oy = cell_center_from_topleft(int(config.OTHER_CELL[0]), int(config.OTHER_CELL[1]), 1, 1)
            painter.setBrush(QBrush(QColor(238,238,0)))
            painter.setPen(QPen(QColor(238,238,0)))
            painter.drawEllipse(QPointF(ox, oy), 0.05, 0.05)

        #Machines + clearance + alle input points
        if self.layout_data:
            for mi, m in enumerate(self.layout_data):
                #print(f"Maschine {mi}: x={m['x']}, y={m['y']}, z={m.get('z', 0)}, w_cells={m['w_cells']}, h_cells={m['h_cells']} und worker={m.get('worker')}")
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

                rect = QRectF(-w_m / 2, -h_m / 2, w_m, h_m)

                clearance_m = float(getattr(config, "MACHINE_CLEARANCE_M", 0.0) or 0.0)
                if clearance_m > 0.0:
                    pad_cells = int(math.ceil((clearance_m / 2.0) / float(config.GRID_SIZE)))
                    pad_m = pad_cells * float(config.GRID_SIZE)
                    clearance_rect = QRectF(
                        rect.x() - pad_m,
                        rect.y() - pad_m,
                        rect.width() + 2.0 * pad_m,
                        rect.height() + 2.0 * pad_m,
                    )
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(0, 255, 0, 60)))
                    painter.drawRect(clearance_rect)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(172, 213, 230)))
                painter.drawRect(rect)

                if self._selected_idx == mi:
                    sel_pen = QPen(QColor(0, 120, 255))
                    sel_pen.setWidthF(2.5 / sx)
                    painter.setPen(sel_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(rect)

                raw_idx = m.get("idx", None)
                if raw_idx is not None:
                    try:
                        label = str(int(raw_idx) + 1)
                    except Exception:
                        label = str(raw_idx)
                        print("Ui_Canvas: 460")
                    labels.append((float(m["x"]), float(m["y"]), label))

                painter.restore()

                # Worker dot (world coords)
                work_r = 0.2
                work_pen = QPen(QColor(0, 0, 0))
                work_pen.setWidthF(1.0 / sx)
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
                    water_pen.setWidthF(1.0 / sx)
                    painter.setPen(water_pen)
                    painter.setBrush(QBrush(QColor(0, 0, 255)))
                    wx, wy = wpt
                    painter.drawEllipse(QPointF(wx, wy), anschluss_r, anschluss_r)

                gpt = machine_gas_point(m)
                if gpt is not None:
                    anschluss_r = 0.1
                    gas_pen = QPen(QColor(0, 0, 0))
                    gas_pen.setWidthF(1.0 / sx)
                    painter.setPen(gas_pen)
                    painter.setBrush(QBrush(QColor(139,69,19)))
                    gx, gy = gpt
                    painter.drawEllipse(QPointF(gx, gy), anschluss_r, anschluss_r)

                opt = machine_other_point(m)
                if opt is not None:
                    anschluss_r = 0.1
                    other_pen = QPen(QColor(0, 0, 0))
                    other_pen.setWidthF(1.0 / sx)
                    painter.setPen(other_pen)
                    painter.setBrush(QBrush(QColor(238,238,0)))
                    ox, oy = opt
                    painter.drawEllipse(QPointF(ox, oy), anschluss_r, anschluss_r)
                #=================================================================================
                #=================================================================================

        # Routed paths (A*)
        if self.layout_data:
            if self._routed_cache is None:
                self._routed_cache = compute_routed_edges(self.layout_data)
            routed = self._routed_cache

            if self.flow_mode == "Worker":
                pen = QPen(QColor(232,97,0))
                pen.setWidthF(1.5 / sx)
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
                pen_w.setWidthF(1.5 / sx)
                painter.setPen(pen_w)
                
                #Wasseranschluss Zeichnen
                for w in routed.get("water", []):
                    pts = w.get("pts") or []
                    if len(pts) < 2:
                        continue
                    for i in range(len(pts) - 1):
                        xa, ya = pts[i]
                        xb, yb = pts[i + 1]
                        painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))

                #Gasanschluss Zeichnen
                pen_g = QPen(QColor(139,69,19))
                pen_g.setWidthF(1.5 / sx)
                painter.setPen(pen_g)
                for g in routed.get("gas", []):
                    pts = g.get("pts") or []
                    if len(pts) < 2:
                        continue
                    for i in range(len(pts) - 1):
                        xa, ya = pts[i]
                        xb, yb = pts[i + 1]
                        painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))

                #Sonstiger Anschluss Zeichnen
                pen_o = QPen(QColor(238,238,0))
                pen_o.setWidthF(1.5 / sx)
                painter.setPen(pen_o)
                for o in routed.get("other", []):
                    pts = o.get("pts") or []
                    if len(pts) < 2:
                        continue
                    for i in range(len(pts) - 1):
                        xa, ya = pts[i]
                        xb, yb = pts[i + 1]
                        painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))
                
            else:
                pen = QPen(QColor(30, 30, 30))
                pen.setWidthF(1.5 / sx)
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
            port_pen.setWidthF(1.0 / sx)
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
                px = ml + xw * sx
                py = mt + yw * sy
                r = QRectF(px - 18, py - 10, 36, 20)
                painter.drawText(r, int(Qt.AlignmentFlag.AlignCenter), text)

        # Axes (diagram style)
        self._draw_axes(
            painter,
            left=ml,
            top=mt,
            inner_w=inner_w,
            inner_h=inner_h,
            sx=float(sx),
            sy=float(sy),
        )


class PopulationCanvas(QWidget):
    """Übersicht: viele Individuen als Miniaturen (ohne Flows, ohne Entry/Exit)."""

    def __init__(self, parent=None, cols=10, rows=1):
        super().__init__(parent)
        self.population = None
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(900, 100)

    def set_population(self, population):
        """Input: List[List[Dict]]; Output: None; Nutzen: setzt Population und repainted."""
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

        self._layout = copy.deepcopy(layout_data)

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.score_label = QLabel(f"Fitness: {float(fitness(self._layout)):.2f}")
        self.toggle_btn = QPushButton("Materialwege")
        self.toggle_btn.clicked.connect(self._toggle_mode)

        
        self.tausch_btn = QPushButton(config.MODUS)
        self.tausch_btn.clicked.connect(self._change_mode)

        top.addWidget(self.score_label, 1)
        top.addWidget(self.toggle_btn)
        top.addWidget(self.tausch_btn)
        root.addLayout(top)

        self.canvas = LayoutCanvas(self)
        self.canvas.set_layout(self._layout)
        root.addWidget(self.canvas, 1)

        def _on_changed(_layout, score: float):
            self.score_label.setText(f"Fitness: {float(score):.2f}")

        self.canvas.layout_changed.connect(_on_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

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