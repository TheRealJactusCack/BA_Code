# FILE: ui_canvas.py

from __future__ import annotations

import math
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QDialogButtonBox
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QTransform, QPolygonF, QFont
from PyQt6.QtCore import Qt, QPointF, QRectF

import config
from helpers import cell_center_from_topleft, machine_input_point, machine_output_point, machine_worker_point

# --- PPT-style orthogonal connectors (no routing) -------------------------------

def _side_world_from_center(cx: float, cy: float, px: float, py: float) -> str:
    """Return world-side of port relative to machine center: left/right/top/bottom."""
    dx = px - cx
    dy = py - cy
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def _dir_vec(side: str) -> tuple[int, int]:
    side = str(side).strip().lower()
    if side == "left":
        return (-1, 0)
    if side == "right":
        return (1, 0)
    if side == "top":
        return (0, -1)
    return (0, 1)  # bottom


def _is_on_ray(sx: float, sy: float, ix: float, iy: float, dx: int, dy: int) -> bool:
    """Check if intersection point (ix,iy) lies in the forward direction of ray from (sx,sy)."""
    if dx != 0:
        return (ix - sx) * dx >= -1e-9
    return (iy - sy) * dy >= -1e-9


def _orth_polyline_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    d1: tuple[int, int],
    d2: tuple[int, int],
) -> list[tuple[float, float]]:
    """
    Build a PPT-like orthogonal connector polyline from (x1,y1) to (x2,y2).
    Try 1-kink if rays intersect in forward direction; else fall back to Z with midline.
    """
    # Straight line if already aligned
    if abs(x1 - x2) < 1e-9 or abs(y1 - y2) < 1e-9:
        return [(x1, y1), (x2, y2)]

    d1x, d1y = d1
    d2x, d2y = d2

    # Candidate 1-kink if one is horizontal and the other vertical
    if (d1x != 0 and d2y != 0) or (d1y != 0 and d2x != 0):
        if d1x != 0:  # start horizontal
            ix, iy = x2, y1
        else:         # start vertical
            ix, iy = x1, y2

        ok1 = _is_on_ray(x1, y1, ix, iy, d1x, d1y)
        ok2 = _is_on_ray(x2, y2, ix, iy, d2x, d2y)
        if ok1 and ok2:
            return [(x1, y1), (ix, iy), (x2, y2)]

    # Z-shape fallback: midline on half distance (choose by start direction axis)
    if d1x != 0:  # start horizontal -> use vertical trunk at mid-x
        midx = (x1 + x2) / 2.0
        return [(x1, y1), (midx, y1), (midx, y2), (x2, y2)]
    else:         # start vertical -> use horizontal trunk at mid-y
        midy = (y1 + y2) / 2.0
        return [(x1, y1), (x1, midy), (x2, midy), (x2, y2)]


def _draw_arrowhead(painter: QPainter, x1: float, y1: float, x2: float, y2: float, *, length: float = 0.25) -> None:
    """Small arrowhead at end of segment (x1,y1)->(x2,y2) in world coords."""
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
    """Große Ansicht: 1 Layout + Materialflüsse + Entry/Exit + Labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_data = None
        self.setMinimumSize(800, 480)

    def set_layout(self, layout_data):
        """Setzt das aktuell anzuzeigende Layout (beste Lösung / einzelnes Individuum)."""
        self.layout_data = layout_data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        #Widget-Hintergrund (UI-Fläche)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        #Welt->Viewport Skalierung (Meter -> Pixel)
        w = self.width()
        h = self.height()
        sx = w / config.FLOOR_W if config.FLOOR_W > 0 else 1.0
        sy = h / config.FLOOR_H if config.FLOOR_H > 0 else 1.0

        painter.save()
        painter.scale(sx, sy)

        #Labels werden nach dem restore() in Device-Koordinaten gerendert
        labels: list[tuple[float, float, str]] = []  # (x_world, y_world, text)

        #Floor (Weltfläche)
        pen = QPen(QColor(180, 180, 180))
        pen.setWidthF(2.0 / sx)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

        #Raster (Grid)
        grid_pen = QPen(QColor(200, 200, 200))
        grid_pen.setWidthF(0.5 / sx)
        painter.setPen(grid_pen)
        for c in range(config.GRID_COLS + 1):
            x = c * config.GRID_SIZE
            painter.drawLine(QPointF(x, 0.0), QPointF(x, config.FLOOR_H))
        for r in range(config.GRID_ROWS + 1):
            y = r * config.GRID_SIZE
            painter.drawLine(QPointF(0.0, y), QPointF(config.FLOOR_W, y))

        #Obstacles (blockierte Zellen)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(90, 90, 90)))
        for (col, row) in config.OBSTACLES:
            rx = col * config.GRID_SIZE
            ry = row * config.GRID_SIZE
            painter.drawRect(QRectF(rx, ry, config.GRID_SIZE, config.GRID_SIZE))

        #Eingang (ENTRY_CELL) Markierung
        #print("ENTRY_CELL:", config.ENTRY_CELL[0], config.ENTRY_CELL[1])
        ex_x, ex_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.setPen(QPen(QColor(0, 160, 0)))
        painter.drawEllipse(QPointF(ex_x, ex_y), 0.1, 0.1)
        #print(ex_x, ex_y)
        
        #Ausgang (EXIT_CELL) Markierung
        #print("EXIT_CELL:", config.EXIT_CELL[0], config.EXIT_CELL[1])
        ox, oy = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.setPen(QPen(QColor(160, 0, 0)))
        painter.drawEllipse(QPointF(ox, oy), 0.1, 0.1)
        #print(ox, oy)

        if self.layout_data:
            #Maschinen (rotierte Rechtecke)
            for m in self.layout_data:
                cx = float(m["x"])
                cy = float(m["y"])
                rot = int(m.get("z", 0))
                w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
                h_m = float(m["h_cells"]) * float(config.GRID_SIZE)

                painter.save()
                t = QTransform()
                t.translate(cx, cy)  # Maschine um Mittelpunkt platzieren
                t.rotate(rot)        # Maschine rotieren
                painter.setTransform(t, True)

                rect = QRectF(-w_m / 2, -h_m / 2, w_m, h_m)

                # ========================= Laufweg-Rand (transparent grün) ==========================
                # Muss zur Kollisionslogik passen: pad_cells = ceil((clearance_m/2) / GRID_SIZE)

                clearance_m = float(getattr(config, "MACHINE_CLEARANCE_M", 0.0) or 0.0)
                if clearance_m > 0.0:
                    pad_cells = int(math.ceil((clearance_m / 2.0) / float(config.GRID_SIZE)))
                    pad_m = pad_cells * float(config.GRID_SIZE)  # Visualisierung in Metern, grid-aligned

                    # Größeres Rechteck um die Maschine (gleicher Mittelpunkt), rotiert mit Maschine
                    clearance_rect = QRectF(
                        rect.x() - pad_m,
                        rect.y() - pad_m,
                        rect.width() + 2.0 * pad_m,
                        rect.height() + 2.0 * pad_m,
                    )

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(0, 255, 0, 60)))  # RGBA: transparentes Grün
                    painter.drawRect(clearance_rect)

                # ====================== Maschine selbst (wie vorher) ==============================
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(172,213,230)))
                painter.drawRect(rect)

                idx = int(m["idx"])
                w_size, d_size = config.MACHINE_SIZES[idx]
                w_draw = float(m["w_cells"]) * float(config.GRID_SIZE)
                h_draw = float(m["h_cells"]) * float(config.GRID_SIZE)

                #Maschinenlabel sammeln (wird später "nicht skaliert" gezeichnet)
                raw_idx = m.get("idx", None)
                if raw_idx is not None:
                    try:
                        label = str(int(raw_idx) + 1)  # 1-basiert anzeigen
                    except Exception:
                        label = str(raw_idx)
                    labels.append((float(m["x"]), float(m["y"]), label))
                painter.restore()

                # Worker-Punkt in WELT-Koordinaten zeichnen (machine_worker_point liefert Weltkoords)
                work_r = 0.2
                work_pen = QPen(QColor(255, 255, 0))
                work_pen.setWidthF(1.0 / sx)
                painter.setPen(work_pen)
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                x, y = machine_worker_point(m)
                painter.drawEllipse(QPointF(x, y), work_r, work_r)

        if self.layout_data:    
            #===================================================================================
            #======================Hier Pfeile zwischen maschinen zeichnen(Worker)==============
            w_edges = getattr(config, "WORKER_CONNECTIONS", [])
            q = len(self.layout_data)
            #print(q , "Maschinen im Layout")
            for a, b in w_edges:
                #print("Worker-Edge from", a, "to", b)
                if not (0 <= int(a) < q):
                        #print(a, "falsch")
                        continue
                if not (0 <= int(b) < q):
                        continue
                a_x, a_y = machine_worker_point(self.layout_data[int(a)])
                b_x, b_y = machine_worker_point(self.layout_data[int(b)])
                print("  from", a_x, a_y, "to", b_x, b_y)
                pen = QPen(QColor(255, 165, 0))  # Orange
                pen.setWidthF(1.5 / sx)  # dünn wie PPT, unabhängig von Zoom
                painter.setPen(pen)
                painter.drawLine(QPointF(a_x, a_y), QPointF(b_x, b_y))

            #===================================================================================
            #======================Hier Pfeile zwischen maschinen zeichnen======================

            edges = list(getattr(config, "MATERIAL_CONNECTIONS", []))
            n = len(self.layout_data)

            entry_world = cell_center_from_topleft(int(config.ENTRY_CELL[0]), int(config.ENTRY_CELL[1]), 1, 1)
            exit_world  = cell_center_from_topleft(int(config.EXIT_CELL[0]),  int(config.EXIT_CELL[1]),  1, 1)

            pen = QPen(QColor(30, 30, 30))
            pen.setWidthF(1.5 / sx)  # dünn wie PPT, unabhängig von Zoom
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            #print("======================= Zeichne Materialflüsse ======================")
            for a, b in edges:
                a_idx = None if a is None else int(a)
                b_idx = None if b is None else int(b)
                print("Edge from", a_idx, "to", b_idx)
                # --- start point + direction ---
                if a_idx is None:
                    x1, y1 = entry_world
                    # Richtung aus Ziel ableiten (dominante Achse)
                    if b_idx is None:
                        tx, ty = exit_world
                    else:
                        tx, ty = machine_input_point(self.layout_data[b_idx])
                    side1 = "right" if (tx - x1) >= 0 else "left" if abs(tx - x1) >= abs(ty - y1) else ("bottom" if (ty - y1) >= 0 else "top")
                    d1 = _dir_vec(side1)
                else:
                    if not (0 <= a_idx < n):
                        continue
                    ma = self.layout_data[a_idx]
                    cx, cy = float(ma["x"]), float(ma["y"])
                    x1, y1 = machine_output_point(ma)
                    side1 = _side_world_from_center(cx, cy, x1, y1)
                    d1 = _dir_vec(side1)

                # --- end point + direction (ray goes outward from port) ---
                if b_idx is None:
                    x2, y2 = exit_world
                    # Richtung aus Start ableiten (dominante Achse)
                    side2 = "left" if (x1 - x2) >= 0 else "right" if abs(x1 - x2) >= abs(y1 - y2) else ("top" if (y1 - y2) >= 0 else "bottom")
                    d2 = _dir_vec(side2)
                else:
                    if not (0 <= b_idx < n):
                        continue
                    mb = self.layout_data[b_idx]
                    cx, cy = float(mb["x"]), float(mb["y"])
                    x2, y2 = machine_input_point(mb)
                    side2 = _side_world_from_center(cx, cy, x2, y2)
                    d2 = _dir_vec(side2)

                pts = _orth_polyline_points(x1, y1, x2, y2, d1, d2)

                # draw segments
                for i in range(len(pts) - 1):
                    xa, ya = pts[i]
                    xb, yb = pts[i + 1]
                    painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))

                # arrowhead on last segment (optional)
                if len(pts) >= 2:
                    xa, ya = pts[-2]
                    xb, yb = pts[-1]
                    _draw_arrowhead(painter, xa, ya, xb, yb, length=0.25)
                #===================================================================================
                
                #Ports zeichnen (kleine Kreise)
                port_r = 0.12
                port_pen = QPen(QColor(0, 0, 0))
                port_pen.setWidthF(1.0 / sx)
                painter.setPen(port_pen)

                painter.setBrush(QBrush(QColor(0, 200, 0)))  # Input grün
                for m in self.layout_data:
                    x, y = machine_input_point(m)
                    painter.drawEllipse(QPointF(x, y), port_r, port_r)

                painter.setBrush(QBrush(QColor(220, 0, 0)))  # Output rot
                for m in self.layout_data:
                    x, y = machine_output_point(m)
                    painter.drawEllipse(QPointF(x, y), port_r, port_r)

        #Welt-Transform zurück (ab hier Pixel/Device-Koordinaten)
        painter.restore()

        #Maschinenlabels in Device-Koordinaten (gleich groß lesbar)
        if labels:
            painter.setPen(QPen(QColor(0, 0, 0)))
            font = QFont()
            font.setPixelSize(14)
            painter.setFont(font)

            for xw, yw, text in labels:
                px = xw * sx
                py = yw * sy
                r = QRectF(px - 18, py - 10, 36, 20)
                painter.drawText(r, int(Qt.AlignmentFlag.AlignCenter), text)


class PopulationCanvas(QWidget):
    """Übersicht: viele Individuen als Miniaturen (ohne Flows, ohne Entry/Exit)."""

    def __init__(self, parent=None, cols=10, rows=1):
        super().__init__(parent)
        self.population = None
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(900, 140)

    def set_population(self, population):
        """Setzt die aktuelle GA-Population (Liste von Individuen)."""
        self.population = population
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        #Widget-Hintergrund
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        #Keine Population -> nichts zeichnen
        if not self.population:
            return

        #Mini-Grid: Aufteilung des Widgets in Zellen (cols x rows)
        W = self.width()
        H = self.height()
        cell_w = W / self.cols
        cell_h = H / self.rows

        for idx, ind in enumerate(self.population):
            #Position der Miniatur im Grid
            col = idx % self.cols
            row = idx // self.cols
            if row >= self.rows:
                break

            cell_x = col * cell_w
            cell_y = row * cell_h

            painter.save()

            #Rahmen/Hintergrund pro Miniatur
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.setBrush(QBrush(QColor(250, 250, 250)))
            painter.drawRect(QRectF(cell_x + 2, cell_y + 2, cell_w - 4, cell_h - 4))

            #Miniatur-Transform: Mini-Fenster -> Weltmaßstab
            margin = 6.0
            avail_w = max(1.0, cell_w - margin * 2)
            avail_h = max(1.0, cell_h - margin * 2)
            scale = min(avail_w / config.FLOOR_W, avail_h / config.FLOOR_H)

            painter.translate(cell_x + margin, cell_y + margin)
            painter.scale(scale, scale)

            #Floor in der Miniatur
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(245, 245, 245)))
            painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

            #Obstacles in der Miniatur
            painter.setBrush(QBrush(QColor(90, 90, 90)))
            for (col_o, row_o) in config.OBSTACLES:
                ox = col_o * config.GRID_SIZE
                oy = row_o * config.GRID_SIZE
                painter.drawRect(QRectF(ox, oy, config.GRID_SIZE, config.GRID_SIZE))

            #Maschinen in der Miniatur (ohne Flows, ohne Labels)
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
                painter.setBrush(QBrush(QColor(172,213,230)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(-w_m / 2, -h_m / 2, w_m, h_m))
                painter.restore()

            painter.restore()


class BestDialog(QDialog):
    """Dialog: zeigt die beste Lösung (LayoutCanvas) im großen Format."""

    def __init__(self, layout_data, parent=None, title="Beste Lösung"):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self.canvas = LayoutCanvas(self)
        self.canvas.set_layout(layout_data)
        layout.addWidget(self.canvas)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
