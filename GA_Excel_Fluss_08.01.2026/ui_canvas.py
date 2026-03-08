# FILE: ui_canvas.py

from __future__ import annotations

import math
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QDialogButtonBox
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QTransform, QPolygonF, QFont
from PyQt6.QtCore import Qt, QPointF, QRectF

import config
from helpers import cell_center_from_topleft, route_all_flows


def _draw_arrowhead(
    painter: QPainter,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    length: float = 0.30,  # Welt-Einheiten (Meter)
    angle_deg: float = 28.0,
) -> None:
    """Pfeilspitze am Ende der Linie (x1,y1)->(x2,y2)."""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return

    theta = math.atan2(dy, dx)
    ang = math.radians(float(angle_deg))

    #Flügelpunkte relativ zur Spitze (x2,y2)
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
        print("ENTRY_CELL:", config.ENTRY_CELL[0], config.ENTRY_CELL[1])
        ex_x, ex_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.setPen(QPen(QColor(0, 160, 0)))
        painter.drawEllipse(QPointF(ex_x, ex_y), 0.1, 0.1)
        print(ex_x, ex_y)
        
        #Ausgang (EXIT_CELL) Markierung
        print("EXIT_CELL:", config.EXIT_CELL[0], config.EXIT_CELL[1])
        ox, oy = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.setPen(QPen(QColor(160, 0, 0)))
        painter.drawEllipse(QPointF(ox, oy), 0.1, 0.1)
        print(ox, oy)

        #Maschinenfarben nach Rotation
        rotation_color = {
            0: QColor(255, 255, 0),     # Gelb
            90: QColor(0, 0, 255),      # Blau
            180: QColor(0, 200, 0),     # Grün
            270: QColor(255, 0, 0),     # Rot
        }

        if self.layout_data:
            #Maschinen (rotierte Rechtecke)
            for m in self.layout_data:
                cx = float(m["x"])
                cy = float(m["y"])
                rot = int(m.get("z", 0))
                w_m = float(m["w_cells"]) * float(config.GRID_SIZE)
                h_m = float(m["h_cells"]) * float(config.GRID_SIZE)
                color = rotation_color.get(rot, rotation_color[0])

                painter.save()
                t = QTransform()
                t.translate(cx, cy)  # Maschine um Mittelpunkt platzieren
                t.rotate(rot)        # Maschine rotieren
                painter.setTransform(t, True)

                rect = QRectF(-w_m / 2, -h_m / 2, w_m, h_m)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(rect)

                #Maschinenlabel sammeln (wird später "nicht skaliert" gezeichnet)
                raw_idx = m.get("idx", None)
                if raw_idx is not None:
                    try:
                        label = str(int(raw_idx) + 1)  # 1-basiert anzeigen
                    except Exception:
                        label = str(raw_idx)
                    labels.append((float(m["x"]), float(m["y"]), label))

                painter.restore()

            #Materialflüsse (Routen) berechnen + zeichnen
            flows, _ = route_all_flows(self.layout_data)

            flow_pen = QPen(QColor(200, 30, 30))  # Flow-Linie
            flow_pen.setWidthF(float(config.FLOW_WIDTH_M))  # Linienstärke in Weltmaß (Meter)
            painter.setPen(flow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for f in flows:
                world = f.get("world")
                if not world or len(world) < 2:
                    continue

                #Flow-Polyline (Segment-für-Segment)
                for i in range(len(world) - 1):
                    x1, y1 = world[i]
                    x2, y2 = world[i + 1]
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

                #Flow-Richtung (Pfeil am Ende: world[0] -> world[-1])
                x1, y1 = world[-2]
                x2, y2 = world[-1]
                _draw_arrowhead(
                    painter,
                    x1, y1, x2, y2,
                    length=max(0.25, float(config.FLOW_WIDTH_M) * 2.0),
                )

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

    def __init__(self, parent=None, cols=10, rows=5):
        super().__init__(parent)
        self.population = None
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(900, 480)

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

        #Maschinenfarben nach Rotation
        rotation_color = {
            0: QColor(255, 255, 0),     # Gelb
            90: QColor(0, 0, 255),      # Blau
            180: QColor(0, 200, 0),     # Grün
            270: QColor(255, 0, 0),     # Rot
        }

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
                color = rotation_color.get(rot, rotation_color[0])

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)
                painter.setBrush(QBrush(color))
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
