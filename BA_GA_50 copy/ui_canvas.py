# -------------------------
# UI CANVAS MODULE
# -------------------------
# Enthält Canvas-Klassen für die Visualisierung von Layouts und Population

from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QDialogButtonBox
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QTransform, QPolygonF
from PyQt6.QtCore import Qt, QPointF, QRectF
import config
from helpers import (
    machine_output_point, machine_input_point, _draw_arrowhead,
    cell_center_from_topleft, occupied_cells, effective_dims
)


class LayoutCanvas(QWidget):
    """Große Ansicht eines Layouts (beste Lösung visualisieren)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_data = None
        self.setMinimumSize(800, 480)
        self.setMaximumSize(1600, 900)

    def set_layout(self, layout_data):
        """Setzt das zu visualisierende Layout."""
        self.layout_data = layout_data
        self.update()

    def paintEvent(self, event):
        """Zeichnet das Layout mit Maschinen, Hindernissen und Materialfluss."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        w = self.width()
        h = self.height()
        sx = w / config.FLOOR_W if config.FLOOR_W > 0 else 1.0
        sy = h / config.FLOOR_H if config.FLOOR_H > 0 else 1.0

        painter.save()
        painter.scale(sx, sy)

        # Floor
        pen = QPen(QColor(180, 180, 180))
        pen.setWidthF(2.0 / sx)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

        # Rasterlinien
        grid_pen = QPen(QColor(200, 200, 200))
        grid_pen.setWidthF(0.5 / sx)
        painter.setPen(grid_pen)
        for c in range(config.GRID_COLS + 1):
            x = c * config.GRID_SIZE
            painter.drawLine(QPointF(x, 0.0), QPointF(x, config.FLOOR_H))
        for r in range(config.GRID_ROWS + 1):
            y = r * config.GRID_SIZE
            painter.drawLine(QPointF(0.0, y), QPointF(config.FLOOR_W, y))

        # Hindernisse zeichnen
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(90, 90, 90)))
        for (col, row) in config.OBSTACLES:
            if 0 <= col < config.GRID_COLS and 0 <= row < config.GRID_ROWS:
                rx = col * config.GRID_SIZE
                ry = row * config.GRID_SIZE
                painter.drawRect(QRectF(rx, ry, config.GRID_SIZE, config.GRID_SIZE))

        # Entry / Exit Marker
        ex_x, ex_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(255, 200, 150)))
        painter.setPen(QPen(QColor(180, 120, 80)))
        painter.drawEllipse(QPointF(ex_x, ex_y), 0.3, 0.3)
        ox, oy = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
        painter.setBrush(QBrush(QColor(200, 255, 200)))
        painter.setPen(QPen(QColor(80, 160, 80)))
        painter.drawEllipse(QPointF(ox, oy), 0.3, 0.3)

        # Farbzuordnung nach Rotation
        rotation_color = {
            0: QColor(120, 160, 255),    # blau
            90: QColor(160, 220, 160),   # grün
            180: QColor(255, 200, 120),  # orange
            270: QColor(200, 160, 255),  # violett
        }

        # Maschinen zeichnen
        if self.layout_data:
            for idx, m in enumerate(self.layout_data):
                cx = m['x']
                cy = m['y']
                rot = int(m.get('z', 0)) % 360
                w_m = m['w_cells'] * config.GRID_SIZE
                h_m = m['h_cells'] * config.GRID_SIZE
                color = rotation_color.get(rot, rotation_color[0])

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)

                rect = QRectF(-w_m/2, -h_m/2, w_m, h_m)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(rect)

                # Text
                font = painter.font()
                font.setPointSize(1)
                painter.setFont(font)
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(idx+1))

                painter.restore()

            # Materialfluss zeichnen
            try:
                flow_pen = QPen(QColor(60, 60, 60))
                flow_pen.setWidthF(0.06 / sx)
                painter.setPen(flow_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                if len(self.layout_data) >= 1:
                    first = self.layout_data[0]
                    ex_x, ex_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
                    in_x, in_y = machine_input_point(first)
                    painter.drawLine(QPointF(ex_x, ex_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, ex_x, ex_y, in_x, in_y, length=0.6)

                for i in range(len(self.layout_data) - 1):
                    a = self.layout_data[i]
                    b = self.layout_data[i+1]
                    out_x, out_y = machine_output_point(a)
                    in_x, in_y = machine_input_point(b)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, out_x, out_y, in_x, in_y, length=0.5)

                if len(self.layout_data) >= 1:
                    last = self.layout_data[-1]
                    exit_x, exit_y = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
                    out_x, out_y = machine_output_point(last)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(exit_x, exit_y))
                    _draw_arrowhead(painter, out_x, out_y, exit_x, exit_y, length=0.6)
            except Exception:
                pass

        painter.restore()


class PopulationCanvas(QWidget):
    """Mini-Ansichten der gesamten Population (10x5 Gitter)."""
    
    def __init__(self, parent=None, cols=10, rows=5):
        super().__init__(parent)
        self.population = None
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(900, 480)
        self.show_grid = True

    def set_population(self, population):
        """Setzt die zu visualisierende Population."""
        self.population = population
        self.update()

    def set_show_grid(self, show: bool):
        """Toggled die Raster-Sichtbarkeit."""
        self.show_grid = bool(show)
        self.update()

    def paintEvent(self, event):
        """Zeichnet Mini-Ansichten aller Layouts."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        if not self.population:
            return

        W = self.width()
        H = self.height()
        cell_w = W / self.cols
        cell_h = H / self.rows

        rotation_color = {
            0: QColor(120, 160, 255),
            90: QColor(160, 220, 160),
            180: QColor(255, 200, 120),
            270: QColor(200, 160, 255),
        }

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

            # Mini-Floor
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(245, 245, 245)))
            painter.drawRect(QRectF(0.0, 0.0, float(config.FLOOR_W), float(config.FLOOR_H)))

            # Hindernisse
            painter.setBrush(QBrush(QColor(90, 90, 90)))
            painter.setPen(Qt.PenStyle.NoPen)
            for (col_o, row_o) in config.OBSTACLES:
                if 0 <= col_o < config.GRID_COLS and 0 <= row_o < config.GRID_ROWS:
                    ox = col_o * config.GRID_SIZE
                    oy = row_o * config.GRID_SIZE
                    painter.drawRect(QRectF(ox, oy, config.GRID_SIZE, config.GRID_SIZE))

            # optionales Mini-Raster
            if self.show_grid:
                grid_pen = QPen(QColor(220, 220, 220))
                grid_pen.setWidthF(0.05)
                painter.setPen(grid_pen)
                for c in range(config.GRID_COLS + 1):
                    x = c * config.GRID_SIZE
                    painter.drawLine(QPointF(x, 0.0), QPointF(x, config.FLOOR_H))
                for r in range(config.GRID_ROWS + 1):
                    y = r * config.GRID_SIZE
                    painter.drawLine(QPointF(0.0, y), QPointF(config.FLOOR_W, y))

            # Entry / Exit
            ex_x, ex_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
            ox, oy = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
            painter.setBrush(QBrush(QColor(255, 200, 150)))
            painter.setPen(QPen(QColor(180, 120, 80)))
            painter.drawEllipse(QPointF(ex_x, ex_y), 0.25, 0.25)
            painter.setBrush(QBrush(QColor(200, 255, 200)))
            painter.setPen(QPen(QColor(80, 160, 80)))
            painter.drawEllipse(QPointF(ox, oy), 0.25, 0.25)

            # Mini-Maschinen
            for m in ind:
                cx = m['x']
                cy = m['y']
                rot = int(m.get('z', 0)) % 360
                w_m = m['w_cells'] * config.GRID_SIZE
                h_m = m['h_cells'] * config.GRID_SIZE
                color = rotation_color.get(rot, rotation_color[0])

                painter.save()
                t = QTransform()
                t.translate(cx, cy)
                t.rotate(rot)
                painter.setTransform(t, True)

                rect = QRectF(-w_m / 2, -h_m / 2, w_m, h_m)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(rect)

                painter.restore()

            # Materialfluss in Mini-Ansicht
            try:
                flow_pen = QPen(QColor(60, 60, 60))
                flow_pen.setWidthF(0.04)
                painter.setPen(flow_pen)
                painter.setBrush(Qt.PenStyle.NoBrush)

                if len(ind) >= 1:
                    first = ind[0]
                    ex_x, ex_y = cell_center_from_topleft(config.ENTRY_CELL[0], config.ENTRY_CELL[1], 1, 1)
                    in_x, in_y = machine_input_point(first)
                    painter.drawLine(QPointF(ex_x, ex_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, ex_x, ex_y, in_x, in_y, length=0.5)

                for i in range(len(ind) - 1):
                    a = ind[i]
                    b = ind[i+1]
                    out_x, out_y = machine_output_point(a)
                    in_x, in_y = machine_input_point(b)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(in_x, in_y))
                    _draw_arrowhead(painter, out_x, out_y, in_x, in_y, length=0.4)

                if len(ind) >= 1:
                    last = ind[-1]
                    exit_x, exit_y = cell_center_from_topleft(config.EXIT_CELL[0], config.EXIT_CELL[1], 1, 1)
                    out_x, out_y = machine_output_point(last)
                    painter.drawLine(QPointF(out_x, out_y), QPointF(exit_x, exit_y))
                    _draw_arrowhead(painter, out_x, out_y, exit_x, exit_y, length=0.5)
            except Exception:
                pass

            painter.resetTransform()
            painter.restore()


class BestDialog(QDialog):
    """Dialog zur Anzeige der besten Lösung."""
    
    def __init__(self, layout_data, parent=None, title="Beste Lösung"):
        super().__init__(parent)
        self.setWindowTitle(title)
        v = QVBoxLayout()
        self.canvas = LayoutCanvas(self)
        self.canvas.set_layout(layout_data)
        v.addWidget(self.canvas)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        v.addWidget(buttons)
        self.setLayout(v)
        self.resize(900, 600)
