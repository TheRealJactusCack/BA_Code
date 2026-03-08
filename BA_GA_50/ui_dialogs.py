# -------------------------
# UI DIALOGS MODULE
# -------------------------
# Enthält alle Dialog-Klassen für die Benutzerinteraktion

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QDialogButtonBox,
    QMessageBox, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
import config


class SizesDialog(QDialog):
    """Dialog zum Bearbeiten der Maschinen-Größen (in Metern)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Maschinengrößen bearbeiten (Meter)")
        self.layout = QVBoxLayout()
        self.form = QGridLayout()
        self.spin_w = []
        self.spin_h = []

        for i in range(config.MACHINE_COUNT):
            w_spin = QDoubleSpinBox()
            w_spin.setRange(0.1, 50.0)
            w_spin.setSingleStep(0.25)
            w_spin.setValue(config.MACHINE_SIZES[i][0] if i < len(config.MACHINE_SIZES) else 1.0)
            h_spin = QDoubleSpinBox()
            h_spin.setRange(0.1, 50.0)
            h_spin.setSingleStep(0.25)
            h_spin.setValue(config.MACHINE_SIZES[i][1] if i < len(config.MACHINE_SIZES) else 1.0)
            self.spin_w.append(w_spin)
            self.spin_h.append(h_spin)
            self.form.addWidget(QLabel(f"M{i+1} Breite (m)"), i, 0)
            self.form.addWidget(w_spin, i, 1)
            self.form.addWidget(QLabel("Höhe (m)"), i, 2)
            self.form.addWidget(h_spin, i, 3)

        self.layout.addLayout(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)
        self.setLayout(self.layout)

    def get_sizes(self):
        """Gibt die bearbeiteten Maschinengrößen in Metern zurück."""
        return [(float(s_w.value()), float(s_h.value())) for s_w, s_h in zip(self.spin_w, self.spin_h)]


class SettingsDialog(QDialog):
    """Dialog zur Einstellung der Floor-Größe."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        layout = QFormLayout()
        self.floor_w = QDoubleSpinBox()
        self.floor_w.setRange(1.0, 1000.0)
        self.floor_w.setValue(config.FLOOR_W)
        self.floor_h = QDoubleSpinBox()
        self.floor_h.setRange(1.0, 1000.0)
        self.floor_h.setValue(config.FLOOR_H)
        layout.addRow("Floor Breite (m):", self.floor_w)
        layout.addRow("Floor Höhe (m):", self.floor_h)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        """Gibt neue Floor-Größe zurück."""
        return float(self.floor_w.value()), float(self.floor_h.value())


class EditorCanvas(QWidget):
    """Canvas für den Factory Editor: zeichnet Raster und Hindernisse."""
    
    def __init__(self, cols, rows, parent=None):
        super().__init__(parent)
        self.cols = cols
        self.rows = rows
        self.cell_size = 24
        self.setMinimumSize(max(200, self.cols * self.cell_size), max(200, self.rows * self.cell_size))
        self.setMouseTracking(True)
        self.drawing = False
        self.draw_mode = True  # True = setzen, False = löschen

    def set_grid(self, cols, rows):
        """Ändert die Raster-Größe."""
        self.cols = cols
        self.rows = rows
        self.setMinimumSize(max(200, self.cols * self.cell_size), max(200, self.rows * self.cell_size))
        self.update()

    def paintEvent(self, event):
        """Zeichnet Raster und Hindernisse."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        w = self.cols * self.cell_size
        h = self.rows * self.cell_size
        
        # draw grid
        pen = QPen(QColor(200, 200, 200))
        painter.setPen(pen)
        for c in range(self.cols + 1):
            x = c * self.cell_size
            painter.drawLine(x, 0, x, h)
        for r in range(self.rows + 1):
            y = r * self.cell_size
            painter.drawLine(0, y, w, y)

        # draw obstacles
        painter.setBrush(QBrush(QColor(80, 80, 80)))
        painter.setPen(Qt.PenStyle.NoPen)
        for (col, row) in config.OBSTACLES:
            if 0 <= col < self.cols and 0 <= row < self.rows:
                painter.drawRect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)

    def mousePressEvent(self, event):
        """Startet das Malen/Löschen von Hindernissen."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.draw_mode = True
            self._toggle_cell(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.drawing = True
            self.draw_mode = False
            self._toggle_cell(event)

    def mouseMoveEvent(self, event):
        """Setzt/Löscht Zellen während Maus bewegt wird."""
        if self.drawing:
            self._toggle_cell(event)

    def mouseReleaseEvent(self, event):
        """Beendet das Malen/Löschen."""
        self.drawing = False

    def _toggle_cell(self, event):
        """Hilfsfunktion zum Setzen/Löschen einer Zelle."""
        x = event.position().x() if hasattr(event, 'position') else event.x()
        y = event.position().y() if hasattr(event, 'position') else event.y()
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
            return
        if self.draw_mode:
            config.OBSTACLES.add((col, row))
        else:
            if (col, row) in config.OBSTACLES:
                config.OBSTACLES.remove((col, row))
        self.update()


class FactoryEditorDialog(QDialog):
    """Dialog, um Floor-Größe zu setzen und Hindernisse (Säulen) zu malen."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Factory Editor")
        self.layout = QVBoxLayout()

        # Floor size controls
        form = QGridLayout()
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, 1000)
        self.w_spin.setValue(config.GRID_COLS)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, 1000)
        self.h_spin.setValue(config.GRID_ROWS)
        form.addWidget(QLabel("Grid Cols (m):"), 0, 0)
        form.addWidget(self.w_spin, 0, 1)
        form.addWidget(QLabel("Grid Rows (m):"), 1, 0)
        form.addWidget(self.h_spin, 1, 1)
        self.layout.addLayout(form)

        # Editor canvas
        self.canvas = EditorCanvas(self.w_spin.value(), self.h_spin.value(), parent=self)
        self.layout.addWidget(self.canvas)

        # Buttons
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Alles löschen")
        clear_btn.clicked.connect(self.clear_obstacles)
        btn_layout.addWidget(clear_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        self.layout.addLayout(btn_layout)

        # Update canvas when sizes change
        self.w_spin.valueChanged.connect(self.on_size_changed)
        self.h_spin.valueChanged.connect(self.on_size_changed)

        self.setLayout(self.layout)

    def on_size_changed(self):
        """Aktualisiert Canvas-Größe wenn Spin-Boxen geändert werden."""
        cols = int(self.w_spin.value())
        rows = int(self.h_spin.value())
        self.canvas.set_grid(cols, rows)

    def clear_obstacles(self):
        """Löscht alle Hindernisse."""
        config.OBSTACLES.clear()
        self.canvas.update()

    def get_values(self):
        """Gibt neue Raster-Größe und Hindernisse zurück."""
        return int(self.w_spin.value()), int(self.h_spin.value()), set(config.OBSTACLES)
