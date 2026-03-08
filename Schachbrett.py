from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtGui import QPainter, QPen, QBrush
from PyQt6.QtCore import Qt, QRect, QPointF
import sys


class BoardWidget(QWidget):
    def __init__(self, rows=8, cols=8, cell_size=60, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # Liste der markierten Felder (als (row, col)-Tupel)
        self.selected_cells = []

        # Größe des Widgets einstellen
        self.setFixedSize(self.cols * self.cell_size, self.rows * self.cell_size)

    def paintEvent(self, event):
        """
        Wird automatisch aufgerufen, wenn das Widget neu gezeichnet wird.
        Hier zeichnen wir:
          - das Schachbrett
          - die markierten Punkte
          - die Linien zwischen den Punkten
        """
        painter = QPainter(self)

        # 1. Schachbrett zeichnen
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = QRect(x, y, self.cell_size, self.cell_size)

                # Hell/dunkel abhängig von (row + col)
                if (row + col) % 2 == 0:
                    painter.setBrush(QBrush(Qt.GlobalColor.lightGray))
                else:
                    painter.setBrush(QBrush(Qt.GlobalColor.darkGray))

                painter.setPen(Qt.PenStyle.NoPen)  # kein Rand um das Feld
                painter.drawRect(rect)

        # 2. Markierte Felder als Punkte zeichnen
        painter.setPen(Qt.GlobalColor.black)
        painter.setBrush(QBrush(Qt.GlobalColor.green))
        radius = self.cell_size * 0.2  # Größe des Punkts (20% der Feldgröße)

        # Wir speichern auch die Zentren der Punkte für die Linien
        centers = []

        for (row, col) in self.selected_cells:
            center_x = col * self.cell_size + self.cell_size / 2
            center_y = row * self.cell_size + self.cell_size / 2
            center = QPointF(center_x, center_y)
            centers.append(center)

            # Kreis zeichnen (Ellipse mit gleichem Radius in x- und y-Richtung)
            painter.drawEllipse(center, radius, radius)

        # 3. Linien zwischen den markierten Feldern zeichnen
        if len(centers) >= 2:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(Qt.GlobalColor.red, 3))  # rote Linie, Dicke 3

            # Zwischen jeweils aufeinanderfolgenden Punkten eine Linie ziehen
            for i in range(len(centers) - 1):
                painter.drawLine(centers[i], centers[i + 1])

    def mousePressEvent(self, event):
        """
        Wird aufgerufen, wenn in das Widget geklickt wird.
        Wir bestimmen, welches Feld angeklickt wurde, und merken es uns.
        """
        pos = event.position()  # QPointF
        x = int(pos.x())
        y = int(pos.y())

        # Spalte und Zeile berechnen
        col = x // self.cell_size
        row = y // self.cell_size

        # Nur hinzufügen, wenn innerhalb des Bretts
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.selected_cells.append((row, col))
            # Neu zeichnen
            self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Schachbrett mit Markierungen (PyQt6)")

        # Unser BoardWidget als zentrales Widget
        board = BoardWidget(rows=8, cols=8, cell_size=60)
        self.setCentralWidget(board)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
