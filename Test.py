from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QDoubleSpinBox, QCheckBox,
    QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QSpinBox, QCheckBox,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QGridLayout, QMainWindow
)
from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, QRect
from PyQt6.QtGui import QPainter, QColor, QTransform, QPen, QBrush
import math
import random

#==========================================================================================================================================
#=================================Chat Code mit mir zusammen================================================================================================
#==========================================================================================================================================

class BoardWidget(QWidget):
    def __init__(self, rows, cols, cell_size, plan=None, parent=None):
        super().__init__(parent)

        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.plan = plan if plan is not None else []  # Plan-Liste speichern

        self.selected_cells = []   # (row, col)

        # Größe des Widgets festlegen
        self.setFixedSize(self.cols * cell_size, self.rows * cell_size)

    def paintEvent(self, event):
        """Zeichnet das Schachbrett, Punkte und Verbindungen."""
        painter = QPainter(self)

        # Schachbrett zeichnen===================================================
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size
                y = r * self.cell_size
                rect = QRect(x, y, self.cell_size, self.cell_size)

                if (r + c) % 2 == 0:
                    painter.setBrush(QBrush(Qt.GlobalColor.white))
                else:
                    painter.setBrush(QBrush(Qt.GlobalColor.lightGray))

                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)

        # Punkte & Linien vorbereiten — nur wenn Plan nicht leer ist
        if not self.plan or len(self.plan) == 0:
            return  # Nichts zeichnen wenn kein Layout vorhanden

        # Ab hier: self.plan ist garantiert nicht leer
        centers = []
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        #Hier wird die Plan-Liste durchgegangen und die Maschinen gezeichnet

        if not self.plan is None:

            #Hier einfügen ob Fluss beachtet wird oder nicht
            if len(self.plan[0][0]) == 8: #Wenn Durchfluss be

                for i in range(len(self.plan[0])):
                    mX = self.plan[0][i][0] * self.cell_size # X-Koordinate (Spalte)
                    mY = self.plan[0][i][1] * self.cell_size # Y-Koordinate (Reihe)
                    mW = self.plan[0][i][2] * self.cell_size # Breite der Maschine
                    mH = self.plan[0][i][3] * self.cell_size # Höhe der Maschine

                    #Maschine zeichnen
                    painter.setBrush(QBrush(Qt.GlobalColor.green))
                    painter.setPen(QPen(Qt.GlobalColor.black, 1)) #Rand um die Maschinen
                    rect = QRect(mX, mY, mW, mH)
                    painter.drawRect(rect)
                    # text in die Maschinen zeichnen
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"M{i+1}")

                    fEX = self.plan[0][i][4] * self.cell_size # Flusseingang X
                    fEY = self.plan[0][i][5] * self.cell_size # Flusseingang Y
                    fAX = self.plan[0][i][6] * self.cell_size # Flussausgang X
                    fAY = self.plan[0][i][7] * self.cell_size # Flussausgang Y
                    centers.append((fEX, fEY, fAX, fAY))
                    centerE = QPointF(fEX, fEY)
                    centerA = QPointF(fAX, fAY)
                    painter.setBrush(QBrush(Qt.GlobalColor.blue))
                    #painter.setPen(QPen(Qt.GlobalColor.black, 1)) #Rand um die Maschinen
                    painter.drawEllipse(centerE, 2, 2) #Punkt Flusseingang
                    painter.drawEllipse(centerA, 2, 2) #Punkt Flussausgang
                
                if len(centers) > 1:
                    painter.setPen(QPen(Qt.GlobalColor.red, 1))
                    for i in range(len(centers) - 1):
                        painter.drawLine(centers[i][2], centers[i][3], centers[i][2], centers[i+1][1])
                        painter.drawLine(centers[i][2], centers[i+1][1], centers[i+1][0], centers[i+1][1])

            # Schleife mit Plan-Liste aus MainWindow
            else:
                for i in range(len(self.plan[0])):
                    painter.setBrush(QBrush(Qt.GlobalColor.green))
                    #painter.setPen(QPen(Qt.GlobalColor.black, 1)) #Rand um die Maschinen
                    mX = self.plan[0][i][0] * self.cell_size # X-Koordinate (Spalte)
                    mY = self.plan[0][i][1] * self.cell_size # Y-Koordinate (Reihe)
                    centers.append((mX, mY))
                    print("test")
                    painter.drawRect(mX, mY, self.cell_size, self.cell_size)

                # Linien zeichnen (verbinde Maschinen in Reihenfolge)
                if len(centers) > 1:
                    painter.setPen(QPen(Qt.GlobalColor.red, 3))
                    for i in range(len(centers) - 1):
                        painter.drawLine(centers[i][0], centers[i][1], centers[i][0], centers[i+1][1])
                        painter.drawLine(centers[i][0], centers[i+1][1], centers[i+1][0], centers[i+1][1])


# Hier können Felder durch Mausklick ausgewählt werden die werden Plan-Liste hinzugefügt

    def mousePressEvent(self, event):
        """Berechnet das geklickte Feld und speichert es."""
        pos = event.position()
        col = int(pos.x()) // self.cell_size
        row = int(pos.y()) // self.cell_size

        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.plan[0].append((col, row))
            self.update()

#==========================================================================================================================================
#=================================Mein Code================================================================================================
#==========================================================================================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.Anzahl_Clicks = 0
        self.setWindowTitle("Optimales Fabriklayout planen")
        
        self.label = QLabel("Willkommen zum Fabriklayout-Planer!")
        self.button = QPushButton("Klick mich")
        self.button_Durchfluss = QPushButton("Durchfluss beachten")
        self.button2 = QPushButton("Layout Generieren")
        self.button_distanz = QPushButton("Manhattan Distanz Berechnen")
        self.button_Anzeigen = QPushButton("Distanz Anzeigen")
        
        # initialisiere Plan / Distanzen VOR BoardWidget!
        self.Plan = []
        self.distances = []
        self.Durchfluss = False
        
        # Übergebe Plan an BoardWidget
        self.board = BoardWidget(rows=50, cols=50, cell_size=10, plan=self.Plan)

        self.button.clicked.connect(self.button_clicked)
        self.button_Durchfluss.clicked.connect(self.Durchfluss_beachten)
        self.button2.clicked.connect(self.layout_generieren)
        self.button_distanz.clicked.connect(self.Manhattan_Distanz)
        self.button_Anzeigen.clicked.connect(self.Distanz_Anzeigen)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.button_Durchfluss)
        layout.addWidget(self.button2)
        layout.addWidget(self.button_distanz)
        layout.addWidget(self.button_Anzeigen)
        layout.addWidget(self.board)

        window = QWidget()
        window.setLayout(layout)

        self.setCentralWidget(window)
    
    def button_clicked(self):
        self.Anzahl_Clicks += 1
        self.label.setText("der Knopf wurde " + str(self.Anzahl_Clicks) + " mal geklickt")
        self.button.setText("Klick mich erneut")

    def Durchfluss_beachten(self):
        """Platzhalter-Funktion für Durchfluss-Feature."""
        if self.Durchfluss == False:
            self.label.setText("Durchfluss wird beachtet (Platzhalter)")
            self.Durchfluss = True
        else:
            self.label.setText("Durchfluss wird nicht beachtet (Platzhalter)")
            self.Durchfluss = False
        return self.Durchfluss

    def layout_generieren(self):
        """Erstellt mehrere zufällige Layouts mit Maschinenpositionen."""
        self.label.setText("Layout wird generiert")
        Facility_X = 50
        Facility_Y = 50
        Anzahl_Maschinen = 10
        Anzahl_Layouts = 10
        self.Plan = []
        for i in range(Anzahl_Layouts):
            Maschine = []
            for j in range(Anzahl_Maschinen):
                Maschine_X = random.randint(0, Facility_X-10)
                Maschine_Y = random.randint(0, Facility_Y-10)
                if self.Durchfluss:                    
                    Breite_X = random.randint(1,10)
                    Breite_Y = random.randint(1,10)
                    generate = True
                    """if len(Maschine) > 1:
                        if Maschine_X + Breite_X <= Maschine[j][0]:
                            generate = False
                            j - 1
                        elif Maschine[j][0] + Maschine[j][2] <= Maschine_X:
                            generate = False
                            j - 1
                        elif Maschine_Y + Breite_Y <= Maschine[j][1]:
                            generate = False
                            j - 1
                        elif Maschine[j][1] + Maschine[j][3] <= Maschine_Y:
                            generate = False
                            j - 1
                    elif generate:"""
                    Variante_E = random.randint(0,3)
                    if Variante_E == 0:
                        FlussE_X = Maschine_X + random.randint(0, Breite_X)
                        FlussE_Y = Maschine_Y
                    elif Variante_E == 1:
                        FlussE_X = Maschine_X + random.randint(0, Breite_X)
                        FlussE_Y = Maschine_Y + Breite_Y
                    elif Variante_E == 2:
                        FlussE_X = Maschine_X
                        FlussE_Y = Maschine_Y + random.randint(0, Breite_Y)
                    else:
                        FlussE_X = Maschine_X + Breite_X
                        FlussE_Y = Maschine_Y + random.randint(0, Breite_Y)
                    Variante_A = random.randint(0,3)
                    if Variante_A == 0:
                        FlussA_X = Maschine_X + random.randint(0, Breite_X)
                        FlussA_Y = Maschine_Y
                    elif Variante_A == 1:
                        FlussA_X = Maschine_X + random.randint(0, Breite_X)
                        FlussA_Y = Maschine_Y + Breite_Y
                    elif Variante_A == 2:
                        FlussA_X = Maschine_X
                        FlussA_Y = Maschine_Y + random.randint(0, Breite_Y)
                    else:
                        FlussA_X = Maschine_X + Breite_X
                        FlussA_Y = Maschine_Y + random.randint(0, Breite_Y)
                    Maschine.append((Maschine_X, Maschine_Y, Breite_X, Breite_Y, FlussE_X, FlussE_Y, FlussA_X, FlussA_Y))
                    print(f"Layout {i+1} Maschine {j+1} Position: ({Maschine_X}, {Maschine_Y}) Breite: ({Breite_X}, {Breite_Y}) Flusseingang: ({FlussE_X}, {FlussE_Y}) Flussausgang: ({FlussA_X}, {FlussA_Y})")

                else:
                    Maschine.append((Maschine_X, Maschine_Y))
                    print(f"Layout {i+1} Maschine {j+1} Position: ({Maschine_X}, {Maschine_Y})")

            self.Plan.append(Maschine)
        
        # Aktualisiere BoardWidget mit neuer Plan
        self.board.plan = self.Plan
        self.board.update()  # Neuzeichnen
        
        self.label.setText(f"{Anzahl_Layouts} Layouts generiert.")
        return self.Plan
    

    def Manhattan_Distanz(self):
        """Berechnet Manhattan-Distanzen für alle gespeicherten Layouts und speichert sie."""
        if not hasattr(self, 'Plan') or not self.Plan:
            self.label.setText("Kein Layout vorhanden. Bitte zuerst Layout generieren.")
            return []
        Distanzen = []
        if self.Durchfluss == False:
            for i, plan in enumerate(self.Plan):
                distanz = 0
                # für jede Maschine zum nächsten in Reihenfolge
                for j in range(len(plan) - 1):
                    x1 = plan[j][0]
                    y1 = plan[j][1]
                    x2 = plan[j+1][0]
                    y2 = plan[j+1][1]
                    distanz += abs(x1 - x2) + abs(y1 - y2)
                Distanzen.append(f"Layout {i+1} hat eine Ges. Distanz von: {distanz}")
        else:
            for i, plan in enumerate(self.Plan):
                distanz = 0
                # für jede Maschine zum nächsten in Reihenfolge (Flusseingang zu Flussausgang)
                for j in range(len(plan) - 1):
                    x1 = plan[j][6]
                    y1 = plan[j][7]
                    x2 = plan[j+1][4]
                    y2 = plan[j+1][5]
                    distanz += abs(x1 - x2) + abs(y1 - y2)
                Distanzen.append(f"Layout {i+1} hat eine Ges. Distanz von: {distanz}")
        self.distances = Distanzen
        self.label.setText("Distanzen berechnet.")
        return Distanzen

    def Distanz_Anzeigen(self):
        """Gibt die berechneten Distanzen in Konsole aus und aktualisiert Label."""
        if not hasattr(self, 'distances') or not self.distances:
            # wenn noch nicht berechnet, erst berechnen
            self.label.setText("Distanzen erst berechnen!")
            return
        for i, d in enumerate(self.distances):
            print(d)
        # zeige letzten Eintrag im Label (kurz)
        if self.distances:
            self.label.setText(self.distances[0])
        else:
            self.label.setText("Keine Distanzen verfügbar.")

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
