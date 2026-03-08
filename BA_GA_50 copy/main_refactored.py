#!/usr/bin/env python3
# ========================================================================================
# FABRIKPLANER - GENETISCHER ALGORITHMUS
# ========================================================================================
# Haupteingangspunkt für die Anwendung
# 
# Struktur:
#   - config.py: Globale Konfigurationen
#   - helpers.py: Hilfsfunktionen (Geometrie, Kollisionserkennung, zufällige Erzeugung)
#   - ga_engine.py: Genetischer Algorithmus, GAEngine, GAWorker
#   - ui_dialogs.py: Dialog-Fenster (Einstellungen, Editor, Größen)
#   - ui_canvas.py: Visualisierungs-Canvas (Layout, Population, Best-Dialog)
#   - ui_main.py: Hauptfenster (MainWindow)
#   - main.py: Einstiegspunkt (diese Datei)
# ========================================================================================

import sys
import traceback
from PyQt6.QtWidgets import QApplication

print("Test1")

def main():
    """Startet die Anwendung."""
    try:
        # Importiere MainWindow (lokales Modul im selben Ordner)
        from ui_main import MainWindow
        
        # Erzeuge QApplication
        app = QApplication(sys.argv)

        # ensure grid counts are up-to-date and print them once at startup
        from helpers import update_grid_counts
        update_grid_counts()

        # Erzeuge und zeige Hauptfenster
        win = MainWindow()
        win.show()
        
        # Starte Event-Loop
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"FEHLER beim Starten der Anwendung: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
