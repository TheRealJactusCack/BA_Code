Fabrikplaner GA (Excel-Import)

Start:
  python3 main_refactored.py

Workflow:
  1) In der UI 'Excel laden…' wählen
  2) Sheet auswählen (falls mehrere)
  3) GA starten

Excel Format:
  - Header-Zeile: id | type | ...
  - Zeilen:
      type=wall:   x1 y1 x2 y2
      type=column: x y w d rot
      type=machine: w d input
  - Optional Meta (oben in A/B): grid_size, floor_w, floor_h, flip_y, entry_*, exit_*
