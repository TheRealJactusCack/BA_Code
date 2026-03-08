# -------------------------
# CONFIGURATION MODULE
# -------------------------
# Enthält alle globalen Konfigurationen und Parameter

# Grundparameter für GA
POPULATION_SIZE = 50
ELITE_KEEP = 25
MACHINE_COUNT = 10

# Raumgröße (Floor) in Einheiten (Meter)
FLOOR_W = 20.0
FLOOR_H = 15.0

# Raster / Grundeinheit: 1 m Raster
GRID_SIZE = 0.25

# Maschinen-Größen: Liste mit (width_m, height_m) pro Maschine (in Metern)
# Standard: jede Maschine ist 1.0 m x 1.0 m
MACHINE_SIZES = [(1.0, 1.0) for _ in range(MACHINE_COUNT)]

# Rasterzellen (werden initial berechnet)
GRID_COLS = max(1, int(FLOOR_W // GRID_SIZE))
GRID_ROWS = max(1, int(FLOOR_H // GRID_SIZE))

# Wareneingang / -ausgang in Zellkoordinaten
ENTRY_CELL = (0, 0)
EXIT_CELL = (int(GRID_COLS) - 1, int(GRID_ROWS) - 1)

# Strafen / Gewichtungen
OVERLAP_PENALTY = 1e9
OUT_OF_BOUNDS_PENALTY = 1e7
DIST_SCALE = 1.0
OBSTACLE_PENALTY = 1e9

# Mutations-Grundwerte
BASE_MUTATION_PROB = 0.15
BASE_MUTATION_POS_STD = 1.0
BASE_MUTATION_ANGLE_STD = 1

MUTATION_PROB = BASE_MUTATION_PROB
MUTATION_POS_STD = BASE_MUTATION_POS_STD
MUTATION_ANGLE_STD = BASE_MUTATION_ANGLE_STD

# Erlaubte Rotationswerte (nur Vielfache von 90°)
ROTATIONS = [0, 90, 180, 270]

# Globaler Abbruch-Flag für GA
STOP_REQUESTED = False

# Hindernisse (Set von (col,row)-Zellen)
OBSTACLES = set()
