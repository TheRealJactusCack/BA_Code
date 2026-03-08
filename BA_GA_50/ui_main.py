# -------------------------
# UI MAIN MODULE
# -------------------------
# Enthält das Hauptfenster (MainWindow)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt
import config
from helpers import update_grid_counts, rebuild_machine_sizes
from ga_engine import GAEngine, run_ga
from ui_dialogs import SizesDialog, SettingsDialog, FactoryEditorDialog
from ui_canvas import PopulationCanvas, BestDialog


class MainWindow(QWidget):
    """Hauptfenster der Anwendung."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fabrikplaner GA")
        self.setMinimumSize(1024, 768)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        print(config.GRID_COLS, config.GRID_ROWS)
        # --- Steuerungsbereich (oben)
        controls_layout = QHBoxLayout()
        self.layout.addLayout(controls_layout)

        # --- Linke Seite: Maschinenanzahl und -größen
        machine_group = QGroupBox("Maschinen")
        controls_layout.addWidget(machine_group)
        machine_layout = QVBoxLayout()
        machine_group.setLayout(machine_layout)

        self.machine_count_spin = QSpinBox()
        self.machine_count_spin.setRange(1, 100)
        self.machine_count_spin.setValue(config.MACHINE_COUNT)
        self.machine_count_spin.valueChanged.connect(self.on_machine_count_changed)
        machine_layout.addWidget(QLabel("Anzahl Maschinen:"))
        machine_layout.addWidget(self.machine_count_spin)

        self.size_button = QPushButton("Größen bearbeiten")
        self.size_button.clicked.connect(self.edit_machine_sizes)
        machine_layout.addWidget(self.size_button)
        
        self.factory_button = QPushButton("Factory Editor")
        self.factory_button.clicked.connect(self.open_factory_editor)
        machine_layout.addWidget(self.factory_button)

        # Entry / Exit Controls
        entry_group = QGroupBox("Entry / Exit (Zellen)")
        controls_layout.addWidget(entry_group)
        entry_layout = QVBoxLayout()
        entry_group.setLayout(entry_layout)

        from PyQt6.QtWidgets import QGridLayout
        entry_grid = QGridLayout()
        entry_layout.addLayout(entry_grid)

        self.entry_col = QSpinBox()
        self.entry_col.setRange(0, max(0, config.GRID_COLS - 1))
        self.entry_col.setValue(config.ENTRY_CELL[0])
        self.entry_row = QSpinBox()
        self.entry_row.setRange(0, max(0, config.GRID_ROWS - 1))
        self.entry_row.setValue(config.ENTRY_CELL[1])
        self.exit_col = QSpinBox()
        self.exit_col.setRange(0, max(0, config.GRID_COLS - 1))
        self.exit_col.setValue(config.EXIT_CELL[0])
        self.exit_row = QSpinBox()
        self.exit_row.setRange(0, max(0, config.GRID_ROWS - 1))
        self.exit_row.setValue(config.EXIT_CELL[1])

        entry_grid.addWidget(QLabel("Entry Col"), 0, 0)
        entry_grid.addWidget(self.entry_col, 0, 1)
        entry_grid.addWidget(QLabel("Entry Row"), 0, 2)
        entry_grid.addWidget(self.entry_row, 0, 3)
        entry_grid.addWidget(QLabel("Exit Col"), 1, 0)
        entry_grid.addWidget(self.exit_col, 1, 1)
        entry_grid.addWidget(QLabel("Exit Row"), 1, 2)
        entry_grid.addWidget(self.exit_row, 1, 3)

        self.entry_col.valueChanged.connect(self.on_entry_exit_changed)
        self.entry_row.valueChanged.connect(self.on_entry_exit_changed)
        self.exit_col.valueChanged.connect(self.on_entry_exit_changed)
        self.exit_row.valueChanged.connect(self.on_entry_exit_changed)

        # --- Rechte Seite: GA-Parameter
        ga_group = QGroupBox("Genetischer Algorithmus")
        controls_layout.addWidget(ga_group)
        ga_layout = QVBoxLayout()
        ga_group.setLayout(ga_layout)

        self.population_size_spin = QSpinBox()
        self.population_size_spin.setRange(1, 1000)
        self.population_size_spin.setValue(config.POPULATION_SIZE)
        ga_layout.addWidget(QLabel("Population Größe:"))
        ga_layout.addWidget(self.population_size_spin)

        self.elite_keep_spin = QSpinBox()
        self.elite_keep_spin.setRange(1, 100)
        self.elite_keep_spin.setValue(config.ELITE_KEEP)
        ga_layout.addWidget(QLabel("Eliten behalten:"))
        ga_layout.addWidget(self.elite_keep_spin)

        self.mutation_prob_spin = QDoubleSpinBox()
        self.mutation_prob_spin.setRange(0.0, 1.0)
        self.mutation_prob_spin.setValue(config.BASE_MUTATION_PROB)
        self.mutation_prob_spin.setSingleStep(0.01)
        ga_layout.addWidget(QLabel("Mutationswahrscheinlichkeit:"))
        ga_layout.addWidget(self.mutation_prob_spin)

        self.mutation_pos_std_spin = QDoubleSpinBox()
        self.mutation_pos_std_spin.setRange(0.1, 10.0)
        self.mutation_pos_std_spin.setValue(config.BASE_MUTATION_POS_STD)
        self.mutation_pos_std_spin.setSingleStep(0.1)
        ga_layout.addWidget(QLabel("Positions-Mutations-StdAbw.:"))
        ga_layout.addWidget(self.mutation_pos_std_spin)

        self.mutation_angle_std_spin = QDoubleSpinBox()
        self.mutation_angle_std_spin.setRange(0.1, 10.0)
        self.mutation_angle_std_spin.setValue(config.BASE_MUTATION_ANGLE_STD)
        self.mutation_angle_std_spin.setSingleStep(0.1)
        ga_layout.addWidget(QLabel("Rotations-Mutations-StdAbw.:"))
        ga_layout.addWidget(self.mutation_angle_std_spin)

        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(1, 10000)
        self.generations_spin.setValue(200)
        ga_layout.addWidget(QLabel("Generationen (Total):"))
        ga_layout.addWidget(self.generations_spin)

        self.start_btn = QPushButton("GA starten (vollständig)")
        self.start_btn.clicked.connect(self.start_ga)
        ga_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Abbrechen")
        self.stop_btn.setEnabled(False)
        ga_layout.addWidget(self.stop_btn)

        # --- Generations-per-click und Next-Button
        adv_layout = QHBoxLayout()
        ga_layout.addLayout(adv_layout)
        self.advance_spin = QSpinBox()
        self.advance_spin.setRange(1, 1000)
        self.advance_spin.setValue(1)
        adv_layout.addWidget(QLabel("Generationen / Klick:"))
        adv_layout.addWidget(self.advance_spin)

        self.next_gen_btn = QPushButton("Nächste Generationen")
        self.next_gen_btn.clicked.connect(self.on_next_generation)
        adv_layout.addWidget(self.next_gen_btn)

        self.show_best_spin = QSpinBox()
        self.show_best_spin.setRange(0, 1000)
        self.show_best_spin.setValue(10)
        adv_layout.addWidget(QLabel("Bestanzeige alle N Gen.:"))
        adv_layout.addWidget(self.show_best_spin)

        # Anzeige Generationszähler
        self.gencounter_label = QLabel("0/0")
        ga_layout.addWidget(self.gencounter_label)

        # --- Zentraler Bereich: Population-Canvas
        self.pop_canvas = PopulationCanvas(self, cols=10, rows=5)
        self.layout.addWidget(self.pop_canvas)

        # --- Statusleiste (unten)
        self.status_label = QLabel("Willkommen beim Fabrikplaner!")
        self.layout.addWidget(self.status_label)

        # --- GA Engine
        self.ga_engine = None

    def closeEvent(self, event):
        """Bestätigungsdialog vor Schließung."""
        reply = QMessageBox.question(self, 'Bestätigung', 'Möchten Sie das Programm wirklich beenden?', 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

    def on_machine_count_changed(self, count):
        """Ändert die Anzahl der Maschinen."""
        config.MACHINE_COUNT = int(count)
        rebuild_machine_sizes(config.MACHINE_COUNT)
        self.status_label.setText(f"Anzahl Maschinen: {count}")

    def edit_machine_sizes(self):
        """Öffnet Dialog zum Bearbeiten der Maschinen-Größen."""
        sizes_dialog = SizesDialog(self)
        sizes_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        sizes_dialog.exec()
        new_sizes = sizes_dialog.get_sizes()
        for i, (w, h) in enumerate(new_sizes):
            config.MACHINE_SIZES[i] = (w, h)
        self.status_label.setText(f"Maschinen-Größen aktualisiert")

    def open_factory_editor(self):
        """Öffnet den Factory Editor zum Festlegen von Hindernissen."""
        dialog = FactoryEditorDialog(self)
        dialog.w_spin.setValue(config.GRID_COLS)
        dialog.h_spin.setValue(config.GRID_ROWS)
        dialog.canvas.set_grid(config.GRID_COLS, config.GRID_ROWS)
        if dialog.exec() == FactoryEditorDialog.DialogCode.Accepted:
            new_cols = int(dialog.w_spin.value())
            new_rows = int(dialog.h_spin.value())
            config.FLOOR_W = float(new_cols * config.GRID_SIZE)
            config.FLOOR_H = float(new_rows * config.GRID_SIZE)
            update_grid_counts()
            config.OBSTACLES = {c for c in config.OBSTACLES if c[0] < config.GRID_COLS and c[1] < config.GRID_ROWS}
            self.entry_col.setRange(0, max(0, config.GRID_COLS - 1))
            self.exit_col.setRange(0, max(0, config.GRID_COLS - 1))
            self.entry_row.setRange(0, max(0, config.GRID_ROWS - 1))
            self.exit_row.setRange(0, max(0, config.GRID_ROWS - 1))
            self.pop_canvas.update()
            self.status_label.setText(f"Factory aktualisiert: {config.GRID_COLS}x{config.GRID_ROWS} Zellen")

    def on_entry_exit_changed(self):
        """Aktualisiert Entry/Exit-Positionen."""
        config.ENTRY_CELL = (int(self.entry_col.value()), int(self.entry_row.value()))
        config.EXIT_CELL = (int(self.exit_col.value()), int(self.exit_row.value()))
        self.status_label.setText(f"Entry/Exit gesetzt: {config.ENTRY_CELL} -> {config.EXIT_CELL}")
        self.pop_canvas.update()

    def on_next_generation(self):
        """Führt eine oder mehrere Generationen durch (schrittweise)."""
        if self.ga_engine is None:
            total = int(self.generations_spin.value())
            self.ga_engine = GAEngine(total)
            self.pop_canvas.set_population(self.ga_engine.population)
            self.gencounter_label.setText(f"{self.ga_engine.generation}/{self.ga_engine.total_generations}")

        advance = int(self.advance_spin.value())
        show_every = int(self.show_best_spin.value())

        for _ in range(advance):
            if self.ga_engine.generation >= self.ga_engine.total_generations:
                break
            best_score, best_ind = self.ga_engine.step()
            self.pop_canvas.set_population(self.ga_engine.population)
            self.pop_canvas.update()
            self.gencounter_label.setText(f"{self.ga_engine.generation}/{self.ga_engine.total_generations}")
            self.status_label.setText(f"Gen {self.ga_engine.generation}: bester Score {self.ga_engine.best_score:.2f}")
            QApplication.processEvents()

            if show_every > 0 and (self.ga_engine.generation % show_every == 0):
                if self.ga_engine.best_ind:
                    dialog = BestDialog(self.ga_engine.best_ind, parent=self, 
                                       title=f"Beste Lösung nach Generation {self.ga_engine.generation}")
                    dialog.exec()

        if self.ga_engine.generation >= self.ga_engine.total_generations:
            if self.ga_engine.best_ind:
                dialog = BestDialog(self.ga_engine.best_ind, parent=self, 
                                   title=f"Beste Lösung (Ende Gen {self.ga_engine.generation})")
                dialog.exec()
            self.next_gen_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def start_ga(self):
        """Startet kompletten GA-Lauf."""
        config.MACHINE_COUNT = int(self.machine_count_spin.value())
        rebuild_machine_sizes(config.MACHINE_COUNT)
        config.POPULATION_SIZE = int(self.population_size_spin.value())
        config.ELITE_KEEP = int(self.elite_keep_spin.value())
        config.MUTATION_PROB = float(self.mutation_prob_spin.value())
        config.MUTATION_POS_STD = float(self.mutation_pos_std_spin.value())
        config.MUTATION_ANGLE_STD = float(self.mutation_angle_std_spin.value())
        gens = int(self.generations_spin.value())

        # UI sperren
        self.start_btn.setEnabled(False)
        self.next_gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("GA läuft...")
        QApplication.processEvents()

        self.machine_count_spin.setEnabled(False)
        self.size_button.setEnabled(False)
        self.population_size_spin.setEnabled(False)
        self.elite_keep_spin.setEnabled(False)
        self.factory_button.setEnabled(False)

        config.STOP_REQUESTED = False
        try:
            self.stop_btn.clicked.disconnect()
        except:
            pass
        self.stop_btn.clicked.connect(lambda: self._set_stop_flag())

        def _cb(generation, total_generations, best_score_cb, best_ind_cb, population=None):
            try:
                self._on_worker_progress(generation, total_generations, best_score_cb, best_ind_cb, population)
            except Exception:
                pass

        try:
            best_ind, best_score = run_ga(gens, progress_callback=_cb)
        except Exception as e:
            best_ind, best_score = None, float('inf')

        self._on_worker_finished(best_ind, best_score)

    def _set_stop_flag(self):
        """Setzt das Stop-Flag."""
        config.STOP_REQUESTED = True

    def _on_worker_progress(self, generation, total_generations, best_score, best_ind, population):
        """Aktualisiert GUI während GA läuft."""
        try:
            if population:
                self.pop_canvas.set_population(population)
            elif best_ind is not None:
                self.pop_canvas.set_population([best_ind])
            self.gencounter_label.setText(f"{generation}/{total_generations}")
            self.status_label.setText(f"Gen {generation}/{total_generations} — bester Score: {best_score:.2f}")
            QApplication.processEvents()
        except Exception:
            pass

    def _on_worker_finished(self, best_ind, best_score):
        """Zeigt Ergebnis und entsperrt GUI."""
        try:
            if best_ind:
                dialog = BestDialog(best_ind, parent=self, title="Beste Lösung (Ende)")
                dialog.exec()
            else:
                QMessageBox.information(self, 'GA beendet', f'GA abgebrochen oder kein Ergebnis. Best score: {best_score}')
        except Exception:
            pass

        # UI entsperren
        self.start_btn.setEnabled(True)
        self.next_gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.machine_count_spin.setEnabled(True)
        self.size_button.setEnabled(True)
        self.population_size_spin.setEnabled(True)
        self.elite_keep_spin.setEnabled(True)
        self.factory_button.setEnabled(True)
