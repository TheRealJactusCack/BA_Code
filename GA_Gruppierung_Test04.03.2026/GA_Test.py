# FILE: GA_Test.py
"""
GA_Test.py

Vollständige Sanity-Checks für:
- Excel Import -> config-Ausrichtung (aligned lists)
- Fixed Machines (Pose stabil; Operatoren dürfen sie nicht ändern)
- Optionale Ports/Worker (keine Defaults zeichnen; helper liefert None wenn fehlt)
- Overlap/Placement sanity
- GA Operatoren (crossover/mutate/tauschen/run_ga)
- UI Canvas statisch: erkennt alle Funktionen, die Layout-Daten (gx/gy/x/y/z) schreiben und
  verlangt einen Fixed-Guard vor dem ersten Schreibzugriff.

WICHTIG:
- ui_canvas wird NICHT importiert (verhindert Circular Import über ui_main).
- mousePressEvent darf NICHT geprüft werden -> ist ausgeschlossen.

Aufruf:
    python GA_Test.py --xlsx "C:/.../layouts_with_machines.xlsx" --sheet "Test"
Optional:
    python GA_Test.py --xlsx "..." --sheet "Test" --ui-canvas "C:/.../ui_canvas.py"
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import ga_engine
from excel_import import apply_excel_layout_to_config
from helpers import (
    clearance_pad_cells,
    machine_gas_point,
    machine_other_point,
    machine_water_point,
    machine_worker_point,
    occupied_cells,
    random_individual,
)


# ----------------------------- Report -----------------------------


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


class Report:
    def __init__(self) -> None:
        self._results: List[CheckResult] = []
        self._lines: List[str] = []
        self._start_ts = time.time()

    def section(self, title: str) -> None:
        self._lines.append("")
        self._lines.append("=" * 80)
        self._lines.append(title)
        self._lines.append("=" * 80)

    def add(self, name: str, ok: bool, details: str = "") -> None:
        self._results.append(CheckResult(name=name, ok=ok, details=details))
        status = "OK" if ok else "FAIL"
        self._lines.append(f"[{status}] {name}")
        if details:
            for ln in details.strip().splitlines():
                self._lines.append(f"    {ln}")

    def add_exception(self, name: str, exc: BaseException) -> None:
        self.add(name, False, f"{type(exc).__name__}: {exc}")

    def summary(self) -> None:
        ok_count = sum(1 for r in self._results if r.ok)
        fail_count = len(self._results) - ok_count
        elapsed = time.time() - self._start_ts
        self._lines.append("")
        self._lines.append("-" * 80)
        self._lines.append(f"SUMMARY: {ok_count} OK / {fail_count} FAIL | elapsed={elapsed:.2f}s")
        self._lines.append("-" * 80)

    def write(self, path: str) -> None:
        self.summary()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines).rstrip() + "\n")


# ----------------------------- Core helpers -----------------------------


def fixed_map() -> List[Optional[Dict[str, Any]]]:
    return getattr(config, "MACHINE_FIXED", [])


def ind_by_idx(ind: List[Dict]) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for m in ind:
        out[int(m.get("idx", -1))] = m
    return out


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def assert_fixed_matches(ind: List[Dict], where: str = "", tol: float = 1e-6) -> Tuple[bool, str]:
    fx = fixed_map()
    if not fx:
        return True, "MACHINE_FIXED ist leer (keine fixed Maschinen)."

    by = ind_by_idx(ind)
    for idx, fd in enumerate(fx):
        if fd is None:
            continue
        if idx not in by:
            return False, f"[{where}] fixed idx={idx} fehlt im Individuum"

        m = by[idx]
        exp_x = float(fd["x"])
        exp_y = float(fd["y"])
        got_x = float(m.get("x"))
        got_y = float(m.get("y"))
        if not approx_equal(got_x, exp_x, tol) or not approx_equal(got_y, exp_y, tol):
            return False, (
                f"[{where}] fixed drift idx={idx}: got(x={got_x},y={got_y}) exp(x={exp_x},y={exp_y}) tol={tol}"
            )

        exp_z = fd.get("z", None)
        if exp_z is not None:
            got_z = int(m.get("z", 0))
            if got_z != int(exp_z):
                return False, f"[{where}] fixed rot drift idx={idx}: got z={got_z} exp z={int(exp_z)}"

    return True, f"[{where}] alle fixed Maschinen korrekt"


# ----------------------------- Check: Config alignment -----------------------------


def check_config_alignment(rep: Report) -> None:
    rep.section("CONFIG / ALIGNMENT")

    try:
        n = int(config.MACHINE_COUNT)
        rep.add("MACHINE_COUNT > 0", n > 0, f"MACHINE_COUNT={n}")
    except Exception as e:
        rep.add_exception("MACHINE_COUNT > 0", e)
        return

    # Fixed list must be aligned
    fx = fixed_map()
    rep.add(
        "MACHINE_FIXED length == MACHINE_COUNT",
        len(fx) == int(config.MACHINE_COUNT),
        f"len(MACHINE_FIXED)={len(fx)} MACHINE_COUNT={int(config.MACHINE_COUNT)}",
    )

    # Optional aligned lists
    list_names = ["MACHINE_PORTS", "MACHINE_WORKERS", "MACHINE_WATER", "MACHINE_GAS", "MACHINE_OTHER"]
    for name in list_names:
        v = getattr(config, name, None)
        if not isinstance(v, list):
            rep.add(f"{name} is list", False, f"type={type(v)}")
            continue
        rep.add(f"{name} length == MACHINE_COUNT", len(v) == int(config.MACHINE_COUNT), f"len={len(v)}")

        bad = []
        for i, el in enumerate(v):
            if el is None:
                continue
            if not isinstance(el, dict):
                bad.append((i, type(el)))
        rep.add(f"{name} elements are None|dict", len(bad) == 0, "" if not bad else str(bad[:10]))


def check_fixed_semantics(rep: Report) -> None:
    rep.section("FIXED MACHINES SEMANTICS")

    fx = fixed_map()
    if not fx:
        rep.add("fixed machines present", True, "Keine fixed Maschinen definiert.")
        return

    fixed_idxs = [i for i, v in enumerate(fx) if v is not None]
    rep.add("fixed indices detected", True, f"fixed_idxs={fixed_idxs}")

    bad = []
    for i in fixed_idxs:
        d = fx[i]
        if not isinstance(d, dict):
            bad.append((i, "not dict"))
            continue
        if "x" not in d or "y" not in d:
            bad.append((i, "missing x/y"))
            continue
        try:
            float(d["x"])
            float(d["y"])
        except Exception:
            bad.append((i, "x/y not numeric"))
        if d.get("z") is not None:
            try:
                z = int(d["z"])
                if z not in config.ROTATIONS:
                    bad.append((i, f"z invalid {z}"))
            except Exception:
                bad.append((i, "z not int"))
    rep.add("MACHINE_FIXED entries valid", len(bad) == 0, "" if not bad else str(bad))


# ----------------------------- Check: Optional points -----------------------------


def check_optional_points(rep: Report) -> None:
    rep.section("OPTIONAL POINTS (WORKER/WATER/GAS/OTHER)")

    ind = random_individual()
    by = ind_by_idx(ind)

    cfgs = [
        ("Worker", getattr(config, "MACHINE_WORKERS", []), machine_worker_point),
        ("Water", getattr(config, "MACHINE_WATER", []), machine_water_point),
        ("Gas", getattr(config, "MACHINE_GAS", []), machine_gas_point),
        ("Other", getattr(config, "MACHINE_OTHER", []), machine_other_point),
    ]

    for label, cfg_list, func in cfgs:
        wrong = []
        missing = []
        for idx in range(int(config.MACHINE_COUNT)):
            m = by[idx]
            has = (idx < len(cfg_list)) and bool(cfg_list[idx])
            pt = func(m)
            if not has and pt is not None:
                wrong.append(idx)
            if has and pt is None:
                missing.append(idx)

        ok = (len(wrong) == 0) and (len(missing) == 0)
        rep.add(
            f"{label}: None if missing, point if present",
            ok,
            "\n".join(
                [
                    f"wrong(drawn but missing)={wrong}" if wrong else "",
                    f"missing(should exist)={missing}" if missing else "",
                ]
            ).strip(),
        )


# ----------------------------- Check: Overlap sanity -----------------------------


def check_overlap(rep: Report) -> None:
    rep.section("PLACEMENT / OVERLAP")

    try:
        p = clearance_pad_cells()
        rep.add("clearance_pad_cells computed", p >= 0, f"pad_cells={p}")
    except Exception as e:
        rep.add_exception("clearance_pad_cells computed", e)

    try:
        ind = random_individual()
        owners: Dict[Tuple[int, int], int] = {}
        overlaps = 0
        for mi, m in enumerate(ind):
            for cell in occupied_cells(m):
                if cell in owners:
                    overlaps += 1
                else:
                    owners[cell] = mi
        rep.add("random_individual has no overlaps (incl. clearance)", overlaps == 0, f"overlap_cells={overlaps}")
    except Exception as e:
        rep.add_exception("random_individual has no overlaps (incl. clearance)", e)


# ----------------------------- Check: GA operator invariance -----------------------------


def check_ga_operator_invariance(rep: Report) -> None:
    rep.section("GA OPERATOR INVARIANCE (FIXED MUST NOT MOVE)")

    a = random_individual()
    b = random_individual()

    ok, msg = assert_fixed_matches(a, where="baseline(a)")
    rep.add("baseline fixed matches", ok, msg)
    if not ok:
        return

    try:
        child = ga_engine.uniform_crossover(copy.deepcopy(a), copy.deepcopy(b))
        ok, msg = assert_fixed_matches(child, where="after uniform_crossover")
        rep.add("uniform_crossover keeps fixed", ok, msg)
    except Exception as e:
        rep.add_exception("uniform_crossover keeps fixed", e)

    try:
        ind = copy.deepcopy(a)
        ga_engine.mutate(ind)
        ok, msg = assert_fixed_matches(ind, where="after mutate")
        rep.add("mutate keeps fixed", ok, msg)
    except Exception as e:
        rep.add_exception("mutate keeps fixed", e)

    try:
        ind = copy.deepcopy(a)
        ga_engine.tauschen(ind, 1.0)
        ok, msg = assert_fixed_matches(ind, where="after tauschen")
        rep.add("tauschen keeps fixed", ok, msg)
    except Exception as e:
        rep.add_exception("tauschen keeps fixed", e)


def check_ga_smoke(rep: Report, gens: int) -> None:
    rep.section("GA SMOKE RUN")

    try:
        best_ind, best_score = ga_engine.run_ga(int(gens), progress_callback=None)
        rep.add("run_ga returns best_ind", best_ind is not None, f"best_score={best_score}")
        if best_ind is not None:
            ok, msg = assert_fixed_matches(best_ind, where="best_ind")
            rep.add("best_ind keeps fixed", ok, msg)
    except Exception as e:
        rep.add_exception("run_ga smoke", e)


# ----------------------------- UI canvas static analysis (NO import) -----------------------------


EXCLUDED_UI_FUNCTIONS = {
    "mousePressEvent",  # explizit NICHT testen
}

GUARD_TOKENS = [
    "MACHINE_FIXED",
    "machine_fixed",
    "is_fixed",
    "fixed_list",
]

# "Write patterns" (Indikatoren, dass Funktion Layout tatsächlich ändert)
WRITE_PATTERNS = [
    r"""\["gx"\]\s*=""",
    r"""\["gy"\]\s*=""",
    r"""\["x"\]\s*=""",
    r"""\["y"\]\s*=""",
    r"""\["z"\]\s*=""",
    r"""normalize_individual\s*\(""",
    r"""self\.layout_data\[[^\]]+\]\s*=\s*""",
]

# Heuristik: Guard muss irgendwo VOR dem ersten Write kommen (early return)
RETURN_PAT = r"""return\b"""


def _find_def_blocks_all(src: str) -> Dict[str, List[str]]:
    """
    Findet ALLE def-Blöcke (inkl. Dubletten).
    Rückgabe: name -> [block1, block2, ...]
    """
    # Find all def lines
    matches = list(re.finditer(r"(^[ \t]*def[ \t]+([A-Za-z_]\w*)[ \t]*\()", src, flags=re.M))
    out: Dict[str, List[str]] = {}
    for i, m in enumerate(matches):
        start = m.start(0)
        name = m.group(2)
        end = matches[i + 1].start(0) if i + 1 < len(matches) else len(src)
        out.setdefault(name, []).append(src[start:end])
    return out


def _block_has_write(block: str) -> bool:
    return any(re.search(p, block) for p in WRITE_PATTERNS)


def _block_guard_before_first_write(block: str) -> Tuple[bool, str]:
    """
    Heuristik:
    - Bestimme Position des ersten Schreibzugriffs (gx/gy/x/y/z/normalize/layout_data assignment).
    - Prüfe davor: kommt ein Guard-Token vor UND ein 'return'.
    """
    first_write_pos = None
    first_write_pat = None
    for p in WRITE_PATTERNS:
        m = re.search(p, block)
        if m:
            if first_write_pos is None or m.start() < first_write_pos:
                first_write_pos = m.start()
                first_write_pat = p

    if first_write_pos is None:
        return True, "no writes detected"

    prefix = block[:first_write_pos]
    has_guard_token = any(tok in prefix for tok in GUARD_TOKENS)
    has_return = re.search(RETURN_PAT, prefix) is not None

    if has_guard_token and has_return:
        return True, f"guard+return before first write (pattern={first_write_pat})"

    return False, (
        f"missing guard/return before first write (pattern={first_write_pat}); "
        f"guard_token_in_prefix={has_guard_token} return_in_prefix={has_return}"
    )


def check_ui_canvas_static(rep: Report, ui_canvas_path: Path) -> None:
    rep.section("UI_CANVAS STATIC (NO IMPORT) — MOVE/SWAP/ROTATE WRITES")

    try:
        text = ui_canvas_path.read_text(encoding="utf-8")
        rep.add("ui_canvas.py readable", True, f"path={ui_canvas_path}")
    except Exception as e:
        rep.add_exception("ui_canvas.py readable", e)
        return

    defs = _find_def_blocks_all(text)

    # Evaluate all defs that write layout/machine pose, excluding explicitly forbidden names.
    checked = 0
    failed = 0

    for name, blocks in sorted(defs.items()):
        if name in EXCLUDED_UI_FUNCTIONS:
            continue

        # Only check functions that actually write to pose/layout.
        write_blocks = [b for b in blocks if _block_has_write(b)]
        if not write_blocks:
            continue

        checked += 1

        # PASS if any definition has valid guard-before-write.
        per_block = []
        ok_any = False
        for bi, b in enumerate(write_blocks, start=1):
            ok, why = _block_guard_before_first_write(b)
            per_block.append(f"def#{bi}: {why}")
            if ok:
                ok_any = True

        if not ok_any:
            failed += 1
            rep.add(
                f"ui_canvas '{name}': fixed-guard before writes",
                False,
                "\n".join(
                    [
                        f"definitions_total={len(blocks)} write_defs={len(write_blocks)}",
                        "No definition had guard+return before first write.",
                        *per_block[:6],
                        "TIP: put fixed-check + return at top of function BEFORE writing gx/gy/x/y/z.",
                    ]
                ),
            )
        else:
            # Also warn on duplicates (debug hygiene)
            warn = ""
            if len(blocks) > 1:
                warn = f" (WARN: {len(blocks)} defs total / duplicates present)"
            rep.add(
                f"ui_canvas '{name}': fixed-guard before writes",
                True,
                "\n".join(
                    [
                        f"definitions_total={len(blocks)} write_defs={len(write_blocks)}{warn}",
                        *per_block[:6],
                    ]
                ),
            )

    rep.add(
        "ui_canvas functions checked (write-affecting only, mousePressEvent excluded)",
        True,
        f"checked={checked} failed={failed}",
    )


# ----------------------------- Main -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="GA sanity tests + debug report")
    parser.add_argument("--xlsx", type=str, required=True, help="Path to Excel (.xlsx)")
    parser.add_argument("--sheet", type=str, default="Test", help="Sheet name (default: Test)")
    parser.add_argument("--gens", type=int, default=10, help="Generations for smoke GA run")
    parser.add_argument("--report", type=str, default="ga_test_report.txt", help="Output report file")
    parser.add_argument(
        "--ui-canvas",
        type=str,
        default=None,
        help="Optional path to ui_canvas.py (default: ui_canvas.py next to GA_Test.py)",
    )
    args = parser.parse_args()

    rep = Report()
    rep.section("GA_TEST START")

    # Excel import
    rep.section("EXCEL IMPORT")
    rep.add("Excel file exists", os.path.exists(args.xlsx), f"xlsx={args.xlsx}")
    if not os.path.exists(args.xlsx):
        rep.write(args.report)
        print(f"Wrote report: {args.report}")
        return 2

    try:
        apply_excel_layout_to_config(args.xlsx, args.sheet)
        rep.add("apply_excel_layout_to_config executed", True, f"sheet={args.sheet}")
    except Exception as e:
        rep.add_exception("apply_excel_layout_to_config executed", e)
        rep.write(args.report)
        print(f"Wrote report: {args.report}")
        return 1

    # Core checks
    check_config_alignment(rep)
    check_fixed_semantics(rep)
    check_optional_points(rep)
    check_overlap(rep)

    # GA checks
    check_ga_operator_invariance(rep)
    check_ga_smoke(rep, gens=args.gens)

    # UI static checks (NO import)
    ui_path = Path(args.ui_canvas) if args.ui_canvas else (Path(__file__).resolve().parent / "ui_canvas.py")
    check_ui_canvas_static(rep, ui_canvas_path=ui_path)

    rep.write(args.report)
    print(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python "C:\Users\Carey\OneDrive\Bachelorarbeit\BA-Code\GA_10.02.2026\GA_Test.py" --xlsx "C:\Users\Carey\OneDrive\Bachelorarbeit\BA-Code\layouts_with_machines.xlsx" --sheet "Test"
"""