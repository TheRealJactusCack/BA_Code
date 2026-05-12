import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from openpyxl import Workbook

import config
from helpers import compute_routed_edges


def create_unique_name() -> tuple[str, int]:
    random_string = random.randint(1111111, 9999999)
    name = f"Layout_{random_string}.xlsx"
    return name, random_string


def _sorted_best_ind(best_ind: List[Dict]) -> List[Dict]:
    return sorted(best_ind, key=lambda machine: int(machine.get("idx", 10**9)))


def _machine_id(machine: Dict) -> str:
    idx = int(machine.get("idx", -1))
    machine_ids = getattr(config, "MACHINE_IDS", []) or []
    if 0 <= idx < len(machine_ids):
        return str(machine_ids[idx])
    if machine.get("id") not in (None, ""):
        return str(machine["id"])
    return str(idx + 1)


def _round_value(value: Optional[float], digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def add_ind_to_sheet(layout_data: Workbook, best_ind: List[Dict]) -> None:
    worksheet = layout_data.create_sheet(title="Optimales layout")
    worksheet.append(["ID", "Label", "X Position", "Y Position", "Rotation", "Breite", "Tiefe"])
    for machine in _sorted_best_ind(best_ind):
        idx = int(machine.get("idx", -1))
        label = machine.get("label", f"idx_{idx}")
        x_pos = _round_value(machine.get("x"))
        y_pos = _round_value(machine.get("y"))
        rot = int(machine.get("z", 0))
        width = ""
        depth = ""
        if 0 <= idx < len(getattr(config, "MACHINE_SIZES", []) or []):
            width, depth = config.MACHINE_SIZES[idx]
        worksheet.append([_machine_id(machine), label, x_pos, y_pos, rot, width, depth])


def _station_entry(machine: Dict) -> Dict:
    idx = int(machine.get("idx", -1))
    label = machine.get("label", f"idx_{idx}")
    width = ""
    depth = ""
    if 0 <= idx < len(getattr(config, "MACHINE_SIZES", []) or []):
        width, depth = config.MACHINE_SIZES[idx]

    return {
        "id": _machine_id(machine),
        "label": label,
        "x": _round_value(machine.get("x")),
        "y": _round_value(machine.get("y")),
        "z": int(machine.get("z", 0)),
        "w": _round_value(width) if width != "" else "",
        "d": _round_value(depth) if depth != "" else "",
    }


def _connection_id(machine_idx: Optional[int], *, input_id: str, output_id: str, best_ind: List[Dict]) -> str:
    if machine_idx is None:
        return input_id if input_id else output_id
    return _machine_id(best_ind[int(machine_idx)])


def _materialflow_entries(best_ind: List[Dict]) -> List[Dict]:
    routed_edges = compute_routed_edges(best_ind)
    input_id = str(getattr(config, "INPUT_ID", "") or "")
    output_id = str(getattr(config, "OUTPUT_ID", "") or "")
    materialfluss = []

    for edge in routed_edges.get("material", []):
        source_id = _connection_id(edge.get("a"), input_id=input_id, output_id=output_id, best_ind=best_ind)
        target_id = _connection_id(edge.get("b"), input_id=input_id, output_id=output_id, best_ind=best_ind)
        points = edge.get("pts") or []
        coordinates = [[_round_value(x), _round_value(y)] for x, y in points]
        materialfluss.append(
            {
                "verbindung": [source_id, target_id],
                "länge": _round_value(edge.get("length_m")),
                "koordinaten": coordinates,
            }
        )
    return materialfluss


def build_json_payload(best_ind: List[Dict]) -> Dict:
    ordered_best_ind = _sorted_best_ind(best_ind)
    return {
        "stationen": [_station_entry(machine) for machine in ordered_best_ind],
        "LagerIds": {
            "input": str(getattr(config, "INPUT_ID", "") or ""),
            "output": str(getattr(config, "OUTPUT_ID", "") or ""),
        },
        "materialfluss": _materialflow_entries(ordered_best_ind),
    }


def save_json(best_ind: List[Dict], json_path: Path) -> None:
    payload = build_json_payload(best_ind)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_sheet(best_ind: List[Dict]) -> None:
    layout_data = Workbook()
    del layout_data["Sheet"]
    add_ind_to_sheet(layout_data, best_ind)

    excel_name, identifier = create_unique_name()
    excel_path = Path(excel_name)
    json_path = excel_path.with_suffix(".json")

    layout_data.save(excel_path)
    save_json(best_ind, json_path)
    print(f"layout gespeichert: {identifier}")
    print(f"Excel: {excel_path}")
    print(f"JSON: {json_path}")


def save_Excel(best_ind: List[Dict]) -> None:
    save_sheet(best_ind)
