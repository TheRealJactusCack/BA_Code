from openpyxl import Workbook
from typing import Any, Dict, List, Optional, Tuple

layout_data = Workbook()

def create_unique_name():
    Name = "Layout"
    return Name

def add_ind_to_sheet(best_ind: List[Dict], geöffnetes_sheet):
    worksheet = layout_data.create_sheet(title = "Optimales layout")
    worksheet.append(["Label", "X Position", "Y Position"])
    return

def save_sheet():
    del layout_data["Sheet"]
    layout_data.save("Optimal Facility Layout")
    print("layout gespeichert")
    return