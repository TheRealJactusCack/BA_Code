from openpyxl import Workbook
import random
from typing import Dict, List

def create_unique_name():
    random_string = random.randint(1111111,9999999)
    Name = f"Layout_{random_string}.xlsx"
    return Name, random_string

def add_ind_to_sheet(layout_data, best_ind):#best_ind: List[Dict]:
    worksheet = layout_data.create_sheet(title = "Optimales layout")
    worksheet.append(["Label", "X Position", "Y Position", "Rotation"])
    for machine in best_ind:
        label = machine.get("Label", "idx")
        x_pos = machine.get("x", "")
        y_pos = machine.get("y", "")
        rot = machine.get("z", "rotation")
        worksheet.append([label, x_pos, y_pos , rot])
    return

def save_sheet(best_ind: List[Dict]):
    layout_data = Workbook()
    del layout_data["Sheet"]
    add_ind_to_sheet(layout_data, best_ind)
    Name, indentifier = create_unique_name()
    layout_data.save(Name)
    print(f"layout gespeichert: {indentifier}")
    return

def save_Excel(best_ind: List[Dict]):
    save_sheet(best_ind)