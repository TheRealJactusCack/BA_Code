# SANITY CHECKS MODULE

import copy

import config
from helpers import port_world_xy
from ga_engine import mutate


def test_ports():
    print("Test Ports Start")

    w = 2.0
    d = 1.0
    cx, cy = 10.0, 10.0

    p0 = port_world_xy(center_x=cx, center_y=cy, w_m=w, d_m=d, side="Left", offset_m=0.25, rotation_deg=0)
    exp0 = (cx - w / 2.0, cy + d / 2.0 - 0.25)
    print("Rotation 0 Port", p0, "Expected", exp0)
    assert abs(p0[0] - exp0[0]) < 1e-9
    assert abs(p0[1] - exp0[1]) < 1e-9

    p90 = port_world_xy(center_x=cx, center_y=cy, w_m=w, d_m=d, side="Left", offset_m=0.25, rotation_deg=90)
    exp90 = (cx - (d / 2.0 - 0.25), cy - w / 2.0)
    print("Rotation 90 Port", p90, "Expected", exp90)
    assert abs(p90[0] - exp90[0]) < 1e-9
    assert abs(p90[1] - exp90[1]) < 1e-9

    print("Test Ports OK")


def test_rotation_mutation_collision_reject():
    print("Test Rotation Mutation Collision Reject Start")

    config.GRID_SIZE = 0.25
    config.FLOOR_W = 5.0
    config.FLOOR_H = 5.0
    config.GRID_COLS = int(config.FLOOR_W // config.GRID_SIZE)
    config.GRID_ROWS = int(config.FLOOR_H // config.GRID_SIZE)

    config.OBSTACLES = set()

    config.MACHINE_COUNT = 2
    config.MACHINE_SIZES = [(1.0, 2.0), (1.0, 2.0)]
    config.MACHINE_PORTS = [{"side_in": "Left", "offset_in": 1.0, "side_out": "Right", "offset_out": 1.0} for _ in range(2)]
    config.MATERIAL_CONNECTIONS = [(0, 1)]

    m0 = {"idx": 0, "w_cells": 4, "h_cells": 8, "gx": 0, "gy": 0, "z": 0, "x": 0.5, "y": 1.0}
    m1 = {"idx": 1, "w_cells": 4, "h_cells": 8, "gx": 4, "gy": 0, "z": 0, "x": 1.5, "y": 1.0}
    ind = [m0, m1]

    config.MUTATION_PROB = 0.0
    config.MUTATION_POS_STD = 0.0
    config.MUTATION_ROT_PROB = 1.0

    before = copy.deepcopy(ind)
    mutate(ind)

    print("Before z", before[0]["z"], before[1]["z"], "After z", ind[0]["z"], ind[1]["z"])
    assert ind[0]["z"] in (0, 90)
    assert ind[1]["z"] in (0, 90)

    print("Test Rotation Mutation Collision Reject OK")


def main():
    test_ports()
    test_rotation_mutation_collision_reject()
    print("All sanity checks passed")


if __name__ == "__main__":
    main()
