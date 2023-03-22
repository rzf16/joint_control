'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
import yaml

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import seaborn
seaborn.set()

from src.system import Unicycle, Bicycle, Quadrotor, LinearizedQuadrotor
from src.recorder import DataRecorder


CFG_PATH = "cfg.yaml"


def vehicles_from_cfg(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    vehicles = {}
    for vehicle_info in cfg["vehicles"]:
        if vehicle_info["type"] == "unicycle":
            vehicles[vehicle_info["name"]] = Unicycle(vehicle_info["name"],
                                                      vehicle_info["vis_params"],
                                                      torch.tensor(vehicle_info["s0"]))
        elif vehicle_info["type"] == "bicycle":
            vehicles[vehicle_info["name"]] = Bicycle(vehicle_info["name"],
                                                     vehicle_info["lf"],
                                                     vehicle_info["lr"],
                                                     vehicle_info["vis_params"],
                                                     torch.tensor(vehicle_info["s0"]))
        elif vehicle_info["type"] == "quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_info["name"]] = Quadrotor(vehicle_info["name"],
                                                           vehicle_info["m"],
                                                           torch.tensor(vehicle_info["inertia"]),
                                                           vehicle_info["vis_params"],
                                                           torch.tensor(vehicle_info["s0"]),
                                                           g=vehicle_info["g"])
            else:
                vehicles[vehicle_info["name"]] = Quadrotor(vehicle_info["name"],
                                                           vehicle_info["m"],
                                                           torch.tensor(vehicle_info["inertia"]),
                                                           vehicle_info["vis_params"],
                                                           torch.tensor(vehicle_info["s0"]))
        elif vehicle_info["type"] == "linearized_quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_info["name"]] = LinearizedQuadrotor(vehicle_info["name"],
                                                                     vehicle_info["m"],
                                                                     torch.tensor(vehicle_info["inertia"]),
                                                                     vehicle_info["vis_params"],
                                                                     torch.tensor(vehicle_info["s0"]),
                                                                     g=vehicle_info["g"])
            else:
                vehicles[vehicle_info["name"]] = LinearizedQuadrotor(vehicle_info["name"],
                                                                     vehicle_info["m"],
                                                                     torch.tensor(vehicle_info["inertia"]),
                                                                     vehicle_info["vis_params"],
                                                                     torch.tensor(vehicle_info["s0"]))
        else:
            print("[Main] Error! Unrecognized vehicle type.")
            exit()

    return vehicles


def main():
    vehicles = vehicles_from_cfg(CFG_PATH)
    recorder = DataRecorder(vehicles.values())

    quad1_tf = 5.0
    quad1_u = torch.zeros((50, 4))
    quad1_u[:5,2] = -np.pi/256
    quad1_u[:,0] = 10.0
    state_traj, timestamps = vehicles["quad1"].apply_control(quad1_u, (0.0, quad1_tf))
    recorder.log_state(vehicles["quad1"], state_traj, timestamps)
    recorder.log_control(vehicles["quad1"], quad1_u, torch.linspace(0.0, quad1_tf, quad1_u.size(0)))

    recorder.plot_state_traj(vehicles["quad1"])
    # recorder.animate2d([vehicles["quad1"]])


if __name__ == "__main__":
    main()