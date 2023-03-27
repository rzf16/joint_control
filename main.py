'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
from typing import List, Callable, Dict, Tuple
import time
import yaml

import torch
import numpy as np

import seaborn
seaborn.set()

from src.vehicles import Unicycle, Bicycle, Quadrotor, LinearizedQuadrotor
from src.system import VehicleSystem
from src.mppi import MPPI


CFG_PATH = "cfg.yaml"


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    system = extract_cfg_vehicles(cfg)
    # TODO: figure out how to deal with u_min/u_max/u0/terminal_cost for only some vehicles that have them
    controller = MPPI(system.generate_discrete_dynamics(cfg["mppi_params"]["dt"]), system.cost_fn,
                      system.state_dim(), system.control_dim(),
                      torch.diag(torch.tensor([n for vehicle_name in system.vehicles.keys()
                                                 for n in cfg["vehicles"][vehicle_name]["mppi_sigma_diag"]])),
                      cfg["mppi_params"]["dt"], cfg["mppi_params"]["horizon"],
                      n_samples=cfg["mppi_params"]["n_samples"], lambda_=cfg["mppi_params"]["lambda_"])
    s = torch.tensor([n for vehicle_name in system.vehicles.keys() for n in cfg["vehicles"][vehicle_name]["s0"]])
    controller.warm_start(s, cfg["mppi_params"]["warm_start_steps"])

    # TODO: make goal test
    # TODO: per-vehicle cost weights
    times = []
    for i in range(100):
        t = i*cfg["mppi_params"]["dt"]
        tic = time.time()
        control = controller.get_command(s)
        toc = time.time()
        times.append(toc - tic)
        system.apply_control(control.repeat(2,1), (t, t+cfg["mppi_params"]["dt"]))
        s = system.get_state()
        # print(controller.cost)

    print(sum(times) / len(times))
    system.recorder.animate2d(list(system.vehicles.values()))
    # system.recorder.plot_state_traj(system.vehicles["quad1"])
    # system.recorder.plot_control_traj(system.vehicles["quad1"])


# Extracts a dictionary of Vehicle objects and cost functions from the configuration
# @input cfg [Dict]: configuration
# @output [VehicleSystem]: VehicleSystem object
def extract_cfg_vehicles(cfg: Dict) -> VehicleSystem:
    vehicles = {}
    cost_fns = {}
    for vehicle_name, vehicle_info in cfg["vehicles"].items():
        if vehicle_info["type"] == "unicycle":
            vehicles[vehicle_name] = Unicycle(vehicle_name,
                                              vehicle_info["vis_params"],
                                              torch.tensor(vehicle_info["s0"]))
        elif vehicle_info["type"] == "bicycle":
            vehicles[vehicle_name] = Bicycle(vehicle_name,
                                             vehicle_info["lf"],
                                             vehicle_info["lr"],
                                             vehicle_info["vis_params"],
                                             torch.tensor(vehicle_info["s0"]))
        elif vehicle_info["type"] == "quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_name] = Quadrotor(vehicle_name,
                                                   vehicle_info["m"],
                                                   torch.tensor(vehicle_info["inertia"]),
                                                   vehicle_info["vis_params"],
                                                   torch.tensor(vehicle_info["s0"]),
                                                   g=vehicle_info["g"])
            else:
                vehicles[vehicle_name] = Quadrotor(vehicle_name,
                                                   vehicle_info["m"],
                                                   torch.tensor(vehicle_info["inertia"]),
                                                   vehicle_info["vis_params"],
                                                   torch.tensor(vehicle_info["s0"]))
        elif vehicle_info["type"] == "linearized_quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_name] = LinearizedQuadrotor(vehicle_name,
                                                             vehicle_info["m"],
                                                             torch.tensor(vehicle_info["inertia"]),
                                                             vehicle_info["vis_params"],
                                                             torch.tensor(vehicle_info["s0"]),
                                                             g=vehicle_info["g"])
            else:
                vehicles[vehicle_name] = LinearizedQuadrotor(vehicle_name,
                                                             vehicle_info["m"],
                                                             torch.tensor(vehicle_info["inertia"]),
                                                             vehicle_info["vis_params"],
                                                             torch.tensor(vehicle_info["s0"]))
        else:
            print("[Main] Error! Unrecognized vehicle type.")
            exit()

        if vehicle_info["objective"] == "goal":
            cost_fns[vehicle_name] = generate_goal_cost(torch.tensor(vehicle_info["cost_params"]["goal"]),
                                                        torch.diag(torch.tensor(vehicle_info["cost_params"]["Q_diag"])),
                                                        torch.diag(torch.tensor(vehicle_info["cost_params"]["R_diag"])))
        elif vehicle_info["objective"] == "traj":
            cost_fns[vehicle_name] = generate_traj_cost(torch.from_numpy(np.genfromtxt(vehicle_info["cost_params"]["traj"],
                                                                                       delimiter=",", dtype=np.float32)),
                                                        torch.diag(torch.tensor(vehicle_info["cost_params"]["Q_diag"])),
                                                        torch.diag(torch.tensor(vehicle_info["cost_params"]["R_diag"])))
        else:
            print("[Main] Error! Unrecognized objective type.")
            exit()

    return VehicleSystem(vehicles, cost_fns)


# Generates an MPPI cost function for a goal-reaching objective
# @input goal [torch.tensor (state_dim)]: goal state
# @input Q [torch.tensor (state_dim x state_dim)]: state weight matrix
# @input R [torch.tensor (control_dim x control_dim)]: control weight matrix
# output [function(torch.tensor (B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]:
#       MPPI cost function
def generate_goal_cost(goal: torch.tensor, Q: torch.tensor, R: torch.tensor) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
    # Goal-reaching cost function for MPPI, computing the cost for a batch of state and control trajectories
    # @input t [torch.tensor (B)]: batch of initial timesteps
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def quad_goal_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # Not using any control cost for now, since MPPI has its own control cost formula!
        B = s.size(0)
        T = s.size(1)
        state_dim = s.size(2)
        # control_dim = u.size(2)
        batch_Q = Q.repeat(B*T,1,1)
        # batch_R = R.repeat(B*T,1,1)
        diffs = goal - s

        cost = torch.bmm(diffs.reshape(B*T,1,state_dim), torch.bmm(batch_Q, diffs.reshape(B*T, state_dim, 1))).reshape(B,T).sum(dim=1)
        # cost += torch.bmm(u.reshape(B*T,1,control_dim), torch.bmm(batch_R, u.reshape(B*T, control_dim, 1))).reshape(B,T).sum(dim=1)
        return cost
    return quad_goal_cost


# Generates an MPPI cost function for a trajectory-tracking objective
# @input ref [torch.tensor (T_ref x state_dim)]: reference trajectory
# @input Q [torch.tensor (state_dim x state_dim)]: state weight matrix
# @input R [torch.tensor (control_dim x control_dim)]: control weight matrix
# output [function(torch.tensor(B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]:
#       MPPI cost function
def generate_traj_cost(ref: torch.tensor, Q: torch.tensor, R: torch.tensor) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
    # Trajectory-tracking cost function for MPPI, computing the cost for a batch of state and control trajectories
    # @input t [torch.tensor (B)]: batch of initial timesteps
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def quad_traj_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # Not using any control cost for now, since MPPI has its own control cost formula!
        B = s.size(0)
        T = s.size(1)
        T_ref = ref.size(0)
        state_dim = s.size(2)
        # control_dim = u.size(2)

        # This assumes t is filled with the same element, but it's REALLY hard to get around this assumption without complex
        # logic to deal with the different length of each trajectory difference!
        ref_end = int(min(t[0] + T, T_ref))
        T_diff = ref_end - int(t[0])
        diffs = ref[int(t[0]):ref_end,:] - s[:,:T_diff,:]

        batch_Q = Q.repeat(B*T_diff,1,1)
        # batch_R = R.repeat(B*T_diff,1,1)

        cost = torch.bmm(diffs.reshape(B*T_diff,1,state_dim), torch.bmm(batch_Q, diffs.reshape(B*T_diff, state_dim, 1))).reshape(B,T_diff).sum(dim=1)
        # cost += torch.bmm(u.reshape(B*T_diff,1,control_dim), torch.bmm(batch_R, u.reshape(B*T_diff, control_dim, 1))).reshape(B,T_diff).sum(dim=1)
        return cost
    return quad_traj_cost


if __name__ == "__main__":
    main()