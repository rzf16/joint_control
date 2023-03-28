'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
from typing import Callable, Dict, Tuple
import time
import yaml

import torch
import numpy as np

import seaborn
seaborn.set()

from src.vehicles import Unicycle, Bicycle, Quadrotor, LinearizedQuadrotor
from src.system import VehicleSystem
from src.mppi import MPPI
from src.costs import generate_goal_cost, generate_traj_cost, \
                      generate_obstacle_cost, generate_collision_cost


CFG_PATH = "cfg.yaml"


def main():
    # Instantiate system and controller
    cfg = yaml.safe_load(open(CFG_PATH))
    system = extract_vehicles(cfg)
    goal_state, goal_test = extract_goal(cfg)
    controller = MPPI(system.generate_discrete_dynamics(cfg["mppi"]["dt"]),
                      system.running_cost,
                      system.state_dim(),
                      system.control_dim(),
                      torch.diag(torch.tensor([n for vehicle_name in system.vehicles.keys()
                                                 for n in cfg["vehicles"][vehicle_name]["mppi_sigma_diag"]])),
                      cfg["mppi"]["dt"],
                      cfg["mppi"]["horizon"],
                      n_samples=cfg["mppi"]["n_samples"],
                      lambda_=cfg["mppi"]["lambda_"],
                      u_min=torch.tensor([n for vehicle_name in system.vehicles.keys()
                                            for n in (cfg["vehicles"][vehicle_name]["u_min"]
                                                      if "u_min" in cfg["vehicles"][vehicle_name].keys()
                                                      else system.vehicles[vehicle_name].control_dim() * [-torch.inf])]),
                      u_max=torch.tensor([n for vehicle_name in system.vehicles.keys()
                                            for n in (cfg["vehicles"][vehicle_name]["u_max"]
                                                      if "u_max" in cfg["vehicles"][vehicle_name].keys()
                                                      else system.vehicles[vehicle_name].control_dim() * [torch.inf])]),
                      u0=torch.cat([torch.tensor([n for n in cfg["vehicles"][vehicle_name]["u0"]])
                                    if "u0" in cfg["vehicles"][vehicle_name]
                                    else torch.nan * torch.ones((cfg["mppi"]["horizon"],
                                                                 system.vehicles[vehicle_name].control_dim()))
                                    for vehicle_name in system.vehicles.keys()], dim=1),
                      terminal_cost=system.terminal_cost,
                      device="cuda0" if torch.cuda.is_available() else "cpu")

    s = torch.tensor([n for vehicle_name in system.vehicles.keys() for n in cfg["vehicles"][vehicle_name]["s0"]])
    controller.warm_start(s, cfg["mppi"]["warm_start_steps"])

    # Main control loop
    times = []
    goal_reached = False
    for i in range(cfg["mppi"]["max_steps"]):
        t = i*cfg["mppi"]["dt"]
        tic = time.time()
        control = controller.get_command(s)
        toc = time.time()
        times.append(toc - tic)
        system.apply_control(control.repeat(2,1), (t, t+cfg["mppi"]["dt"]))
        s = system.get_state()

        if goal_test(s.unsqueeze(0)).squeeze():
            goal_reached = True
            print(f"MPPI reached the goal after {i+1} steps ({t+cfg['mppi']['dt']} seconds)!")
            break

    if not goal_reached:
        print(f"MPPI failed to reach the goal within tolerance after {i+1} steps ({t+cfg['mppi']['dt']} seconds) D:")
    print(f"Final state: {s.tolist()}")
    print(f"Final goal error: {(goal_state - s).tolist()}")
    print(f"Average MPPI compute time: {sum(times) / len(times)} seconds")

    # Data recording and visualization
    system.recorder.write_data(CFG_PATH,
                               vehicles=[vehicle for vehicle in system.vehicles.values()
                                         if cfg["vehicles"][vehicle.name]["write_data"]])

    for vehicle_name in system.vehicles.keys():
        if cfg["vehicles"][vehicle_name]["plot_state"]:
            system.recorder.plot_state_traj(system.vehicles[vehicle_name])
        if cfg["vehicles"][vehicle_name]["plot_control"]:
            system.recorder.plot_control_traj(system.vehicles[vehicle_name])

    obstacles = [(*obs["center"], obs["radius"], obs["height"]) for obs in cfg["obstacles"]]

    system.recorder.plot_traj2d([vehicle for vehicle in system.vehicles.values()
                                 if cfg["vehicles"][vehicle.name]["plot_traj2d"]],
                                obstacles=obstacles)
    system.recorder.plot_traj3d([vehicle for vehicle in system.vehicles.values()
                                 if cfg["vehicles"][vehicle.name]["plot_traj3d"]])

    system.recorder.animate2d([vehicle for vehicle in system.vehicles.values()
                               if cfg["vehicles"][vehicle.name]["animate2d"]],
                              obstacles=obstacles,
                              hold_traj=cfg["animation"]["2d"]["hold_traj"],
                              n_frames=cfg["animation"]["2d"]["n_frames"] if cfg["animation"]["3d"]["n_frames"] > 0 else None,
                              fps=cfg["animation"]["2d"]["fps"],
                              end_wait=cfg["animation"]["2d"]["end_wait"],
                              write=cfg["animation"]["2d"]["filename"] if cfg["animation"]["3d"]["filename"] else None)
    system.recorder.animate3d([vehicle for vehicle in system.vehicles.values()
                               if cfg["vehicles"][vehicle.name]["animate3d"]],
                               hold_traj=cfg["animation"]["3d"]["hold_traj"],
                               n_frames=cfg["animation"]["3d"]["n_frames"] if cfg["animation"]["3d"]["n_frames"] > 0 else None,
                               fps=cfg["animation"]["3d"]["fps"],
                               end_wait=cfg["animation"]["3d"]["end_wait"],
                               write=cfg["animation"]["3d"]["filename"] if cfg["animation"]["3d"]["filename"] else None)


# Extracts a VehicleSystem from the configuration
# @input cfg [Dict]: configuration
# @output [VehicleSystem]: VehicleSystem object
def extract_vehicles(cfg: Dict) -> VehicleSystem:
    obstacles = torch.stack([torch.tensor([*obs["center"], obs["radius"], obs["height"]]) for obs in cfg["obstacles"]])

    vehicles = {}
    running_costs = {}
    terminal_costs = {}
    for vehicle_name, vehicle_info in cfg["vehicles"].items():
        if vehicle_info["type"] == "unicycle":
            vehicles[vehicle_name] = Unicycle(vehicle_name,
                                              torch.tensor(vehicle_info["s0"]),
                                              vehicle_info["collision_radius"],
                                              vehicle_info["collision_height"],
                                              vehicle_info["vis_params"])
        elif vehicle_info["type"] == "bicycle":
            vehicles[vehicle_name] = Bicycle(vehicle_name,
                                             torch.tensor(vehicle_info["s0"]),
                                             vehicle_info["collision_radius"],
                                             vehicle_info["collision_height"],
                                             vehicle_info["lf"],
                                             vehicle_info["lr"],
                                             vehicle_info["vis_params"])
        elif vehicle_info["type"] == "quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_name] = Quadrotor(vehicle_name,
                                                   torch.tensor(vehicle_info["s0"]),
                                                   vehicle_info["collision_radius"],
                                                   vehicle_info["collision_height"],
                                                   vehicle_info["m"],
                                                   torch.tensor(vehicle_info["inertia"]),
                                                   vehicle_info["vis_params"],
                                                   g=vehicle_info["g"])
            else:
                vehicles[vehicle_name] = Quadrotor(vehicle_name,
                                                   torch.tensor(vehicle_info["s0"]),
                                                   vehicle_info["collision_radius"],
                                                   vehicle_info["collision_height"],
                                                   vehicle_info["m"],
                                                   torch.tensor(vehicle_info["inertia"]),
                                                   vehicle_info["vis_params"])
        elif vehicle_info["type"] == "linearized_quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_name] = LinearizedQuadrotor(vehicle_name,
                                                             torch.tensor(vehicle_info["s0"]),
                                                             vehicle_info["collision_radius"],
                                                             vehicle_info["collision_height"],
                                                             vehicle_info["m"],
                                                             torch.tensor(vehicle_info["inertia"]),
                                                             vehicle_info["vis_params"],
                                                             g=vehicle_info["g"])
            else:
                vehicles[vehicle_name] = LinearizedQuadrotor(vehicle_name,
                                                             torch.tensor(vehicle_info["s0"]),
                                                             vehicle_info["collision_radius"],
                                                             vehicle_info["collision_height"],
                                                             vehicle_info["m"],
                                                             torch.tensor(vehicle_info["inertia"]),
                                                             vehicle_info["vis_params"])
        else:
            print("[Main] Error! Unrecognized vehicle type.")
            exit()

        if vehicle_info["objective"]["type"] == "goal":
            Q = torch.diag(torch.tensor(vehicle_info["objective"]["Q_diag"]))
            R = torch.diag(torch.tensor(vehicle_info["objective"]["R_diag"]))
            running_costs[vehicle_name] = generate_goal_cost(torch.tensor(vehicle_info["objective"]["goal"]),
                                                             vehicle_info["cost_weight"] * Q,
                                                             vehicle_info["cost_weight"] * R)
        elif vehicle_info["objective"]["type"] == "traj":
            Q = torch.diag(torch.tensor(vehicle_info["objective"]["Q_diag"]))
            R = torch.diag(torch.tensor(vehicle_info["objective"]["R_diag"]))
            running_costs[vehicle_name] = generate_traj_cost(torch.from_numpy(np.genfromtxt(vehicle_info["objective"]["traj"],
                                                                                            delimiter=",", dtype=np.float32)),
                                                             vehicle_info["cost_weight"] * Q,
                                                             vehicle_info["cost_weight"] * R)
        else:
            print("[Main] Error! Unrecognized objective type.")
            exit()

        terminal_costs[vehicle_name] = generate_obstacle_cost(vehicles[vehicle_name],
                                                              obstacles,
                                                              cfg["mppi"]["collision_cost"])

    system = VehicleSystem(vehicles, running_costs, terminal_costs, [], [])

    joint_running_costs = []
    joint_terminal_costs = [generate_collision_cost(system, cfg["mppi"]["collision_cost"])]
    # TODO: read in costs from cfg
    system.joint_terminal_costs.extend(joint_terminal_costs)

    return system

# Extracts the goal and a goal test function from the configuration
# @input cfg [Dict]: configuration
# @output [function(torch.tensor (B x state_dim)) -> torch.tensor]: goal test
def extract_goal(cfg: Dict) -> Tuple[torch.tensor, Callable[[torch.tensor], torch.tensor]]:
    goal_state = []
    tolerance = []
    for vehicle_info in cfg["vehicles"].values():
        if vehicle_info["objective"]["type"] == "goal":
            goal_state.extend(vehicle_info["objective"]["goal"])
        elif vehicle_info["objective"]["type"] == "traj":
            goal_state.extend(np.genfromtxt(vehicle_info["objective"]["traj"],delimiter=",",dtype=np.float32)[-1,:].tolist())
        else:
            print("[Main] Error! Unrecognized objective type.")
            exit()
        tolerance.extend(vehicle_info["objective"]["tolerance"])
    goal_state = torch.tensor(goal_state)
    tolerance = torch.tensor(tolerance)

    # Returns true if the goal is reached within the tolerance
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B)]: boolean tensor specifying goal reached
    def goal_test(s: torch.tensor):
        return torch.all(torch.abs(goal_state - s) < tolerance, dim=1)

    return goal_state, goal_test


if __name__ == "__main__":
    main()