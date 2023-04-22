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
                      generate_obstacle_cost, generate_collision_cost, generate_distance_cost


CFG_PATH = "cfg.yaml"


def main():
    # Instantiate system and controller
    cfg = yaml.safe_load(open(CFG_PATH))
    system = extract_vehicles(cfg)
    objectives, goal_states, goal_test = extract_goal(cfg, system)
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
    dt = cfg["mppi"]["dt"]
    times = []
    for i in range(cfg["mppi"]["max_steps"]):
        t = i*dt
        tic = time.time()
        control = controller.get_command(s)
        toc = time.time()
        times.append(toc - tic)
        system.apply_control(control.repeat(2,1), (t, t+dt))
        s = system.get_state()

        all_goals_reached, goal_reached = goal_test(s.unsqueeze(0))
        all_goals_reached = all_goals_reached.squeeze()
        goal_reached = {key: value.squeeze() for key, value in goal_reached.items()}
        if all_goals_reached:
            print(f"MPPI reached all goals after {i+1} steps ({t+cfg['mppi']['dt']} seconds)!")
            print("\n")
            break

    if not all_goals_reached:
        print(f"MPPI failed to reach the goals within tolerance after {i+1} steps ({t+dt} seconds) D:")
        for vehicle_name, success in goal_reached.items():
            if success:
                print(f"{vehicle_name} reached its goal!")
            else:
                print(f"{vehicle_name} failed to reach its goal D:")
        print("\n")

    for vehicle_name, vehicle in system.vehicles.items():
        print(f"{vehicle_name} final state: {vehicle.get_state().tolist()}")
        print(f"{vehicle_name} goal error: {(vehicle.get_state() - goal_states[vehicle_name]).tolist()}")
        print("\n")
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
                                [objective for vehicle_name, objective in objectives.items()
                                           if cfg["vehicles"][vehicle_name]["plot_traj2d"]],
                                obstacles=obstacles)
    # system.recorder.plot_traj3d([vehicle for vehicle in system.vehicles.values()
    #                              if cfg["vehicles"][vehicle.name]["plot_traj3d"]])

    system.recorder.animate2d([vehicle for vehicle in system.vehicles.values()
                                       if cfg["vehicles"][vehicle.name]["animate2d"]],
                              [objective for vehicle_name, objective in objectives.items()
                                         if cfg["vehicles"][vehicle_name]["animate2d"]],
                              obstacles=obstacles,
                              hold_traj=cfg["animation"]["2d"]["hold_traj"],
                              n_frames=cfg["animation"]["2d"]["n_frames"] if cfg["animation"]["2d"]["n_frames"] > 0 else None,
                              fps=cfg["animation"]["2d"]["fps"],
                              end_wait=cfg["animation"]["2d"]["end_wait"],
                              write=cfg["animation"]["2d"]["filename"] if cfg["animation"]["2d"]["filename"] else None)
    # system.recorder.animate3d([vehicle for vehicle in system.vehicles.values()
    #                            if cfg["vehicles"][vehicle.name]["animate3d"]],
    #                            hold_traj=cfg["animation"]["3d"]["hold_traj"],
    #                            n_frames=cfg["animation"]["3d"]["n_frames"] if cfg["animation"]["3d"]["n_frames"] > 0 else None,
    #                            fps=cfg["animation"]["3d"]["fps"],
    #                            end_wait=cfg["animation"]["3d"]["end_wait"],
    #                            write=cfg["animation"]["3d"]["filename"] if cfg["animation"]["3d"]["filename"] else None)


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
                                              vehicle_info["vis_params"],
                                              noise=torch.diag(torch.tensor(vehicle_info["noise"])))
        elif vehicle_info["type"] == "bicycle":
            vehicles[vehicle_name] = Bicycle(vehicle_name,
                                             torch.tensor(vehicle_info["s0"]),
                                             vehicle_info["collision_radius"],
                                             vehicle_info["collision_height"],
                                             vehicle_info["lf"],
                                             vehicle_info["lr"],
                                             vehicle_info["vis_params"],
                                             noise=torch.diag(torch.tensor(vehicle_info["noise"])))
        elif vehicle_info["type"] == "quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_name] = Quadrotor(vehicle_name,
                                                   torch.tensor(vehicle_info["s0"]),
                                                   vehicle_info["collision_radius"],
                                                   vehicle_info["collision_height"],
                                                   vehicle_info["m"],
                                                   torch.tensor(vehicle_info["inertia"]),
                                                   vehicle_info["vis_params"],
                                                   g=vehicle_info["g"],
                                                   noise=torch.diag(torch.tensor(vehicle_info["noise"])))
            else:
                vehicles[vehicle_name] = Quadrotor(vehicle_name,
                                                   torch.tensor(vehicle_info["s0"]),
                                                   vehicle_info["collision_radius"],
                                                   vehicle_info["collision_height"],
                                                   vehicle_info["m"],
                                                   torch.tensor(vehicle_info["inertia"]),
                                                   vehicle_info["vis_params"],
                                                   noise=torch.diag(torch.tensor(vehicle_info["noise"])))
        elif vehicle_info["type"] == "linearized_quadrotor":
            if "g" in vehicle_info.keys():
                vehicles[vehicle_name] = LinearizedQuadrotor(vehicle_name,
                                                             torch.tensor(vehicle_info["s0"]),
                                                             vehicle_info["collision_radius"],
                                                             vehicle_info["collision_height"],
                                                             vehicle_info["m"],
                                                             torch.tensor(vehicle_info["inertia"]),
                                                             vehicle_info["vis_params"],
                                                             g=vehicle_info["g"],
                                                             noise=torch.diag(torch.tensor(vehicle_info["noise"])))
            else:
                vehicles[vehicle_name] = LinearizedQuadrotor(vehicle_name,
                                                             torch.tensor(vehicle_info["s0"]),
                                                             vehicle_info["collision_radius"],
                                                             vehicle_info["collision_height"],
                                                             vehicle_info["m"],
                                                             torch.tensor(vehicle_info["inertia"]),
                                                             vehicle_info["vis_params"],
                                                             noise=torch.diag(torch.tensor(vehicle_info["noise"])))
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
    for joint_cost in cfg["joint_costs"]:
        if joint_cost["running"]:
            pass
        else:
            if joint_cost["type"] == "distance":
                joint_terminal_costs.append(generate_distance_cost(system, joint_cost["ego"],
                                                                   joint_cost["targets"], joint_cost["dist"], joint_cost["cost"]))
            else:
                print("[Main] Error! Unrecognized joint cost type.")
                exit()

    system.joint_running_costs.extend(joint_running_costs)
    system.joint_terminal_costs.extend(joint_terminal_costs)

    return system

# Extracts the goal and a goal test function from the configuration
# @input cfg [Dict]: configuration
# @output [Dict[str: torch.tensor]]: objectives (either a goal state or trajectory) for each vehicle
# @output [Dict[str: torch.tensor (state_dim)]]: goal states for each vehicle
# @output [function(torch.tensor (B x full_state_dim)) -> torch.tensor (B, n_vehicles)]: goal test
def extract_goal(cfg: Dict, system: VehicleSystem) -> Tuple[Dict, Dict, Callable[[torch.tensor], Tuple[Dict, torch.tensor]]]:
    objectives = {}
    goal_states = {}
    tolerances = {}
    for vehicle_name, vehicle_info in cfg["vehicles"].items():
        if vehicle_info["objective"]["type"] == "goal":
            objectives[vehicle_name] = torch.tensor(vehicle_info["objective"]["goal"])
            goal_states[vehicle_name] = torch.tensor(vehicle_info["objective"]["goal"])
        elif vehicle_info["objective"]["type"] == "traj":
            objectives[vehicle_name] = torch.from_numpy(np.genfromtxt(vehicle_info["objective"]["traj"],
                                                                      delimiter=",", dtype=np.float32))
            goal_states[vehicle_name] = objectives[vehicle_name][-1,:]
        else:
            print("[Main] Error! Unrecognized objective type.")
            exit()
        tolerances[vehicle_name] = torch.tensor(vehicle_info["objective"]["tolerance"])

    # Returns true if the goal is reached within the tolerance
    # @input s [torch.tensor (B x full_state_dim)]: batch of states
    # @output [torch.tensor (B)]: tensor indicating goal reached for all vehicles
    # @output [Dict[str: torch.tensor (B)]]: boolean tensor specifying goal reached for each vehicle
    def goal_test(s: torch.tensor):
        goal_reached = {}
        all_goals_reached = torch.ones(s.size(0), dtype=torch.bool)
        for vehicle_name in system.vehicles.keys():
            vehicle_state = s[:, system.state_idxs[vehicle_name][0]:system.state_idxs[vehicle_name][1]]
            goal_reached[vehicle_name] = torch.all(torch.abs(vehicle_state - goal_states[vehicle_name]) < tolerances[vehicle_name], dim=1)
            all_goals_reached = torch.logical_and(all_goals_reached, goal_reached[vehicle_name])
        return all_goals_reached, goal_reached

    return objectives, goal_states, goal_test


if __name__ == "__main__":
    main()