'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
from typing import List, Callable, Dict, Tuple
import yaml

import torch
import numpy as np

import seaborn
seaborn.set()

from src.system import Vehicle, Unicycle, Bicycle, Quadrotor, LinearizedQuadrotor
from src.recorder import DataRecorder
from src.integration import ExplicitEulerIntegrator
from src.mppi import MPPI


CFG_PATH = "cfg.yaml"


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    vehicles, cost_fns = extract_cfg_vehicles(cfg)
    recorder = DataRecorder(vehicles.values())

    mppi_dynamics = generate_mppi_dynamics(vehicles.values(), cfg["mppi_params"]["dt"])
    mppi_cost = generate_joint_cost(vehicles.values(), cost_fns.values())
    # TODO: do this more cleanly! figure out how to deal with u_min/u_max/u0/terminal_cost for only some vehicles that have them
    controller = MPPI(mppi_dynamics, mppi_cost,
                      sum([vehicle.state_dim() for vehicle in vehicles.values()]),
                      sum([vehicle.control_dim() for vehicle in vehicles.values()]),
                      torch.diag(torch.tensor([n for vehicle_name in vehicles.keys()
                                                 for n in cfg["vehicles"][vehicle_name]["mppi_sigma_diag"]])),
                      cfg["mppi_params"]["dt"], cfg["mppi_params"]["horizon"],
                      n_samples=cfg["mppi_params"]["n_samples"], lambda_=cfg["mppi_params"]["lambda_"])
    s = torch.tensor([n for vehicle_name in vehicles.keys() for n in cfg["vehicles"][vehicle_name]["s0"]])
    recorder.log_state(vehicles["bike1"], s.unsqueeze(0), torch.tensor([0.0]))
    controller.warm_start(s, cfg["mppi_params"]["warm_start_steps"])

    # TODO: make goal test
    # TODO: make state/control to vehicle and vice versa map
    for i in range(150):
        t = i*cfg["mppi_params"]["dt"]
        control = controller.get_command(s)
        vehicles["bike1"].apply_control(control.repeat(2,1), (t, t+cfg["mppi_params"]["dt"]))
        recorder.log_state(vehicles["bike1"], vehicles["bike1"].get_state().unsqueeze(0), torch.tensor([t]))
        recorder.log_control(vehicles["bike1"], control.unsqueeze(0), torch.tensor([t]))
        s = vehicles["bike1"].get_state()

    recorder.animate2d([vehicles["bike1"]])


# Extracts a dictionary of Vehicle objects and cost functions from the configuration
# @input cfg [Dict]: configuration
# @output [Dict[str, Vehicle]]: dictionary of Vehicle objects
# @output [Dict[function(torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]]:
#       dictionary of cost functions
def extract_cfg_vehicles(cfg: Dict) -> Tuple[Dict[str, Vehicle], Dict[str, Callable]]:
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
                                                                                               delimiter=",")),
                                                                torch.diag(torch.tensor(vehicle_info["cost_params"]["Q_diag"])),
                                                                torch.diag(torch.tensor(vehicle_info["cost_params"]["R_diag"])))
        else:
            print("[Main] Error! Unrecognized objective type.")
            exit()

    return vehicles, cost_fns


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
    def mppi_goal_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
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
    return mppi_goal_cost


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
    def mppi_traj_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # Not using any control cost for now, since MPPI has its own control cost formula!
        B = s.size(0)
        T = s.size(1)
        T_ref = ref.size(0)
        state_dim = s.size(2)
        # control_dim = u.size(2)
        batch_Q = Q.repeat(B*T,1,1)
        # batch_R = R.repeat(B*T,1,1)
        # This assumes t is filled with the same element, but it's REALLY hard to get around this assumption without complex
        # logic to deal with the different length of each trajectory difference!
        ref_end = int(min(t[0] + T, T_ref))
        T_diff = ref_end - t[0]
        diffs = ref[t[0]:ref_end,:] - s[:T_diff]

        cost = torch.bmm(diffs.reshape(B*T_diff,1,state_dim), torch.bmm(batch_Q, diffs.reshape(B*T_diff, state_dim, 1))).reshape(B,T_diff).sum(dim=1)
        # cost += torch.bmm(u.reshape(B*T_diff,1,control_dim), torch.bmm(batch_R, u.reshape(B*T_diff, control_dim, 1))).reshape(B,T_diff).sum(dim=1)
        return cost
    return mppi_traj_cost


# Generates a joint MPPI cost function for several vehicles
# @input vehicles [List[Vehicle]]: list of vehicles
# @input cost_fns [List[function(torch.tensor(B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]]:
#       vehicle cost functions
# @output [function(torch.tensor(B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]:
#       joint cost function
def generate_joint_cost(vehicles: List[Vehicle],
                        cost_fns: List[Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]]) ->\
                       Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
    # Joint cost function for MPPI, computing the cost for a batch of state and control trajectories
    # @input t [torch.tensor (B)]: batch of initial timesteps
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def mppi_joint_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        B = t.size(0)
        cost = torch.zeros(B)

        # Start indices for the states and controls corresponding to the vehicle
        state_idx = 0
        control_idx = 0
        for vehicle, cost_fn in zip(vehicles, cost_fns):
            # Add the cost for this vehicle
            cost += cost_fn(t, s[:,:,state_idx:state_idx+vehicle.state_dim()], u[:,:,control_idx:control_idx+vehicle.control_dim()])
            # Move on to the next vehicle
            state_idx += vehicle.state_dim()
            control_idx += vehicle.control_dim()

        return cost
    return mppi_joint_cost

# Generates an MPPI dynamics function
# @input vehicles [List[Vehicle]]: list of vehicles
# @input dt [float]: time step length
# @output [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B x T x state_dim)]:
#       MPPI dynamics rollout function
def generate_mppi_dynamics(vehicles: List[Vehicle], dt: float) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
    # Dynamics rollout function for MPPI, rolling out a batch of initial states and times using a batch of control trajectories
    # @input t [torch.tensor (B)]: batch of initial times
    # @input s0 [torch.tensor (B x state_dim)]: batch of initial states
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B x T x state_dim)]: batch of state trajectories
    def mppi_dynamics(t: torch.tensor, s0: torch.tensor, u: torch.tensor) -> torch.tensor:
        B = t.size(0)
        T = u.size(1)
        state_dim = s0.size(1)

        state_traj = torch.zeros((B, T, state_dim))
        # Start indices for the states and controls corresponding to the vehicle
        state_idx = 0
        control_idx = 0
        for vehicle in vehicles:
            integrator = ExplicitEulerIntegrator(dt, vehicle.continuous_dynamics)
            # Get the state for this vehicle
            curr_state = s0[:,state_idx:state_idx+vehicle.state_dim()]
            # Roll out!
            for t_idx in range(T):
                curr_state = integrator(t+t_idx*dt, curr_state.unsqueeze(1), u[:,t_idx,control_idx:control_idx+vehicle.control_dim()].unsqueeze(1)).squeeze(1)
                state_traj[:,t_idx,state_idx:state_idx+vehicle.state_dim()] = curr_state
            # Move on to the next vehicle
            state_idx += vehicle.state_dim()
            control_idx += vehicle.control_dim()

        return state_traj
    return mppi_dynamics


if __name__ == "__main__":
    main()