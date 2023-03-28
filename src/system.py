'''
Vehicle system combining multiple vehicles and data recording into one class
Author: rzfeng
'''
from copy import deepcopy
from typing import Dict, Callable, Tuple, Optional

import torch

from src.vehicles import Vehicle
from src.recorder import DataRecorder
from src.integration import ExplicitEulerIntegrator


class VehicleSystem:
    # @input vehicles [Dict[str, Vehicle]]: dictionary of vehicles
    # @input running_costs [Dict[str, function(torch.tensor(B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]]:
    #       dictionary of vehicle running cost functions
    # @input terminal_costs [Dict[str, Optional[function(torch.tensor (B x T x state_dim)) -> torch.tensor (B)]]]:
    #       dictionary of vehicle terminal cost functions
    # TODO: add joint cost functions
    def __init__(self, vehicles: Dict[str, Vehicle],
                       running_costs: Dict[str, Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]],
                       terminal_costs: Dict[str, Optional[Callable[[torch.tensor], torch.tensor]]],
                       ):
        self.vehicles = vehicles
        self.running_costs = running_costs
        self.terminal_costs = terminal_costs
        self.state = torch.cat([vehicle.get_state() for vehicle in self.vehicles.values()])

        self.state_idxs = {}
        self.control_idxs = {}
        state_idx = 0
        control_idx = 0
        for vehicle_name, vehicle in self.vehicles.items():
            self.state_idxs[vehicle_name] = (state_idx, state_idx + vehicle.state_dim())
            self.control_idxs[vehicle_name] = (control_idx, control_idx + vehicle.control_dim())
            state_idx += vehicle.state_dim()
            control_idx += vehicle.control_dim()

        self.recorder = DataRecorder(self.vehicles.values())

    def get_state_description(self):
        return [var_description for vehicle in self.vehicles.values() for var_description in vehicle.get_state_description()]

    def get_control_description(self):
        return [var_description for vehicle in self.vehicles.values() for var_description in vehicle.get_control_description()]

    def state_dim(self):
        return sum([vehicle.state_dim() for vehicle in self.vehicles.values()])

    def control_dim(self):
        return sum([vehicle.control_dim() for vehicle in self.vehicles.values()])

    def get_state(self):
        return deepcopy(self.state)

    # Computes the state derivatives for a batch of states and controls
    # @input t [torch.tensor (B)]: time points
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @input u [torch.tensor (B x control_dim)]: batch of controls
    # @output [torch.tensor (B x state_dim)]: batch of state derivatives
    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        ds = torch.zeros_like(s)
        for vehicle_name, vehicle in self.vehicles.items():
            vehicle_s = s[:,self.state_idxs[vehicle_name][0]:self.state_idxs[vehicle_name][1]]
            vehicle_u = u[:,self.control_idxs[vehicle_name][0]:self.control_idxs[vehicle_name][1]]
            ds[:,self.state_idxs[vehicle_name][0]:self.state_idxs[vehicle_name][1]] = vehicle.continuous_dynamics(t, vehicle_s, vehicle_u)
        return ds

    # Generates a discrete dynamics rollout function
    # @input dt [float]: time step length
    # @output [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B x T x state_dim)]:
    #       dynamics rollout function
    def generate_discrete_dynamics(self, dt: float) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
        vehicle_fns = [vehicle.generate_discrete_dynamics(dt) for vehicle in self.vehicles.values()]
        # Discrete dynamics rollout function, rolling out a batch of initial states and times using a batch of control trajectories
        # @input t [torch.tensor (B)]: batch of initial times
        # @input s0 [torch.tensor (B x state_dim)]: batch of initial states
        # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
        # @output [torch.tensor (B x T x state_dim)]: batch of state trajectories
        def discrete_dynamics(t: torch.tensor, s0: torch.tensor, u: torch.tensor) -> torch.tensor:
            B = t.size(0)
            T = u.size(1)
            state_dim = s0.size(1)

            state_traj = torch.zeros((B, T, state_dim))
            for vehicle_name, vehicle_fn in zip(self.vehicles.keys(), vehicle_fns):
                vehicle_s0 = s0[:,self.state_idxs[vehicle_name][0]:self.state_idxs[vehicle_name][1]]
                vehicle_u = u[:,:,self.control_idxs[vehicle_name][0]:self.control_idxs[vehicle_name][1]]
                state_traj[:,:,self.state_idxs[vehicle_name][0]:self.state_idxs[vehicle_name][1]] = vehicle_fn(t, vehicle_s0, vehicle_u)
            return state_traj
            # integrator = ExplicitEulerIntegrator(dt, self.continuous_dynamics)
            # curr_state = deepcopy(s0)
            # for t_idx in range(T):
            #     curr_state = integrator(t+t_idx*dt, curr_state.unsqueeze(1), u[:,t_idx,:].unsqueeze(1)).squeeze(1)
            #     state_traj[:,t_idx,:] = curr_state
            # return state_traj
        return discrete_dynamics

    # Joint running cost function, computing the cost for a batch of state and control trajectories
    # @input t [torch.tensor (B)]: batch of initial timesteps
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def running_cost(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        B = t.size(0)
        cost = torch.zeros(B)
        for vehicle_name in self.vehicles.keys():
            # Add the cost for this vehicle
            cost += self.running_costs[vehicle_name](t, s[:,:,self.state_idxs[vehicle_name][0]:self.state_idxs[vehicle_name][1]],
                                                     u[:,:,self.control_idxs[vehicle_name][0]:self.control_idxs[vehicle_name][1]])
        return cost

    # Joint terminal cost function, computing the cost for a batch of states
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @output [torch.tensor (B)]: batch of costs
    def terminal_cost(self, s: torch.tensor) -> torch.tensor:
        B = s.size(0)
        cost = torch.zeros(B)
        for vehicle_name in self.vehicles.keys():
            # Add the cost for this vehicle
            if self.terminal_costs[vehicle_name] is not None:
                cost += self.terminal_costs[vehicle_name](s[:,:,self.state_idxs[vehicle_name][0]:self.state_idxs[vehicle_name][1]])
        return cost

    # Applies a control sequence to the system
    # @input u [torch.tensor (T x control_dim)]: linear spline control trajectory
    # @input t_span [Tuple(float, float)]: time duration
    # @input t_eval Optional[torch.tensor (N)]: evaluation time points
    # @output [torch.tensor (N x state_dim)]: state trajectory
    # @output [torch.tensor(N)]: timestamps of state trajectory
    def apply_control(self, u: torch.tensor, t_span: Tuple[float, float], t_eval: Optional[torch.tensor] = None) -> torch.tensor:
        state_traj = []
        timestamps = None
        for vehicle_name, vehicle in self.vehicles.items():
            vehicle_u = u[:, self.control_idxs[vehicle_name][0]:self.control_idxs[vehicle_name][1]]
            vehicle_state_traj, timestamps = vehicle.apply_control(vehicle_u, t_span, t_eval)
            state_traj.append(vehicle_state_traj)

            self.recorder.log_control(vehicle, vehicle_u, t_span[0] * torch.ones(vehicle_u.size(0)))
            self.recorder.log_state(vehicle, vehicle_state_traj, timestamps)

        state_traj = torch.cat(state_traj, dim=1)
        self.state = torch.cat([vehicle.get_state() for vehicle in self.vehicles.values()])
        return state_traj, timestamps


# TODO: LatchingVehicleSystem