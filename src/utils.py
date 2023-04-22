'''
Handy utility functions
Author: rzfeng
'''
from typing import Callable, List, Optional

import torch

from src.integration import ExplicitEulerIntegrator


# Wraps angles (in radians) to (-pi, pi]
# @input theta [torch.tensor (B)]: angles in radians
# @output [torch.tensor (B)]: wrapped angles
def wrap_radians(theta: torch.tensor) -> torch.tensor:
    wrapped = torch.remainder(theta, 2*torch.pi)
    wrapped[wrapped > torch.pi] -= 2*torch.pi
    return wrapped


# Converts NED poses to standard axes
# @input ned [torch.tensor (B,6)]: poses in NED with ZYX Euler angles (x, y, z, alpha, beta, gamma)
# @output [torch.tensor (B,6)]: points in NWU axes with ZYX euler angles
def ned_to_nwu(ned: torch.tensor) -> torch.tensor:
    return torch.stack((ned[:,0], -ned[:,1], -ned[:,2],
                        -ned[:,3], -ned[:,4], ned[:,5]), dim=1)


# Converts standard poses to NED axes
# @input nwu [torch.tensor (B,6)]: poses in NWU axes with ZYX Euler angles (x, y, z, alpha, beta, gamma)
# @output [torch.tensor (B,6)]: points in NED axes with ZYX euler angles
def nwu_to_ned(nwu: torch.tensor) -> torch.tensor:
    return torch.stack((nwu[:,0], -nwu[:,1], -nwu[:,2],
                        -nwu[:,3], -nwu[:,4], nwu[:,5]), dim=1)


# Generates a discrete dynamics rollout function
# @input dt [float]: time step length
# @input continuous_dynamics [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor(B x control_dim)) -> torch.tensor (B x state_dim)]:
#       continuous dynamics function computing state derivatives
# @output [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B x T x state_dim)]:
#       dynamics rollout function
def generate_discrete_dynamics(dt: float, continuous_dynamics: Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]) \
                              -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
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
        integrator = ExplicitEulerIntegrator(dt, continuous_dynamics)
        curr_state = s0.clone()
        for t_idx in range(T):
            curr_state = integrator(t+t_idx*dt, curr_state.unsqueeze(1), u[:,t_idx,:].unsqueeze(1)).squeeze(1)
            state_traj[:,t_idx,:] = curr_state
        return state_traj
    return discrete_dynamics