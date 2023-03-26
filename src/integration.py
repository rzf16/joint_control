'''
Discrete integration methods
Author: rzfeng
'''
from abc import ABC, abstractmethod
from typing import Callable

import torch


# Abstract class for integration methods
class Integrator(ABC):
    # @input dt [float]: time step
    # @input fn [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor (B x control_dim)) -> torch.tensor (B x state_dim)]:
    #     batched dynamics function returning state derivatives for state-control pairs at specified time points
    def __init__(self, dt: float, dynamics: Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]):
        self.dt = dt
        self.dynamics = dynamics

    # Integrates a batch of state and control trajectories forward one time step
    # @input t [torch.tensor (B)]: batch of time points
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of linear spline control trajectories
    # @output [torch.tensor (B x T x state_dim)]: batch of next state trajectories
    @abstractmethod
    def __call__(self, t: torch.tensor, s: torch.tensor, u: torch.tensor):
        raise NotImplementedError()


class ExplicitEulerIntegrator(Integrator):
    def __init__(self, dt: float, dynamics: Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]):
        super().__init__(dt, dynamics)

    def __call__(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        B = s.size(0)
        T = s.size(1)
        state_dim = s.size(2)
        control_dim = u.size(2)

        # Get the proper time points to pass into the dynamics function (in case they are time-dependent)
        t_steps = self.dt * torch.arange(0, T)
        batch_t = t.unsqueeze(1) + t_steps.repeat(B,1)

        ds = self.dynamics(batch_t.reshape((B*T,)), s.reshape((B*T, state_dim)), u.reshape((B*T, control_dim))).reshape((B, T, state_dim))
        return s + self.dt * ds