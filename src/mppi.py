'''
Model predictive path integral controller
Author: rzfeng
'''
from typing import Callable, Optional

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI:
    # @input dynamics [function(torch.tensor (K), torch.tensor (K x state_dim), torch.tensor (K x T x control_dim)) -> torch.tensor (K x T x state_dim)]:
    #       batched dynamics function mapping times, initial states, and linear spline control trajectories to state trajectories
    # @input running_cost [function(torch.tensor (K), torch.tensor (K x T x state_dim), torch.tensor (K x T x control_dim)) -> torch.tensor (K)]:
    #       batched cost function computing the running cost for state-control trajectories and timesteps
    # @input state_dim [int]: state dimensionality
    # @input control_dim [int]: control dimensionality
    # @input noise_sigma [torch.tensor (control_dim x control_dim)]: covariance for control sampling
    # @input dt [float]: time step length
    # @input horizon [int]: time horizon (in steps)
    # @input n_samples [int]: number of control trajectory samples to take
    # @input lambda_ [float]: temperature parameter for control exploration
    # @input u_min [torch.tensor (state_dim)]: minimum control
    # @input u_max [torch.tensor (state_dim)]: maximum control
    # @input u0 [torch.tensor (T x state_dim)]: initial control trajectory
    # @input terminal_cost [function(torch.tensor (K x T x state_dim)) -> torch.tensor (K)]:
    #       batched terminal cost function for state trajectories
    # @input device [str]: device to run on
    def __init__(self, dynamics: Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor],
                       running_cost: Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor],
                       state_dim: int,
                       control_dim: int,
                       noise_sigma: torch.tensor,
                       dt: float,
                       horizon: int,
                       n_samples: int = 1000,
                       lambda_: float = 1e-2,
                       u_min: Optional[torch.tensor] = None,
                       u_max: Optional[torch.tensor] = None,
                       u0: Optional[torch.tensor] = None,
                       terminal_cost: Optional[Callable[[torch.tensor], torch.tensor]] = None,
                       device="cpu"):
        self.device = device

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.T = horizon
        self.K = n_samples
        self.lambda_ = lambda_

        self.noise_sigma = noise_sigma.to(self.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(torch.zeros(self.control_dim).to(self.device), self.noise_sigma)
        self.u_min = u_min.to(self.device) if u_min is not None else u_min
        self.u_max = u_max.to(self.device) if u_max is not None else u_max
        self.u0 = u0.to(self.device) if u0 is not None else u0

        self.X = None
        self.U = self.u0 if self.u0 is not None else self.noise_dist.sample((self.T,))
        self.cost = None
        self.time = 0.0
        self.timesteps = 0

    # Warm starts MPPI by running optimization for n_steps steps without executing
    # @input state [torch.tensor (state_dim)]: current state
    # @input n_steps [int]: number of steps to warm start with
    def warm_start(self, state: torch.tensor, n_steps: int):
        for _ in range(n_steps):
            self.get_command(state, shift=0)

    # Computes the MPPI control trajectory for an initial state
    # @input state [torch.tensor (state_dim)]: current state
    # @input shift [int]: time steps shift the trajectory forward (assuming the first control(s) is executed)
    # @output [torch.tensor (control_dim)]: first control of the control trajectory
    def get_command(self, state: torch.tensor, shift: int = 1) -> torch.tensor:
        batch_state = state.to(self.device).repeat(self.K, 1)
        batch_control = self.U.repeat(self.K,1,1)
        batch_noise_sigma_inv = self.noise_sigma_inv.repeat(self.K, 1, 1)

        # Sample K trajectory perturbations
        perturbations = self.noise_dist.sample((self.K, self.T))
        control_traj_samples = self.U + perturbations
        control_traj_samples = torch.minimum(control_traj_samples, self.u_max) if self.u_max is not None else control_traj_samples
        control_traj_samples = torch.maximum(control_traj_samples, self.u_min) if self.u_min is not None else control_traj_samples

        # Roll out trajectories
        state_traj_samples = self.dynamics(self.time * torch.ones(self.K), batch_state, control_traj_samples)

        # Compute costs
        sample_costs = self.running_cost(self.timesteps * torch.ones(self.K), state_traj_samples, control_traj_samples) + \
                       self.lambda_ * torch.bmm(batch_control,
                                                torch.bmm(batch_noise_sigma_inv,
                                                          perturbations.transpose(1,2)).sum(dim=2, keepdim=True)
                                                ).squeeze().sum(dim=1)
        if self.terminal_cost is not None:
            sample_costs += self.terminal_cost(state_traj_samples)

        # Compute weights
        beta = sample_costs.min()
        weights = torch.exp(-(sample_costs - beta) / self.lambda_)
        weights /= weights.sum()

        # Modify nominal control trajectory
        self.U += (weights.reshape(-1,1,1) * perturbations).sum(dim=0)

        # Get the predicted state trajectory from the nominal control trajectory
        self.X = self.dynamics(torch.tensor([self.time]), state.to(self.device).unsqueeze(0), self.U.unsqueeze(0)).squeeze(0)

        control = self.U[0,:]
        if shift > 0:
            self.U = self.U.roll(-shift, dims=0)
            self.U[-shift,:] = 0.0
            self.time += self.dt * shift
            self.timesteps += shift
        return control