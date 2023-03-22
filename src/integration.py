'''
Discrete integration methods
Author: rzfeng
'''
from abc import ABC, abstractmethod

import torch


# Abstract class for integration methods
class Integrator(ABC):
    # @input dt [float]: time step
    def __init__(self, dt: float):
        self.dt = dt

    # Integrates a state trajectory s over the period dt using ds
    @abstractmethod
    def __call__(self, s: torch.tensor, ds: torch.tensor):
        raise NotImplementedError()


class ExplicitEulerIntegrator(Integrator):
    def __init__(self, dt: float):
        super().__init__(dt)

    # Integrates a state trajectory s over the period dt using ds with Explicit Euler integration
    # @input s [torch.tensor (T x state_dim)]: state trajectory
    # @input ds [torch.tensor (T x state_dim)]: state derivative trajectory
    # @output [torch.tensor (T x state_dim)]: next state trajectory
    def __call__(self, s: torch.tensor, ds: torch.tensor) -> torch.tensor:
        return s + self.dt * ds


class ImplicitEulerIntegrator(Integrator):
    def __init__(self, dt: float):
        super().__init__(dt)

    # Integrates a state trajectory s over the period dt using ds with Explicit Euler integration
    # @input s [torch.tensor (T x state_dim)]: state trajectory
    # @input ds [torch.tensor (T+1 x state_dim)]: state derivative trajectory
    # @output [torch.tensor (T x state_dim)]: next state trajectory
    def __call__(self, s: torch.tensor, ds: torch.tensor) -> torch.tensor:
        return s + self.dt * ds[1:,:]


class TrapezoidalIntegrator(Integrator):
    def __init__(self, dt: float):
        super().__init__(dt)

    # Integrates a state trajectory s over the period dt using ds with trapezoidal integration
    # @input s [torch.tensor (T x state_dim)]: state trajectory
    # @input ds [torch.tensor (T+1 x state_dim)]: state derivative trajectory
    # @output [torch.tensor (T x state_dim)]: next state trajectory
    def __call__(self, s: torch.tensor, ds: torch.tensor) -> torch.tensor:
        return s + self.dt * (ds[:-1,:] + ds[1:,:]) / 2.0


# TODO: Make HermiteSimpsonIntegrator