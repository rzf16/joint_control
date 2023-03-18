'''
Simulation models for UGV and quadrotor
Author: rzfeng
'''
from abc import ABC, abstractmethod, abstractclassmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
from torchdiffeq import odeint
from scipy.spatial.transform import Rotation as R

from src.vis_utils import draw_box, draw_cylinder
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from src.utils import wrap_radians


# Class for describing a state or control variable
@dataclass
class VarDescription:
    name: str
    var_type: str # real or circle
    description: str


class Vehicle(ABC):
    # @input name [str]: vehicle name
    # @input s0 [torch.tensor (state_dim)]: initial vehicle state
    # @input vis_params [Dict("color": str)]: visualization parameters
    def __init__(self, name: str, s0: torch.tensor, vis_params: Dict):
        assert(s0.size(0) == self.state_dim())
        self.name = name
        self._state = s0
        self.vis_params = vis_params

    @abstractclassmethod
    def get_state_description(cls) -> List[VarDescription]:
        raise NotImplementedError()

    @abstractclassmethod
    def get_control_description(cls) -> List[VarDescription]:
        raise NotImplementedError()

    # Returns the layout for state trajectory plotting
    # @output [np.ndarray (AxB)]: layout for state trajectory plotting, where AxB >= state_dim and
    #                             each element is the index to a state variable
    @abstractclassmethod
    def get_state_plot_layout(cls) -> np.ndarray:
        raise NotImplementedError()

    # Returns the layout for control trajectory plotting
    # @output [np.ndarray (AxB)]: layout for control trajectory plotting, where AxB >= control_dim and
    #                             each element is the index to a control variable
    @abstractclassmethod
    def get_control_plot_layout(cls) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def state_dim(cls) -> int:
        return len(cls.get_state_description())

    @classmethod
    def control_dim(cls) -> int:
        return len(cls.get_control_description())

    def set_state(self, s: torch.tensor):
        self._state = s
        # Wrap any angles
        state_description = self.get_state_description()
        for i in range(self.state_dim()):
            if state_description[i].var_type == "circle":
                self._state[i] = wrap_radians(torch.tensor([self._state[i]])).squeeze()

    def get_state(self):
        return deepcopy(self._state)

    # Computes the state derivatives for a batch of states and controls
    # @input t [torch.tensor (B)]: time points
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @input u [torch.tensor (B x control_dim)]: batch of controls
    # @output [torch.tensor (B x state_dim)]: batch of state derivatives
    @abstractmethod
    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        raise NotImplementedError()

    # Computes the Jacobians for a batch of states and controls
    # @input t [torch.tensor (B)]: time points
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @input u [torch.tensor (B x control_dim)]: batch of controls
    # @output [Optional[torch.tensor (B x state_dim x state_dim)]]: batch of Jacobians
    def jacobian(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> Optional[torch.tensor]:
        return None

    # Forward-integrates the dynamics of the system given a start state and a control trajectory
    # @input s0 [torch.tensor (state_dim)]: initial state
    # @input u [torch.tensor (T x control_dim)]: linear spline control trajectory
    # @input t_span [Tuple(float, float)]: time span
    # @input t_eval Optional[torch.tensor (N)]: evaluation time points
    # @output [torch.tensor (N x state_dim)]: state trajectory
    def integrate(self, s0: torch.tensor, u: torch.tensor, t_span: Tuple[float, float], t_eval: Optional[torch.tensor] = None) -> torch.tensor:
        t_spaced = torch.linspace(t_span[0], t_span[1], u.size(0))
        t_eval = t_spaced if t_eval is None else t_eval

        # Wraps the continuous dynamics into the form required by the ODE solver
        # @input t [torch.tensor ()]: time point
        # @input s [torch.tensor (state_dim)]: state
        # @output [torch.tensor (state_dim)]: state derivative
        def ode_dynamics(t: torch.tensor, s: torch.tensor):
            u_t = torch.tensor([np.interp(t, t_spaced, u[:,i].numpy()) for i in range(self.control_dim())])
            ds_t = self.continuous_dynamics(t.unsqueeze(0), s.unsqueeze(0), u_t.unsqueeze(0))
            return ds_t.squeeze(0)

        sol = odeint(ode_dynamics, s0, t_eval)
        return sol, t_eval

    # Applies a control sequence to the system
    # @input u [torch.tensor (T x control_dim)]: linear spline control trajectory
    # @input t_span [Tuple(float, float)]: time duration
    # @input t_eval Optional[torch.tensor (N)]: evaluation time points
    # @output [torch.tensor (N x state_dim)]: state trajectory
    def apply_control(self, u: torch.tensor, t_span: Tuple[float, float], t_eval: Optional[torch.tensor] = None) -> torch.tensor:
        state_traj, timestamps = self.integrate(self._state, u, t_span, t_eval)
        self.set_state(state_traj[-1,:])
        return state_traj, timestamps

    # Extracts the 2D xy vehicle position from a batch of states
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B x 2)]: batch of 2D positions
    @classmethod
    def get_pos2d(cls, s: torch.tensor) -> torch.tensor:
        return cls.get_pos3d(s)[:,:2]

    # Extracts the 3D vehicle position from a batch of states
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B x 3)]: batch of positions
    @abstractclassmethod
    def get_pos3d(cls, s: torch.tensor) -> torch.tensor:
        raise NotImplementedError()

    # Adds the 2D vehicle visualization to Matplotlib axes
    # @input ax [Axes]: axes to visualize on
    # @input s [torch.tensor (state_dim)]: state to visualize
    # @output [List[Artist])]: Matplotlib artists for the visualization
    @abstractmethod
    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        raise NotImplementedError()

    # Adds the 3D vehicle visualization to Matplotlib axes
    # @input ax [Axes3D]: axes to visualize on
    # @input s [torch.tensor (state_dim)]: state to visualize
    # @output [List[Artist])]: Matplotlib artists for the visualization
    @abstractmethod
    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        raise NotImplementedError()


class Unicycle(Vehicle):
    # @input name [str]: vehicle name
    # @input vis_params [Dict("length": float, "width": float, "h"eight: float,
    #                         "wheel_radius": float, "wheel_width": float, "color": str)]:
    #     paramters for vehicle visualization (length, width, height, wheel radius, wheel width, color)
    # @input s0 [torch.tensor (state_dim)]: initial vehicle state
    def __init__(self, name: str, vis_params: Dict, s0: torch.tensor):
        super().__init__(name, s0, vis_params)

    @classmethod
    def get_state_description(cls) -> List[VarDescription]:
        return [
            VarDescription("x", "real", "x position"),
            VarDescription("y", "real", "y position"),
            VarDescription("theta", "circle", "yaw"),
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("v", "real", "velocity"),
            VarDescription("omega", "circle", "angular velocity")
        ]

    @classmethod
    def get_state_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0],
            [1],
            [2],
        ])

    @classmethod
    def get_control_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0],
            [1]
        ])

    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # This should be fine since indexing returns views, not copies!
        theta = s[:,2]
        vel = u[:,0]
        omega = wrap_radians(u[:,1])

        ds = torch.zeros_like(s)
        ds[:,0] = vel * torch.cos(theta)
        ds[:,1] = vel * torch.sin(theta)
        ds[:,2] = omega
        return ds

    @classmethod
    def get_pos3d(cls, s: torch.tensor) -> torch.tensor:
        return torch.cat((s[:,:2], torch.ones((s.size(0), 1))), dim=1)

    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        anchor = s[:2].numpy() - 0.5*np.array([self.vis_params["length"], self.vis_params["width"]])
        patch = Rectangle(anchor, self.vis_params["length"], self.vis_params["width"],
                          angle=np.degrees(s[2]), rotation_point=tuple(s[:2].tolist()),
                          color=self.vis_params["color"])
        ax.add_patch(patch)
        return [patch]

    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw body
        body_center = np.append(s[:2].numpy(), 0.5*self.vis_params["height"] + 0.5*self.vis_params["wheel_radius"])
        rot = R.from_euler("z", s[2])
        body_dims = np.array([self.vis_params["length"], self.vis_params["width"], self.vis_params["height"]])
        artists.extend(draw_box(ax, body_center, rot.as_quat(), body_dims, self.vis_params["color"]))

        # Draw wheels
        # for l in (-self.lr, self.lf):
        #     for j in (-1, 1):
        #         # Rotate about the center of the body, not the origin!
        #         wheel_center = np.array([l, j*0.5*(self.vis_params["width"] + self.vis_params["wheel_width"]), 0.0])
        #         wheel_center = rot.apply(wheel_center)
        #         wheel_center += body_center
        #         wheel_axis = rot.apply(np.array([0., 1., 0.]))
        #         wheel_dims = np.array([self.vis_params["wheel_radius"], self.vis_params["wheel_width"]])
        #         artists.extend(draw_cylinder(ax, wheel_center, wheel_axis, wheel_dims, self.vis_params["color"]))

        return artists

class Bicycle(Vehicle):
    # @input name [str]: vehicle name
    # @input lf [float]: length from the front axle to the center of gravity
    # @input lr [float]: length from the rear axle to the center of gravity
    # @input vis_params [Dict("length": float, "width": float, "h"eight: float,
    #                         "wheel_radius": float, "wheel_width": float, "color": str)]:
    #     paramters for vehicle visualization (length, width, height, wheel radius, wheel width, color)
    # @input s0 [torch.tensor (state_dim)]: initial vehicle state
    def __init__(self, name: str, lf: float, lr: float, vis_params: Dict, s0: torch.tensor):
        super().__init__(name, s0, vis_params)
        self.lf = lf
        self.lr = lr

    @classmethod
    def get_state_description(cls) -> List[VarDescription]:
        return [
            VarDescription("x", "real", "x position"),
            VarDescription("y", "real", "y position"),
            VarDescription("psi", "circle", "yaw"),
            VarDescription("v", "real", "velocity")
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("a", "real", "acceleration"),
            VarDescription("delta", "circle", "steering angle")
        ]

    @classmethod
    def get_state_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0, 1],
            [2, 3]
        ])

    @classmethod
    def get_control_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0],
            [1]
        ])

    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # This should be fine since indexing returns views, not copies!
        vel = s[:,3]
        psi = s[:,2]
        accel = u[:,0]
        steer_angle = wrap_radians(u[:,1])
        beta = torch.atan2(self.lr * torch.tan(steer_angle), torch.tensor(self.lf + self.lr))

        ds = torch.zeros_like(s)
        ds[:,0] = vel * torch.cos(psi + beta)
        ds[:,1] = vel * torch.sin(psi + beta)
        ds[:,2] = vel * torch.sin(beta) / self.lr
        ds[:,3] = accel
        return ds

    @classmethod
    def get_pos3d(cls, s: torch.tensor) -> torch.tensor:
        return torch.cat((s[:,:2], torch.ones((s.size(0), 1))), dim=1)

    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        anchor = s[:2].numpy() - 0.5*np.array([self.vis_params["length"], self.vis_params["width"]])
        patch = Rectangle(anchor, self.vis_params["length"], self.vis_params["width"],
                          angle=np.degrees(s[2]), rotation_point=tuple(s[:2].tolist()),
                          color=self.vis_params["color"])
        ax.add_patch(patch)
        return [patch]

    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw body
        body_center = np.append(s[:2].numpy(), 0.5*self.vis_params["height"] + 0.5*self.vis_params["wheel_radius"])
        rot = R.from_euler("z", s[2])
        body_dims = np.array([self.vis_params["length"], self.vis_params["width"], self.vis_params["height"]])
        artists.extend(draw_box(ax, body_center, rot.as_quat(), body_dims, self.vis_params["color"]))

        # Draw wheels
        # for l in (-self.lr, self.lf):
        #     for j in (-1, 1):
        #         # Rotate about the center of the body, not the origin!
        #         wheel_center = np.array([l, j*0.5*(self.vis_params["width"] + self.vis_params["wheel_width"]), 0.0])
        #         wheel_center = rot.apply(wheel_center)
        #         wheel_center += body_center
        #         wheel_axis = rot.apply(np.array([0., 1., 0.]))
        #         wheel_dims = np.array([self.vis_params["wheel_radius"], self.vis_params["wheel_width"]])
        #         artists.extend(draw_cylinder(ax, wheel_center, wheel_axis, wheel_dims, self.vis_params["color"]))

        return artists