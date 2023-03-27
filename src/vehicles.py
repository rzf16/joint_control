'''
Simulation models for UGV and quadrotor
Author: rzfeng
'''
from abc import ABC, abstractmethod, abstractclassmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Callable

import torch
import numpy as np
from torchdiffeq import odeint
from scipy.spatial.transform import Rotation as R

from src.vis_utils import draw_box, draw_cylinder
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D

from src.utils import wrap_radians, ned_to_nwu, nwu_to_ned
from src.integration import ExplicitEulerIntegrator


# Class for describing a state or control variable
@dataclass
class VarDescription:
    name: str
    var_type: str # real or circle
    description: str
    unit: str


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

    # Generates a discrete dynamics rollout function
    # @input dt [float]: time step length
    # @output [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B x T x state_dim)]:
    #       dynamics rollout function
    def generate_discrete_dynamics(self, dt: float) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
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
            integrator = ExplicitEulerIntegrator(dt, self.continuous_dynamics)
            curr_state = deepcopy(s0)
            for t_idx in range(T):
                curr_state = integrator(t+t_idx*dt, curr_state.unsqueeze(1), u[:,t_idx,:].unsqueeze(1)).squeeze(1)
                state_traj[:,t_idx,:] = curr_state
            return state_traj
        return discrete_dynamics

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

        sol = odeint(ode_dynamics, s0, t_eval, method="rk4") # Dopri5 was giving some issues with step size
        return sol, t_eval

    # Applies a control sequence to the system
    # @input u [torch.tensor (T x control_dim)]: linear spline control trajectory
    # @input t_span [Tuple(float, float)]: time duration
    # @input t_eval Optional[torch.tensor (N)]: evaluation time points
    # @output [torch.tensor (N x state_dim)]: state trajectory
    # @output [torch.tensor(N)]: timestamps of state trajectory
    def apply_control(self, u: torch.tensor, t_span: Tuple[float, float], t_eval: Optional[torch.tensor] = None) -> torch.tensor:
        state_traj, timestamps = self.integrate(self._state, u, t_span, t_eval)
        # Wrap any angles
        state_description = self.get_state_description()
        for i in range(self.state_dim()):
            if state_description[i].var_type == "circle":
                state_traj[:,i] = wrap_radians(state_traj[:,i])
        self.set_state(state_traj[-1,:])
        return state_traj, timestamps

    # Extracts the SE(2) vehicle pose from a batch of states
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B x 3)]: batch of SE(2) poses
    @classmethod
    def get_pose_se2(cls, s: torch.tensor) -> torch.tensor:
        se3 = cls.get_pose_se3(s)
        return torch.stack((se3[:,0], se3[:,1], se3[:,3]), dim=1)

    # Extracts the SE(3) vehicle pose from a batch of states
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B x 6)]: batch of SE(3) poses with ZYX Euler angles
    @abstractclassmethod
    def get_pose_se3(cls, s: torch.tensor) -> torch.tensor:
        raise NotImplementedError()

    # Extracts the 2D xy vehicle position from a batch of states
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B x 2)]: batch of 2D positions
    @classmethod
    def get_pos2d(cls, s: torch.tensor) -> torch.tensor:
        return cls.get_pose_se3(s)[:,:2]

    # Extracts the 3D vehicle position from a batch of states
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @output [torch.tensor (B x 3)]: batch of 3D positions
    @classmethod
    def get_pos3d(cls, s: torch.tensor) -> torch.tensor:
        return cls.get_pose_se3(s)[:,:3]

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
    # @input vis_params [Dict("length": float, "width": float, "height": float,
    #                         "wheel_radius": float, "wheel_width": float, "color": str)]:
    #     paramters for vehicle visualization (length, width, height, wheel radius, wheel width, color)
    # @input s0 [torch.tensor (state_dim)]: initial vehicle state
    def __init__(self, name: str, vis_params: Dict, s0: torch.tensor):
        super().__init__(name, s0, vis_params)

    @classmethod
    def get_state_description(cls) -> List[VarDescription]:
        return [
            VarDescription("x", "real", "x position", "m"),
            VarDescription("y", "real", "y position", "m"),
            VarDescription("theta", "circle", "yaw", "rad"),
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("v", "real", "velocity", "m/s"),
            VarDescription("omega", "circle", "angular velocity", "rad/s")
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
    def get_pose_se3(cls, s: torch.tensor) -> torch.tensor:
        return torch.cat((s[:,:2], torch.zeros((s.size(0), 1)), s[:,2].unsqueeze(1), torch.zeros((s.size(0), 2))), dim=1)

    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        se2 = self.get_pose_se2(s.unsqueeze(0)).squeeze().numpy()
        anchor = se2[:2] - 0.5*np.array([self.vis_params["length"], self.vis_params["width"]])
        patch = Rectangle(anchor, self.vis_params["length"], self.vis_params["width"],
                          angle=np.degrees(se2[2]), rotation_point=tuple(se2[:2].tolist()),
                          color=self.vis_params["color"])
        ax.add_patch(patch)
        return [patch]

    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw body
        se2 = self.get_pose_se2(s.unsqueeze(0)).squeeze().numpy()
        body_center = np.append(se2[:2], 0.5*self.vis_params["height"] + 0.5*self.vis_params["wheel_radius"])
        rot = R.from_euler("z", se2[2])
        body_dims = np.array([self.vis_params["length"], self.vis_params["width"], self.vis_params["height"]])
        artists.extend(draw_box(ax, body_center, rot.as_quat(), body_dims, self.vis_params["color"]))

        # Draw wheels
        for l in (-self.lr, self.lf):
            for j in (-1, 1):
                # Rotate about the center of the body, not the origin!
                wheel_center = np.array([l, j*0.5*(self.vis_params["width"] + self.vis_params["wheel_width"]), self.vis_params["wheel_radius"]])
                wheel_center = rot.apply(wheel_center)
                wheel_center += np.append(body_center[:2], 0.)
                # Cylinders take forever to plot
                # wheel_axis = rot.apply(np.array([0., 1., 0.]))
                # wheel_dims = np.array([self.vis_params["wheel_width"], self.vis_params["wheel_radius"]])
                # artists.extend(draw_cylinder(ax, wheel_center, wheel_axis, wheel_dims, self.vis_params["color"]))
                wheel_dims = np.array([2.0*self.vis_params["wheel_radius"], self.vis_params["wheel_width"], 2.0*self.vis_params["wheel_radius"]])
                artists.extend(draw_box(ax, wheel_center, rot.as_quat(), wheel_dims, self.vis_params["color"]))

        return artists


class Bicycle(Vehicle):
    # @input name [str]: vehicle name
    # @input lf [float]: length from the front axle to the center of gravity
    # @input lr [float]: length from the rear axle to the center of gravity
    # @input vis_params [Dict("length": float, "width": float, "height": float,
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
            VarDescription("x", "real", "x position", "m"),
            VarDescription("y", "real", "y position", "m"),
            VarDescription("psi", "circle", "yaw", "rad"),
            VarDescription("v", "real", "velocity", "m/s")
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("a", "real", "acceleration", "m/s^2"),
            VarDescription("delta", "circle", "steering angle", "rad")
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
    def get_pose_se3(cls, s: torch.tensor) -> torch.tensor:
        return torch.cat((s[:,:2], torch.zeros((s.size(0), 1)), s[:,2].unsqueeze(1), torch.zeros((s.size(0), 2))), dim=1)

    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        se2 = self.get_pose_se2(s.unsqueeze(0)).squeeze().numpy()
        anchor = se2[:2] - 0.5*np.array([self.vis_params["length"], self.vis_params["width"]])
        patch = Rectangle(anchor, self.vis_params["length"], self.vis_params["width"],
                          angle=np.degrees(se2[2]), rotation_point=tuple(se2[:2].tolist()),
                          color=self.vis_params["color"])
        ax.add_patch(patch)
        return [patch]

    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw body
        se2 = self.get_pose_se2(s.unsqueeze(0)).squeeze().numpy()
        body_center = np.append(se2[:2], 0.5*self.vis_params["height"] + 0.5*self.vis_params["wheel_radius"])
        rot = R.from_euler("z", se2[2])
        body_dims = np.array([self.vis_params["length"], self.vis_params["width"], self.vis_params["height"]])
        artists.extend(draw_box(ax, body_center, rot.as_quat(), body_dims, self.vis_params["color"]))

        # Draw wheels
        for l in (-self.lr, self.lf):
            for j in (-1, 1):
                # Rotate about the center of the body, not the origin!
                wheel_center = np.array([l, j*0.5*(self.vis_params["width"] + self.vis_params["wheel_width"]), self.vis_params["wheel_radius"]])
                wheel_center = rot.apply(wheel_center)
                wheel_center += np.append(body_center[:2], 0.)
                # Cylinders take forever to plot
                # wheel_axis = rot.apply(np.array([0., 1., 0.]))
                # wheel_dims = np.array([self.vis_params["wheel_width"], self.vis_params["wheel_radius"]])
                # artists.extend(draw_cylinder(ax, wheel_center, wheel_axis, wheel_dims, self.vis_params["color"]))
                wheel_dims = np.array([2.0*self.vis_params["wheel_radius"], self.vis_params["wheel_width"], 2.0*self.vis_params["wheel_radius"]])
                artists.extend(draw_box(ax, wheel_center, rot.as_quat(), wheel_dims, self.vis_params["color"]))

        return artists


class Quadrotor(Vehicle):
    # @input name [str]: vehicle name
    # @input m [float]: vehicle mass
    # @input inertia [torch.tensor (3)]: vehicle inertia values (I_x, I_y, I_z)
    # @input g [float]: acceleration from gravity
    # @input vis_params [Dict("side_length": float, "height": float, "prop_radius": float, "prop_height": float, "color": str)]:
    #     paramters for vehicle visualization (side length, height, propeller radius, propeller height, color)
    # @input s0 [torch.tensor (state_dim)]: initial vehicle state
    def __init__(self, name: str, m: float, inertia: torch.tensor, vis_params: Dict, s0: torch.tensor, g: float = 9.81):
        super().__init__(name, s0, vis_params)
        self.m = m
        self.inertia = inertia
        self.g = g

    @classmethod
    def get_state_description(cls) -> List[VarDescription]:
        return [
            VarDescription("x_ned", "real", "x position (NED)", "m"),
            VarDescription("y_ned", "real", "y position (NED)", "m"),
            VarDescription("z_ned", "real", "z position (NED)", "m"),
            VarDescription("alpha", "circle", "z rotation (for ZYX Euler angles)", "rad"),
            VarDescription("beta", "circle", "y rotation (for ZYX Euler angles)", "rad"),
            VarDescription("gamma", "circle", "x rotation (for ZYX Euler angles)", "rad"),
            VarDescription("dx_ned", "real", "x velocity (NED)", "m/s"),
            VarDescription("dy_ned", "real", "y velocity (NED)", "m/s"),
            VarDescription("dz_ned", "real", "z velocity (NED)", "m/s"),
            VarDescription("dalpha", "real", "z angular velocity (body frame)", "rad/s"),
            VarDescription("dbeta", "real", "y angular velocity (body frame)", "rad/s"),
            VarDescription("dgamma", "real", "x angular velocity (body frame)", "rad/s")
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("f_t", "real", "vertical thrust", "N"),
            VarDescription("tau_x", "real", "torque about x", "N"),
            VarDescription("tau_y", "real", "torque about y", "N"),
            VarDescription("tau_z", "real", "torque about z", "N")
        ]

    @classmethod
    def get_state_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0, 3, 6, 9],
            [1, 4, 7, 10],
            [2, 5, 8, 11]
        ])

    @classmethod
    def get_control_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0],
            [1],
            [2],
            [3]
        ])

    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # This should be fine since indexing returns views, not copies!
        alpha = s[:,3] # psi
        beta = s[:,4] # theta
        gamma = s[:,5] # phi
        dx = s[:,6]
        dy = s[:,7]
        dz = s[:,8]
        dalpha = s[:,9] # r
        dbeta = s[:,10] # q
        dgamma = s[:,11] # p
        f_t = u[:,0]
        tau_x = u[:,1]
        tau_y = u[:,2]
        tau_z = u[:,3]

        ds = torch.zeros_like(s)
        ds[:,0] = dx
        ds[:,1] = dy
        ds[:,2] = dz
        ds[:,3] = dbeta * torch.sin(gamma) / torch.cos(beta) + dgamma * torch.cos(gamma) / torch.cos(beta)
        ds[:,4] = dbeta * torch.cos(gamma) - dalpha * torch.sin(gamma)
        ds[:,5] = dalpha + dbeta * torch.sin(gamma) * torch.tan(beta) + dgamma * torch.cos(gamma) * torch.tan(beta)
        ds[:,6] = -(torch.sin(gamma) * torch.sin(alpha) + torch.cos(gamma) * torch.cos(alpha) * torch.sin(beta)) * f_t / self.m
        # Mistake in the paper???? They seem to negate the expression from a previous step for no reason
        # ds[:,7] = -(torch.cos(alpha) * torch.sin(gamma) - torch.cos(gamma) * torch.sin(alpha) * torch.sin(beta)) * f_t / self.m
        ds[:,7] = -(torch.cos(gamma) * torch.sin(alpha) * torch.sin(beta) - torch.cos(alpha) * torch.sin(gamma)) * f_t / self.m
        ds[:,8] = self.g - (torch.cos(gamma) * torch.cos(beta)) * f_t / self.m
        ds[:,9] = ((self.inertia[0] - self.inertia[1]) * dgamma * dbeta + tau_z) / self.inertia[2]
        ds[:,10] = ((self.inertia[2] - self.inertia[0]) * dgamma * dalpha + tau_y) / self.inertia[1]
        ds[:,11] = ((self.inertia[1] - self.inertia[2]) * dbeta * dalpha + tau_x) / self.inertia[0]
        return ds

    @classmethod
    def get_pose_se3(cls, s: torch.tensor) -> torch.tensor:
        # Converting from NED to our standard axes
        return ned_to_nwu(s[:,:6])

    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        artists = []
        se2 = self.get_pose_se2(s.unsqueeze(0)).squeeze().numpy()

        # Draw structure
        anchor = se2[:2] - 0.5*np.array([self.vis_params["side_length"], 0.1*self.vis_params["side_length"]])
        patch = Rectangle(anchor, self.vis_params["side_length"], 0.1*self.vis_params["side_length"],
                          angle=np.degrees(se2[2]), rotation_point=tuple(se2[:2].tolist()),
                          color=self.vis_params["color"])
        artists.append(ax.add_patch(patch))

        anchor = se2[:2] - 0.5*np.array([0.1*self.vis_params["side_length"], self.vis_params["side_length"]])
        patch = Rectangle(anchor, 0.1*self.vis_params["side_length"], self.vis_params["side_length"],
                          angle=np.degrees(se2[2]), rotation_point=tuple(se2[:2].tolist()),
                          color=self.vis_params["color"])
        artists.append(ax.add_patch(patch))

        # Draw propellers
        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            prop_center = se2[:2] + np.array([i,0.])
            patch = Circle(prop_center, radius=self.vis_params["prop_radius"], color=self.vis_params["color"])
            artists.append(ax.add_patch(patch))

        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            prop_center = se2[:2] + np.array([0.,i])
            patch = Circle(prop_center, radius=self.vis_params["prop_radius"], color=self.vis_params["color"])
            artists.append(ax.add_patch(patch))

        return artists

    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw body
        se3 = self.get_pose_se3(s.unsqueeze(0)).squeeze().numpy()
        # Converting from NED Euler angles to NWU
        rot = R.from_euler("ZYX", se3[3:].tolist())
        body_dims = np.array([0.1*self.vis_params["side_length"], self.vis_params["side_length"], self.vis_params["height"]])
        artists.extend(draw_box(ax, se3[:3], rot.as_quat(), body_dims, self.vis_params["color"]))
        body_dims = np.array([self.vis_params["side_length"], 0.1*self.vis_params["side_length"], self.vis_params["height"]])
        artists.extend(draw_box(ax, se3[:3], rot.as_quat(), body_dims, self.vis_params["color"]))

        # Draw propellers
        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            # Rotate about the center of the body, not the origin!
            # prop_center = np.array([i, 0., 0.5*self.vis_params["height"]])
            prop_center = np.array([i, 0., 0.5*(self.vis_params["height"] + self.vis_params["prop_height"])])
            prop_center = rot.apply(prop_center)
            prop_center += se3[:3]
            # Issues with transparent cylinder!
            # prop_axis = rot.apply(np.array([0., 0., 1.]))
            # prop_dims = np.array([self.vis_params["prop_height"], self.vis_params["prop_radius"]])
            # artists.extend(draw_cylinder(ax, prop_center, prop_axis, prop_dims, self.vis_params["color"]))
            prop_dims = np.array([2.0*self.vis_params["prop_radius"], 2.0*self.vis_params["prop_radius"], self.vis_params["prop_height"]])
            artists.extend(draw_box(ax, prop_center, rot.as_quat(), prop_dims, self.vis_params["color"]))

        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            # Rotate about the center of the body, not the origin!
            # prop_center = np.array([i, 0., 0.5*self.vis_params["height"]])
            prop_center = np.array([0., i, 0.5*(self.vis_params["height"] + self.vis_params["prop_height"])])
            prop_center = rot.apply(prop_center)
            prop_center += se3[:3]
            # Issues with transparent cylinder!
            # prop_axis = rot.apply(np.array([0., 0., 1.]))
            # prop_dims = np.array([self.vis_params["prop_height"], self.vis_params["prop_radius"]])
            # artists.extend(draw_cylinder(ax, prop_center, prop_axis, prop_dims, self.vis_params["color"]))
            prop_dims = np.array([2.0*self.vis_params["prop_radius"], 2.0*self.vis_params["prop_radius"], self.vis_params["prop_height"]])
            artists.extend(draw_box(ax, prop_center, rot.as_quat(), prop_dims, self.vis_params["color"]))

        return artists


class LinearizedQuadrotor(Vehicle):
    # @input name [str]: vehicle name
    # @input m [float]: vehicle mass
    # @input inertia [torch.tensor (3)]: vehicle inertia values (I_x, I_y, I_z)
    # @input g [float]: acceleration from gravity
    # @input vis_params [Dict("side_length": float, "height": float, "prop_radius": float, "prop_height": float, "color": str)]:
    #     paramters for vehicle visualization (side length, height, propeller radius, propeller height, color)
    # @input s0 [torch.tensor (state_dim)]: initial vehicle state
    def __init__(self, name: str, m: float, inertia: torch.tensor, vis_params: Dict, s0: torch.tensor, g: float = 9.81):
        super().__init__(name, s0, vis_params)
        self.m = m
        self.inertia = inertia
        self.g = g

    @classmethod
    def get_state_description(cls) -> List[VarDescription]:
        return [
            VarDescription("x_ned", "real", "x position (NED)", "m"),
            VarDescription("y_ned", "real", "y position (NED)", "m"),
            VarDescription("z_ned", "real", "z position (NED)", "m"),
            VarDescription("alpha", "circle", "z rotation (for ZYX Euler angles)", "rad"),
            VarDescription("beta", "circle", "y rotation (for ZYX Euler angles)", "rad"),
            VarDescription("gamma", "circle", "x rotation (for ZYX Euler angles)", "rad"),
            VarDescription("u_ned", "real", "x velocity (body frame)", "m/s"),
            VarDescription("v_ned", "real", "y velocity (body frame)", "m/s"),
            VarDescription("w_ned", "real", "z velocity (body frame)", "m/s"),
            VarDescription("dalpha", "real", "z angular velocity (body frame)", "rad/s"),
            VarDescription("dbeta", "real", "y angular velocity (body frame)", "rad/s"),
            VarDescription("dgamma", "real", "x angular velocity (body frame)", "rad/s")
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("f_t", "real", "vertical thrust", "N"),
            VarDescription("tau_x", "real", "torque about x", "N"),
            VarDescription("tau_y", "real", "torque about y", "N"),
            VarDescription("tau_z", "real", "torque about z", "N")
        ]

    @classmethod
    def get_state_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0, 3, 6, 9],
            [1, 4, 7, 10],
            [2, 5, 8, 11]
        ])

    @classmethod
    def get_control_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0],
            [1],
            [2],
            [3]
        ])

    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # TODO: think about constraining the yaw and yaw rate to zero? as in webb2013_kinodynamic
        # This should be fine since indexing returns views, not copies!
        alpha = s[:,3] # psi
        beta = s[:,4] # theta
        gamma = s[:,5] # phi
        u_ned = s[:,6]
        v_ned = s[:,7]
        w_ned = s[:,8]
        dalpha = s[:,9] # r
        dbeta = s[:,10] # q
        dgamma = s[:,11] # p
        f_t = u[:,0]
        tau_x = u[:,1]
        tau_y = u[:,2]
        tau_z = u[:,3]

        ds = torch.zeros_like(s)
        ds[:,0] = u_ned
        ds[:,1] = v_ned
        ds[:,2] = w_ned
        ds[:,3] = dalpha
        ds[:,4] = dbeta
        ds[:,5] = dgamma
        ds[:,6] = -self.g * beta
        ds[:,7] = self.g * gamma
        # Another mistake in the paper? They omit gravity here
        ds[:,8] = self.g - f_t / self.m
        ds[:,9] = tau_z / self.inertia[2]
        ds[:,10] = tau_y / self.inertia[1]
        ds[:,11] = tau_x / self.inertia[0]
        return ds

    @classmethod
    def get_pose_se3(cls, s: torch.tensor) -> torch.tensor:
        # Converting from NED to our standard axes
        return ned_to_nwu(s[:,:6])

    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        artists = []
        se2 = self.get_pose_se2(s.unsqueeze(0)).squeeze().numpy()

        # Draw structure
        anchor = se2[:2] - 0.5*np.array([self.vis_params["side_length"], 0.1*self.vis_params["side_length"]])
        patch = Rectangle(anchor, self.vis_params["side_length"], 0.1*self.vis_params["side_length"],
                          angle=np.degrees(se2[2]), rotation_point=tuple(se2[:2].tolist()),
                          color=self.vis_params["color"])
        artists.append(ax.add_patch(patch))

        anchor = se2[:2] - 0.5*np.array([0.1*self.vis_params["side_length"], self.vis_params["side_length"]])
        patch = Rectangle(anchor, 0.1*self.vis_params["side_length"], self.vis_params["side_length"],
                          angle=np.degrees(se2[2]), rotation_point=tuple(se2[:2].tolist()),
                          color=self.vis_params["color"])
        artists.append(ax.add_patch(patch))

        # Draw propellers
        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            prop_center = se2[:2] + np.array([i,0.])
            patch = Circle(prop_center, radius=self.vis_params["prop_radius"], color=self.vis_params["color"])
            artists.append(ax.add_patch(patch))

        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            prop_center = se2[:2] + np.array([0.,i])
            patch = Circle(prop_center, radius=self.vis_params["prop_radius"], color=self.vis_params["color"])
            artists.append(ax.add_patch(patch))

        return artists

    def add_vis3d(self, ax: Axes3D, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw body
        se3 = self.get_pose_se3(s.unsqueeze(0)).squeeze().numpy()
        # Converting from NED Euler angles to NWU
        rot = R.from_euler("ZYX", se3[3:].tolist())
        body_dims = np.array([0.1*self.vis_params["side_length"], self.vis_params["side_length"], self.vis_params["height"]])
        artists.extend(draw_box(ax, se3[:3], rot.as_quat(), body_dims, self.vis_params["color"]))
        body_dims = np.array([self.vis_params["side_length"], 0.1*self.vis_params["side_length"], self.vis_params["height"]])
        artists.extend(draw_box(ax, se3[:3], rot.as_quat(), body_dims, self.vis_params["color"]))

        # Draw propellers
        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            # Rotate about the center of the body, not the origin!
            # prop_center = np.array([i, 0., 0.5*self.vis_params["height"]])
            prop_center = np.array([i, 0., 0.5*(self.vis_params["height"] + self.vis_params["prop_height"])])
            prop_center = rot.apply(prop_center)
            prop_center += se3[:3]
            # Issues with transparent cylinder!
            # prop_axis = rot.apply(np.array([0., 0., 1.]))
            # prop_dims = np.array([self.vis_params["prop_height"], self.vis_params["prop_radius"]])
            # artists.extend(draw_cylinder(ax, prop_center, prop_axis, prop_dims, self.vis_params["color"]))
            prop_dims = np.array([2.0*self.vis_params["prop_radius"], 2.0*self.vis_params["prop_radius"], self.vis_params["prop_height"]])
            artists.extend(draw_box(ax, prop_center, rot.as_quat(), prop_dims, self.vis_params["color"]))

        for i in (-0.5*self.vis_params["side_length"], 0.5*self.vis_params["side_length"]):
            # Rotate about the center of the body, not the origin!
            # prop_center = np.array([i, 0., 0.5*self.vis_params["height"]])
            prop_center = np.array([0., i, 0.5*(self.vis_params["height"] + self.vis_params["prop_height"])])
            prop_center = rot.apply(prop_center)
            prop_center += se3[:3]
            # Issues with transparent cylinder!
            # prop_axis = rot.apply(np.array([0., 0., 1.]))
            # prop_dims = np.array([self.vis_params["prop_height"], self.vis_params["prop_radius"]])
            # artists.extend(draw_cylinder(ax, prop_center, prop_axis, prop_dims, self.vis_params["color"]))
            prop_dims = np.array([2.0*self.vis_params["prop_radius"], 2.0*self.vis_params["prop_radius"], self.vis_params["prop_height"]])
            artists.extend(draw_box(ax, prop_center, rot.as_quat(), prop_dims, self.vis_params["color"]))

        return artists