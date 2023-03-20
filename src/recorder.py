'''
Data recording and visualization
'''
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from src.vis_utils import equalize_axes3d

from src.system import Vehicle


# Class for a generic trajectory
@dataclass
class Trajectory:
    data: List[torch.tensor]
    timestamps: List[float]

    def as_np(self) -> Tuple:
        return torch.stack(self.data).numpy(), np.array(self.timestamps)


# Class for vehicle data
@dataclass
class VehicleTrajectory:
    state_traj: Trajectory
    control_traj: Trajectory


# Class for data recording and visualization
class DataRecorder:
    # @input vehicles [List[Vehicle]]: list of vehicles
    def __init__(self, vehicles: List[Vehicle]):
        self.data = {vehicle.name: VehicleTrajectory(Trajectory([],[]),Trajectory([],[])) for vehicle in vehicles}

    # Logs a batch of states
    # @input vehicle [Vehicle]: vehicle object
    # @input s [torch.tensor (T x state_dim)]: batch of states to add
    # @input t [torch.tensor (T)]: timestamps
    def log_state(self, vehicle: Vehicle, s: torch.tensor, t: torch.tensor):
        self.data[vehicle.name].state_traj.data.extend(list(s))
        self.data[vehicle.name].state_traj.timestamps.extend(t.tolist())

    # Logs a batch of controls
    # @input vehicle [Vehicle]: vehicle object
    # @input u [torch.tensor (T x control_dim)]: batch of controls to add
    # @input t [torch.tensor (T)]: timestamps
    def log_control(self, vehicle: Vehicle, u: torch.tensor, t: torch.tensor):
        self.data[vehicle.name].control_traj.data.extend(list(u))
        self.data[vehicle.name].control_traj.timestamps.extend(t.tolist())

    # Plots the state trajectory of a vehicle
    # @input vehicle [Vehicle]: vehicle object
    def plot_state_traj(self, vehicle: Vehicle):
        states, times = self.data[vehicle.name].state_traj.as_np()
        layout = vehicle.get_state_plot_layout()
        description = vehicle.get_state_description()

        fig, ax = plt.subplots(nrows=layout.shape[0], ncols=layout.shape[1], sharex=True, squeeze=False)
        fig.tight_layout()
        fig.supxlabel("time")
        fig.suptitle(f"{vehicle.name} state trajectory")
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                if layout[i,j] >= 0:
                    col.plot(times, states[:,layout[i,j]])
                    col.set_ylabel(description[layout[i,j]].name)

        plt.show()

    # Plots the control trajectory of a vehicle
    # @input vehicle [Vehicle]: vehicle object
    def plot_control_traj(self, vehicle: Vehicle):
        controls, times = self.data[vehicle.name].control_traj.as_np()
        layout = vehicle.get_control_plot_layout()
        description = vehicle.get_control_description()

        fig, ax = plt.subplots(nrows=layout.shape[0], ncols=layout.shape[1], sharex=True, squeeze=False)
        fig.tight_layout()
        fig.supxlabel("time")
        fig.suptitle(f"{vehicle.name} control trajectory")
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                if layout[i,j] >= 0:
                    col.plot(times, controls[:,layout[i,j]])
                    col.set_ylabel(description[layout[i,j]].name)

        plt.show()

    # Plots the 2D xy trajectory of vehicles
    # @input vehicles [List[Vehicle]]: vehicles to visualize
    def plot_traj2d(self, vehicles: List[Vehicle]):
        plt.figure()

        for vehicle in vehicles:
            states, times = self.data[vehicle.name].state_traj.as_np()
            pts = vehicle.get_pos2d(torch.from_numpy(states)).numpy()
            plt.plot(pts[:,0], pts[:,1], label=vehicle.name, color=vehicle.vis_params["color"])

        plt.title(f"2D trajectories")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    # Plots the 3D trajectory of vehicles
    # @input vehicles [List[Vehicle]]: vehicles to visualize
    def plot_traj3d(self, vehicles: List[Vehicle]):
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False, aspect="equal")
        fig.add_axes(ax)

        for vehicle in vehicles:
            states, times = self.data[vehicle.name].state_traj.as_np()
            pts = vehicle.get_pos3d(torch.from_numpy(states)).numpy()
            ax.plot(pts[:,0], pts[:,1], pts[:,2], label=vehicle.name, color=vehicle.vis_params["color"])

        equalize_axes3d(ax)
        ax.set_title(f"3D trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()

        plt.show()

    # Animates the 2D trajectory of vehicles
    # @input vehicles [List[Vehicle]]: vehicles to animate
    # @input hold_traj [bool]: maintain a line representing the 2D position trajectory
    # @input n_frames [Optional[int]]: number of frames to animate
    # @input fps [int]: frames per second
    # @input end_wait [float]: seconds to wait at the end before finishing the animation
    # @input write [Optional[str]]: filename to write the animation to; None indicates not to write
    def animate2d(self, vehicles: List[Vehicle], hold_traj: bool = True,
                  n_frames: Optional[int] = None, fps: int = 5, end_wait: float = 1.0, write: Optional[str] = None):
        fig = plt.figure()
        ax = plt.axes(aspect="equal")

        state_trajs = [self.data[vehicle.name].state_traj.as_np() for vehicle in vehicles]
        pos2ds = [vehicle.get_pos2d(torch.from_numpy(state_traj[0])).numpy() for vehicle, state_traj in zip(vehicles, state_trajs)]

        # Plot all patches to get the axis limits
        for vehicle, state_traj in zip(vehicles, state_trajs):
            for state in state_traj[0]:
                vehicle.add_vis2d(ax, torch.from_numpy(state))
        ax.autoscale_view()
        # Now clear the axes and set the axis limits
        lims = (ax.get_xlim(), ax.get_ylim())
        ax.clear()
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.autoscale(False)

        traj_lines = [ax.plot([],[], color=vehicle.vis_params["color"])[0] for vehicle in vehicles] if hold_traj else []
        vehicle_drawings = [[] for vehicle in vehicles]
        # Animation function for Matplotlib
        def anim_fn(t):
            for v_idx in range(len(vehicles)):
                # Get the appropriate index for the time point
                n = np.searchsorted(state_trajs[v_idx][1], t)
                if n >= state_trajs[v_idx][1].size:
                    continue

                # Draw trajectories
                if hold_traj:
                    traj_lines[v_idx].set_data(pos2ds[v_idx][:n,0], pos2ds[v_idx][:n,1])

                # Replace previous drawings
                for artist in vehicle_drawings[v_idx]:
                    artist.remove()
                vehicle_drawings[v_idx] = vehicles[v_idx].add_vis2d(ax, torch.from_numpy(state_trajs[v_idx][0][n,:]))

            return traj_lines + [artist for artists in vehicle_drawings for artist in artists]

        max_t = max([state_traj[1].max() for state_traj in state_trajs])
        n_frames = max([state_traj[1].size for state_traj in state_trajs]) if n_frames is None else n_frames
        end_buffer = int(np.ceil(end_wait * fps))
        frame_iter = np.append(np.linspace(0.0, max_t, n_frames), max_t * np.ones(end_buffer))
        anim = animation.FuncAnimation(fig, anim_fn, frames=frame_iter, interval = 1000.0 / fps, blit=True)

        legend_patches = []
        for vehicle in vehicles:
            legend_patches.append(Patch(color=vehicle.vis_params["color"], label=vehicle.name))
        ax.legend(handles=legend_patches, loc="right")
        ax.set_title(f"2D trajectory animation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if write is not None:
            anim.save(write)

        plt.show()

    # Animates the 3D trajectory of vehicles
    # TODO: investigate how to set camera angle
    # TODO: investigate why the live animation doesn't show vehicle bodies
    # @input vehicles [List[Vehicle]]: vehicles to animate
    # @input hold_traj [bool]: maintain a line representing the 2D position trajectory
    # @input n_frames [Optional[int]]: number of frames to animate
    # @input fps [int]: frames per second
    # @input end_wait [float]: seconds to wait at the end before finishing the animation
    # @input camera_angle Optional[Tuple[float] (3)]: camera angle (elevation, azimuth, roll)
    # @input write [Optional[str]]: filename to write the animation to; None indicates not to write
    def animate3d(self, vehicles: List[Vehicle], hold_traj: bool = True,
                  n_frames: Optional[int] = None, fps: int = 5, end_wait: float = 1.0, write: Optional[str] = None,
                  camera_angle: Optional[np.ndarray] = None):
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False, aspect="equal")
        fig.add_axes(ax)

        state_trajs = [self.data[vehicle.name].state_traj.as_np() for vehicle in vehicles]
        pos3ds = [vehicle.get_pos3d(torch.from_numpy(state_traj[0])).numpy() for vehicle, state_traj in zip(vehicles, state_trajs)]

        # Plot all patches to get the axis limits
        for vehicle, state_traj in zip(vehicles, state_trajs):
            for state in state_traj[0]:
                vehicle.add_vis3d(ax, torch.from_numpy(state))
        ax.autoscale_view()
        # Now clear the axes and set the axis limits
        lims = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
        ax.clear()
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
        equalize_axes3d(ax)
        ax.autoscale(False)
        if camera_angle is not None:
            ax.view_init(elev=camera_angle[0], azim=camera_angle[1], roll=camera_angle[2])

        traj_lines = [ax.plot([],[],[], color=vehicle.vis_params["color"])[0] for vehicle in vehicles] if hold_traj else []
        vehicle_drawings = [[] for vehicle in vehicles]
        # Animation function for Matplotlib
        def anim_fn(t):
            for v_idx in range(len(vehicles)):
                # Get the appropriate index for the time point
                n = np.searchsorted(state_trajs[v_idx][1], t)
                if n >= state_trajs[v_idx][1].size:
                    continue

                # Draw trajectories
                if hold_traj:
                    traj_lines[v_idx].set_data(pos3ds[v_idx][:n,0], pos3ds[v_idx][:n,1])
                    traj_lines[v_idx].set_3d_properties(pos3ds[v_idx][:n,2])

                # Replace previous drawings
                for artist in vehicle_drawings[v_idx]:
                    artist.remove()
                vehicle_drawings[v_idx] = vehicles[v_idx].add_vis3d(ax, torch.from_numpy(state_trajs[v_idx][0][n,:]))

            return traj_lines + [artist for artists in vehicle_drawings for artist in artists]

        max_t = max([state_traj[1].max() for state_traj in state_trajs])
        n_frames = max([state_traj[1].size for state_traj in state_trajs]) if n_frames is None else n_frames
        end_buffer = int(np.ceil(end_wait * fps))
        frame_iter = np.append(np.linspace(0.0, max_t, n_frames), max_t * np.ones(end_buffer))
        anim = animation.FuncAnimation(fig, anim_fn, frames=frame_iter, interval = 1000.0 / fps, blit=True)

        legend_patches = []
        for vehicle in vehicles:
            legend_patches.append(Patch(color=vehicle.vis_params["color"], label=vehicle.name))
        ax.legend(handles=legend_patches)
        ax.set_title(f"3D trajectory animation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if write is not None:
            anim.save(write)

        plt.show()