'''
Data recording and visualization
Author: rzfeng
'''
import os
import shutil
import yaml
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch, Circle
from mpl_toolkits.mplot3d import Axes3D
from src.vis_utils import equalize_axes3d

from src.vehicles import Vehicle


DATA_PATH = "data/"
MEDIA_DIR = "media/"


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

    # Writes recorded data to disk as NumPy arrays
    # @input cfg_path [str]: path to configuration file
    # @input vehicles [List[Vehicle]]: vehicles to write data for
    # @input prefix [str]: data directory prefix (e.g. "run" -> run1, run2, etc.)
    def write_data(self, cfg_path: str, vehicles: List[Vehicle], prefix: str = "run"):
        if not vehicles:
            return

        # Get the current run number and make the corresponding directory
        idx = 1
        write_dir = os.path.join(DATA_PATH, prefix+f"{idx:03}")
        while os.path.exists(write_dir):
            idx += 1
            write_dir = os.path.join(DATA_PATH, prefix+f"{idx:03}")
        os.makedirs(write_dir)

        # Copy the configuration info
        shutil.copy(cfg_path, os.path.join(write_dir, "cfg.yaml"))

        # Write data as NumPy arrays
        vehicle_names = [vehicle.name for vehicle in vehicles]
        for vehicle_name in vehicle_names:
            states, state_times = self.data[vehicle_name].state_traj.as_np()
            controls, control_times = self.data[vehicle_name].state_traj.as_np()
            np.save(os.path.join(write_dir, f"{vehicle_name}_states.npy"), states)
            np.save(os.path.join(write_dir, f"{vehicle_name}_state_times.npy"), state_times)
            np.save(os.path.join(write_dir, f"{vehicle_name}_controls.npy"), controls)
            np.save(os.path.join(write_dir, f"{vehicle_name}_control_times.npy"), control_times)

        print(f"[Recorder] Data written to {write_dir}!")

    # Loads trajectory data from a directory
    # @input dir [str]: directory to load from (EXCLUDING "data/")
    def from_data(self, dir: str):
        # Grab vehicle names from the copied configuration
        load_dir = os.path.join(DATA_PATH, dir)
        cfg = yaml.safe_load(open(os.path.join(load_dir, "cfg.yaml")))
        vehicle_names = [vehicle["name"] for vehicle in cfg["vehicles"]] # Is this robust? Maybe use substrings instead?

        # Load data from NumPy arrays
        for vehicle_name in vehicle_names:
            states = np.load(os.path.join(load_dir, f"{vehicle_name}_states.npy"))
            state_times = np.load(os.path.join(load_dir, f"{vehicle_name}_state_times.npy"))
            controls = np.load(os.path.join(load_dir, f"{vehicle_name}_controls.npy"))
            control_times = np.load(os.path.join(load_dir, f"{vehicle_name}_control_times.npy"))

            states = [torch.from_numpy(state) for state in states]
            state_times = state_times.tolist()
            controls = [torch.from_numpy(control) for control in controls]
            control_times = control_times.tolist()

            self.data[vehicle_name] = VehicleTrajectory(Trajectory(states, state_times), Trajectory(controls, control_times))

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
    # @input obstacles [List[Tuple[float]]]: cylindrical obstacles (x, y, radius, height)
    def plot_traj2d(self, vehicles: List[Vehicle], obstacles: List[Tuple[float]] = []):
        if not vehicles:
            return

        plt.figure()

        for vehicle in vehicles:
            states, times = self.data[vehicle.name].state_traj.as_np()
            pts = vehicle.get_pos2d(torch.from_numpy(states)).numpy()
            plt.plot(pts[:,0], pts[:,1], label=vehicle.name, color=vehicle.vis_params["color"])

        for obs in obstacles:
            plt.Circle(obs[:2], obs[2], color='k')

        plt.title(f"2D trajectories")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    # Plots the 3D trajectory of vehicles
    # @input vehicles [List[Vehicle]]: vehicles to visualize
    def plot_traj3d(self, vehicles: List[Vehicle]):
        if not vehicles:
            return

        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False, aspect="equal")
        fig.add_axes(ax)

        for vehicle in vehicles:
            states, times = self.data[vehicle.name].state_traj.as_np()
            pts = vehicle.get_pos3d(torch.from_numpy(states)).numpy()
            ax.plot(pts[:,0], pts[:,1], pts[:,2], label=vehicle.name, color=vehicle.vis_params["color"])

        # TODO: draw obstacles

        equalize_axes3d(ax)
        ax.set_title(f"3D trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()

        plt.show()

    # Animates the 2D trajectory of vehicles
    # @input vehicles [List[Vehicle]]: vehicles to animate
    # @input obstacles [List[Tuple[float]]]: cylindrical obstacles (x, y, radius, height)
    # @input hold_traj [bool]: maintain a line representing the 2D position trajectory
    # @input n_frames [Optional[int]]: number of frames to animate
    # @input fps [int]: frames per second
    # @input end_wait [float]: seconds to wait at the end before finishing the animation
    # @input write [Optional[str]]: filename to write the animation to (EXCLUDING "media/""); None indicates not to write
    def animate2d(self, vehicles: List[Vehicle], obstacles: List[Tuple[float]] = [], hold_traj: bool = True,
                  n_frames: Optional[int] = None, fps: int = 5, end_wait: float = 1.0, write: Optional[str] = None):
        if not vehicles:
            return

        # TODO: move the legend off the plot
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

        for obs in obstacles:
            ax.add_patch(Circle(obs[:2], obs[2], color='k'))

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
        ax.legend(handles=legend_patches, loc="upper left")
        ax.set_title(f"2D trajectory animation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if write is not None:
            write_path = os.path.join(MEDIA_DIR, write)
            overwrite = True
            if os.path.exists(write_path):
                overwrite = input(f"[Recorder] Write path {write_path} exists. Overwrite? (y/n) ") == "y"
            if overwrite:
                anim.save(os.path.join(MEDIA_DIR, write))

        plt.show()

    # Animates the 3D trajectory of vehicles
    # TODO: investigate why the live animation doesn't show vehicle bodies
    # @input vehicles [List[Vehicle]]: vehicles to animate
    # @input hold_traj [bool]: maintain a line representing the 2D position trajectory
    # @input n_frames [Optional[int]]: number of frames to animate
    # @input fps [int]: frames per second
    # @input end_wait [float]: seconds to wait at the end before finishing the animation
    # @input camera_angle Optional[Tuple[float] (3)]: camera angle (elevation, azimuth, roll)
    # @input write [Optional[str]]: filename to write the animation to (EXCLUDING "media/""); None indicates not to write
    def animate3d(self, vehicles: List[Vehicle], hold_traj: bool = True,
                  n_frames: Optional[int] = None, fps: int = 5, end_wait: float = 1.0, write: Optional[str] = None,
                  camera_angle: Optional[np.ndarray] = None):
        if not vehicles:
            return

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

        # TODO: draw obstacles

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
            write_path = os.path.join(MEDIA_DIR, write)
            overwrite = True
            if os.path.exists(write_path):
                overwrite = input(f"[Recorder] Write path {write_path} exists. Overwrite? (y/n) ") == "y"
            if overwrite:
                anim.save(os.path.join(MEDIA_DIR, write))

        plt.show()