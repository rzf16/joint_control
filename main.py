'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
import torch
import numpy as np

import seaborn
seaborn.set()

from src.system import Unicycle, Bicycle
from src.recorder import DataRecorder


def main():
    vis_params1 = {
        "length": 3.0,
        "width": 1.5,
        "height": 1.0,
        "wheel_radius": 0.4,
        "wheel_width": 0.3,
        "color": "deepskyblue"
    }
    vis_params2 = {
        "length": 3.0,
        "width": 1.5,
        "height": 1.0,
        "wheel_radius": 0.4,
        "wheel_width": 0.3,
        "color": "lime"
    }
    bike1 = Bicycle("bike1", 1.0, 1.0, vis_params1, torch.zeros(4))
    uni1 = Unicycle("uni1", vis_params2, torch.zeros(3))
    recorder = DataRecorder([bike1, uni1])

    tf = 5.0
    u = torch.zeros((50, 2))
    u[:,0] = 0.5
    u[:,1] = np.pi/16
    state_traj, timestamps = bike1.apply_control(u, (0.0, tf))
    recorder.log_state(bike1, state_traj, timestamps)
    recorder.log_control(bike1, u, torch.linspace(0.0, tf, u.size(0)))

    tf = 10.0
    u[:,1] = -np.pi/16
    state_traj, timestamps = uni1.apply_control(u, (0.0, tf))
    recorder.log_state(uni1, state_traj, timestamps)
    recorder.log_control(uni1, u, torch.linspace(0.0, tf, u.size(0)))

    recorder.animate2d([bike1, uni1], write="media/traj.mp4")


if __name__ == "__main__":
    main()