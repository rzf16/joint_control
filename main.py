'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import seaborn
seaborn.set()

from src.system import Unicycle, Bicycle, Quadrotor, LinearizedQuadrotor
from src.recorder import DataRecorder

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from src.vis_utils import equalize_axes3d


def main():
    linquad1_vis_params = {
        "side_length": 30.0,
        "height": 10.0,
        "prop_radius": 5.0,
        "prop_height": 5.0,
        "color": "deepskyblue"
    }
    linquad1_s0 = torch.zeros(Quadrotor.state_dim())
    linquad1_s0[4] = -torch.pi/8
    linquad1_s0[5] = -torch.pi/8
    linquad1 = Quadrotor("linquad1", 1.0, torch.tensor([0.3, 0.1, 0.5]), linquad1_vis_params, linquad1_s0)

    recorder = DataRecorder([linquad1])

    linquad1_tf = 5.0
    linquad1_u = torch.zeros((50, 4))
    linquad1_u[:,0] = 30.0
    state_traj, timestamps = linquad1.apply_control(linquad1_u, (0.0, linquad1_tf))
    recorder.log_state(linquad1, state_traj, timestamps)
    recorder.log_control(linquad1, linquad1_u, torch.linspace(0.0, linquad1_tf, linquad1_u.size(0)))

    recorder.animate2d([linquad1])


if __name__ == "__main__":
    main()