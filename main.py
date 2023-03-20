'''
Main procedure for Optimal Control final project
Author: rzfeng
'''
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import seaborn
seaborn.set()

from src.system import Unicycle, Bicycle, Quadrotor
from src.recorder import DataRecorder

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from src.vis_utils import equalize_axes3d


def main():
    quad1_vis_params = {
        "side_length": 30.0,
        "height": 10.0,
        "prop_radius": 10.0,
        "prop_height": 5.0,
        "color": "deepskyblue"
    }
    quad1_s0 = torch.zeros(Quadrotor.state_dim())
    quad1_s0[4] = -torch.pi/4
    quad1_s0[5] = -torch.pi/4
    quad1 = Quadrotor("quad1", 1.0, torch.tensor([0.3, 0.1, 0.5]), quad1_vis_params, quad1_s0)

    bike1_vis_params = {
        "length": 10.0,
        "width": 5.0,
        "height": 1.0,
        "wheel_radius": 0.4,
        "wheel_width": 0.3,
        "color": "lime"
    }
    bike1_s0 = torch.zeros(4)
    bike1 = Bicycle("bike1", 1.0, 1.0, bike1_vis_params, bike1_s0)

    recorder = DataRecorder([quad1, bike1])

    quad1_tf = 5.0
    quad1_u = torch.zeros((50, 4))
    quad1_u[:,0] = 30.0
    state_traj, timestamps = quad1.apply_control(quad1_u, (0.0, quad1_tf))
    recorder.log_state(quad1, state_traj, timestamps)
    recorder.log_control(quad1, quad1_u, torch.linspace(0.0, quad1_tf, quad1_u.size(0)))

    bike1_tf = 5.0
    bike1_u = torch.zeros((50, 2))
    bike1_u[:,0] = 20.0
    bike1_u[:,1] = np.pi/256
    state_traj, timestamps = bike1.apply_control(bike1_u, (0.0, bike1_tf))
    recorder.log_state(bike1, state_traj, timestamps)
    recorder.log_control(bike1, bike1_u, torch.linspace(0.0, bike1_tf, bike1_u.size(0)))

    recorder.animate3d([quad1], write="media/traj.mp4")
    # fig = plt.figure()
    # ax = Axes3D(fig, auto_add_to_figure=False, aspect="equal")
    # fig.add_axes(ax)
    # quad1.add_vis3d(ax, quad1.get_state())
    # equalize_axes3d(ax)
    # # ax.autoscale(False)
    # plt.show()


if __name__ == "__main__":
    main()