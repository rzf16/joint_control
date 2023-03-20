'''
Generic 3D visualization tools for Matplotlib
Author: rzfeng
'''
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Draws a 3D box on Matplotlib axes
# @input ax [Axes3D]: axes to visualize on
# @input center [np.ndarray (3)]: center of the box
# @input quat [np.ndarray (4)]: orientation of the box as a unit quaternion (x, y, z, w)
# @input dims [np.array (3)]: box length (x), width (y), height (z)
# @input color [str]: color
# @output [List[Artist]]: Matplotlib artists
def draw_box(ax: Axes3D, center: np.ndarray, quat: np.ndarray, dims: np.ndarray, color: str) -> List[Poly3DCollection]:
    artists = []

    # Generate permutations of the half-dimensions
    permuted_half_dims = np.tile(0.5*dims, (8,1))
    permuted_half_dims[:4,0] *= -1
    permuted_half_dims[::4,1] *= -1
    permuted_half_dims[1::4,1] *= -1
    permuted_half_dims[::2,2] *= -1

    # Rotate vertices about the center of the box (not the origin!)
    rot = R.from_quat(quat)
    vertices = center + rot.apply(permuted_half_dims)

    # Plot each face
    for i in range(3):
        x = vertices[permuted_half_dims[:,i] < 0, 0]
        y = vertices[permuted_half_dims[:,i] < 0, 1]
        z = vertices[permuted_half_dims[:,i] < 0, 2]
        artists.append(ax.plot_surface(x.reshape((2,2)), y.reshape((2,2)), z.reshape((2,2)), color=color, antialiased=False))

        x = vertices[permuted_half_dims[:,i] > 0, 0]
        y = vertices[permuted_half_dims[:,i] > 0, 1]
        z = vertices[permuted_half_dims[:,i] > 0, 2]
        artists.append(ax.plot_surface(x.reshape((2,2)), y.reshape((2,2)), z.reshape((2,2)), color=color, antialiased=False))

    return artists


# Draws a cylinder on Matplotlib axes
# @input ax [Axes3D]: axes to visualize on
# @input center [np.ndarray (3)]: center of the cylinder
# @input axis [np.ndarray (3)]: unit vector indicating the axis of the cylinder
# @input dims [np.array (2)]: cylinder height, radius
# @input color [str]: color
# @output [List[Artist]]: Matplotlib artists
def draw_cylinder(ax: Axes3D, center: np.ndarray, axis: np.ndarray, dims: np.ndarray, color: str) -> List[Poly3DCollection]:
    artists = []

    # Get unit vectors aligned with the axis
    axis /= np.linalg.norm(axis) # Just in case!
    not_axis = np.array([1.,0.,0.]) if (axis != np.array([1.,0.,0.])).any() else np.array([0.,1.,0.])
    v1 = np.cross(axis, not_axis)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    v2 /= np.linalg.norm(v2)

    # Get points
    n = 100
    h = np.linspace(-0.5*dims[0], 0.5*dims[0], n)
    theta = np.linspace(0.0, 2*np.pi, n)
    batch_h = h.repeat(n)
    batch_theta = np.tile(theta, n)
    pts = center + np.expand_dims(batch_h, 1) @ np.expand_dims(axis, 0) + \
          dims[1] * np.sin(np.expand_dims(batch_theta, 1)) @ np.expand_dims(v1, 0) + \
          dims[1] * np.cos(np.expand_dims(batch_theta, 1)) @ np.expand_dims(v2, 0)

    # TODO: issue with surface appearing transparent!
    artists.append(ax.plot_surface(pts[:,0].reshape((n,n)), pts[:,1].reshape((n,n)), pts[:,2].reshape((n,n)), color=color, antialiased=False))
    artists.append(ax.add_collection3d(Poly3DCollection([pts[:n,:], pts[-n:,:]], color=color)))

    return artists


# Sets an equal aspect ratio for Axes3D
# (from https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio)
#@input ax [Axes3D]: axes to equalize
def equalize_axes3d(ax: Axes3D):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)