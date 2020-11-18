import time
import math
import sys
import datetime
import numpy as np
from eulerangles import *
from utility import *
from plot_util import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Pos:
    def __init__(self):
        self.pred_transform_t_1 = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
        self.out_pred_array = []
        self.final_list = []
        self.sigma_list = []
        self.angle = 0
        self.arrw = [1, 0, 0]

pos = Pos()

fig1 = plt.figure()
ax1 = Axes3D(fig1) #fig1.add_subplot(1, 1, 1)
plt.ion()
fig1.show()
fig1.canvas.draw()


def parse_msg(ether_msg, len_pose=12):

    for idx in range(0, len(ether_msg), len_pose):
        final_pose = ether_msg[idx:idx+6]
        sigma_pose = ether_msg[idx+6:idx+12]

        #print(final_pose)
        #print(sigma_pose)
        pos.final_list.append(final_pose)
        pos.sigma_list.append(sigma_pose)

        pred_transform_t = convert_eul_to_matrix(0, 0, 0, final_pose)
        abs_pred_transform = np.dot(pos.pred_transform_t_1, pred_transform_t)
        pos.out_pred_array.append(
        [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
        abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
        abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2], abs_pred_transform[2, 3]])
        pos.pred_transform_t_1 = abs_pred_transform

    #print(pos.out_pred_array[-1])
    '''
    out_pred_array_np = np.array(pos.out_pred_array)
    out_gt_array_np = np.zeros(out_pred_array_np.shape)
    plot2d(out_pred_array_np, out_gt_array_np, join('traj.png'))
    '''

    euler_rot = np.array([[abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2]],
    [abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2]],
    [abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2]]], dtype=float)
    euler_rad = mat2euler(euler_rot)
    x = math.cos(euler_rad[0])*math.cos(euler_rad[1])
    y = math.sin(euler_rad[0])*math.cos(euler_rad[1])
    z = math.sin(euler_rad[1])

    ax1.clear()
    alpha_num = len(pos.final_list)
    if alpha_num > 100:
        print(alpha_num)
        alpha_num = 100
    alpha_deg = np.linspace(0.05, 1, alpha_num)
    #print(alpha_deg)

    for i in range(1, alpha_num):
        if i!=1:
            ax1.scatter([pos.final_list[-i][0]], [pos.final_list[-i][1]], [pos.final_list[-i][2]], color='b', marker='o', alpha=alpha_deg[-i])
        else:
            ax1.quiver([pos.final_list[-i][0]], [pos.final_list[-i][1]], [pos.final_list[-i][2]],
            x, y, z, color='r', length=0.3, alpha=alpha_deg[-i])

    ax1.view_init(elev=30., azim=pos.angle)
    pos.angle += 1.
    ax1.set_xlabel('X Axis (m)')
    ax1.set_ylabel('Y Axis (m)')
    ax1.set_zlabel('Z Axis (m)')

    ax1.set_xlim(-0, 1)
    ax1.set_ylim(-0, 1)
    ax1.set_zlim(-0, 1)

    fig1.canvas.draw()



