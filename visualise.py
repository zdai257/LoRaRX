import time
import sys
import datetime
import numpy as np
from eulerangles import *
from utility import *
from plot_util import *


class Pos:
    def __init__(self):
        self.pred_transform_t_1 = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
        self.out_pred_array = []

pos = Pos()

def parse_msg(ether_msg, len_pose=12):

    for idx in range(0, len(ether_msg), len_pose):
        final_pose = ether_msg[idx:idx+6]
        sigma_pose = ether_msg[idx+6:idx+12]

        print(final_pose)
        print(sigma_pose)

        pred_transform_t = convert_eul_to_matrix(0, 0, 0, final_pose)
        abs_pred_transform = np.dot(pos.pred_transform_t_1, pred_transform_t)
        pos.out_pred_array.append(
        [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
        abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
        abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2], abs_pred_transform[2, 3]])
        pos.pred_transform_t_1 = abs_pred_transform

    print(pos.out_pred_array)
    out_pred_array_np = np.array(pos.out_pred_array)
    out_gt_array_np = np.zeros(out_pred_array_np.shape)
    plot2d(out_pred_array_np, out_gt_array_np, join('traj.png'))



