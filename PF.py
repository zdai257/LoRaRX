import os
from os.path import join
import time
from pfilter import (
    ParticleFilter,
    gaussian_noise,
    cauchy_noise,
    squared_error,
    independent_sample,
)
import numpy as np
import math
from eulerangles import *
from utility import *
from plot_util import *
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib import gridspec

# testing only
from scipy.stats import norm, gamma, uniform
import skimage.draw


# LoRa RX Coordinates
R1 = np.array([[0., 0., 0.],
               [15., 5., 0.],
               [30., -22., 0.],
               [5., 4., 0.],
               [2., -2.5, 0.]])
# Path Loss Model params
ALPHA = -55#-45.712  # -28.57 * 1.6
BETA = -5.06
SIGMA = 4.887


# prior sampling function for each variable
prior_fn = independent_sample(
    [
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.1).rvs,
        gamma(a=1, loc=0, scale=1).rvs,
        norm(loc=0, scale=0.01).rvs,
        norm(loc=0, scale=0.05).rvs,
    ]
)


def measurement(X):
    """Given an Nx(3 + RSSIs) matrix of derived measurements,
    compute measurement of [TRANS_X, TRANS_Y, ROT_Z, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    anchor = 1
    Z = np.zeros((X.shape[0], 3 + anchor))
    for i, x in enumerate(X):
        trans_x = dt * math.sqrt(x[2] ** 2 + x[3] ** 2) * math.cos(dt * x[4])
        trans_y = dt * math.sqrt(x[2] ** 2 + x[3] ** 2) * math.sin(dt * x[4])
        rot_z = dt * x[4]
        h = np.array([trans_x, trans_y, rot_z]).reshape((-1, 1))
        for row in range(0, anchor):
            dis = np.linalg.norm(x[:2] - R1[row, :2])
            thres_dis = 1.
            if dis > thres_dis:
                rssi = ALPHA * math.log10(dis) + BETA
            else:
                rssi = ALPHA * math.log10(thres_dis) + BETA
            # Measurement comprises (X, Y, Rot_Z, RSSIs...)
            h = np.vstack((h, np.array([[rssi]])))
        # print("hx return shape: ", h.shape)
        Z[i] = h.reshape((1, -1))
    return Z


# motion dynamics
def constantAW(x):
    dt = 0.1
    xp = (
            x
            @ np.array(
        [
            [0, 0, dt * x[3, 0] * math.sin(x[2, 0]) - 0.5 * dt ** 2 * x[5, 0] * math.sin(x[2, 0]),
             dt * math.cos(x[2, 0]), 0, 0.5 * dt ** 2 * math.cos(x[2, 0])],
            [0, 0, dt * x[3, 0] * math.cos(x[2, 0]) + 0.5 * dt ** 2 * x[5, 0] * math.sin(x[2, 0]),
             dt * math.sin(x[2, 0]), 0, 0.5 * dt ** 2 * math.sin(x[2, 0])],
            [0, 0, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, dt],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
    ).T
    )

    return xp


# names (this is just for reference for the moment!)
columns = ["x", "y", "theta", "v", "w", "a"]


class PF_Fusion():
    def __init__(self, dt=0.1, anchor=1, num_particle=200, resample_rate=0.2, blit=True, visual=False):
        self.dt =dt
        self.anchor = anchor
        self.visual = visual
        # create the Particle Filter
        self.pf = ParticleFilter(
            prior_fn=prior_fn,
            observe_fn=measurement,
            n_particles=num_particle,
            dynamics_fn=constantAW,
            noise_fn=lambda x: cauchy_noise(x, sigmas=[0.05, 0.05, 0.01, 0.05, 0.001, 0.005]),
            weight_fn=lambda x, y: squared_error(x, y, sigma=2),
            resample_proportion=resample_rate,
            column_names=columns,
        )

        # GT
        self.pred_transform_t_1 = np.array(
            [[1., 0, 0, 0],
             [0, 1., 0, 0],
             [0, 0, 1., 0],
             [0, 0, 0, 1.]])
        self.out_pred_array = []
        self.final_list = []
        self.sigma_list = []
        self.rssi_list, self.rssi_list2, self.rssi_list3 = [], [], []

        # Visualisation init
        self.blit = blit
        self.handle_scat = None
        self.handle_arrw = None
        self.handle_scat_ekf = None
        self.handle_arrw_ekf = None
        self.ax1background = None
        self.ax2background = None

        self.fig2 = plt.figure(figsize=(8, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        self.ax21 = self.fig2.add_subplot(gs[0], projection='3d')
        self.ax22 = self.fig2.add_subplot(gs[1])
        # self.ax21 = self.fig2.add_subplot(2, 1, 1, projection='3d') #fig1.gca(projection='3d') #Axes3D(fig1)
        # self.ax22 = self.fig2.add_subplot(2, 1, 2)
        plt.ion()
        plt.tight_layout()
        #self.set_view()

        self.fig2.canvas.draw()
        if self.blit:
            self.ax1background = self.fig2.canvas.copy_from_bbox(self.ax21.bbox)
            self.ax2background = self.fig2.canvas.copy_from_bbox(self.ax22.bbox)
        if self.visual:
            plt.show(block=False)

        # logging
        self.xs = []
        self.track = []
        self.time = []
        self.path = []
        self.path_pf = []
        self.abs_yaw = 0



    def new_measure(self, *args, **kwargs):
        start_t = time.time()
        len_pose = 12
        gap = 0
        msg_list = []
        rssis = []
        for arg in args:
            msg_list.append(arg)
        if self.anchor:
            rssis = msg_list[-self.anchor:]
            msg_list = msg_list[:-self.anchor]

        self.rssi_list.extend(rssis)
        if self.anchor >= 2:
            self.rssi_list2.append(rssis[1])
        if self.anchor >= 3:
            self.rssi_list3.append(rssis[2])

        for idx in range(0, len(msg_list), len_pose):
            final_pose = msg_list[idx:idx + 6]
            sigma_pose = msg_list[idx + 6:idx + 12]
            # print(final_pose)
            # print(sigma_pose)
            self.final_list.append(final_pose)
            self.sigma_list.append(sigma_pose)

            pred_transform_t = convert_eul_to_matrix(0, 0, 0, final_pose)
            abs_pred_transform = np.dot(self.pred_transform_t_1, pred_transform_t)
            self.out_pred_array.append(
                [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
                 abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
                 abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2],
                 abs_pred_transform[2, 3]])
            self.pred_transform_t_1 = abs_pred_transform

            # pos.odom_quat = tf.transformations.quaternion_from_matrix(pos.pred_transform_t_1)
            # print(pos.odom_quat)
        gap = int(len(msg_list) / len_pose)
        # print(self.out_pred_array[-1])

        euler_rot = np.array([[abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2]],
                              [abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2]],
                              [abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2]]],
                             dtype=float)
        # euler_rad = (yaw, pitch, roll)
        euler_rad = mat2euler(euler_rot)
        self.abs_yaw = euler_rad[0]
        # print("Current Eular = ", euler_rad)
        self.odom_quat = np.array(euler2quat(euler_rad[0], euler_rad[1], euler_rad[2]))
        # print("Current Quaternion = ", self.odom_quat)

        # Unit Vector from Eular Angle; Simplify Orientation Representation by Pitch = 0
        self.U = math.cos(self.abs_yaw)  # math.cos(euler_rad[0])*math.cos(euler_rad[1])
        self.V = math.sin(self.abs_yaw)  # math.sin(euler_rad[0])*math.cos(euler_rad[1])
        self.W = 0  # math.sin(euler_rad[1])

        self.path.append([abs_pred_transform[0, 3], abs_pred_transform[1, 3], 0])

        print("Elapsed time PoseTransform = ", time.time() - start_t)
        start_t = time.time()

        # Trigger PF
        self.rt_run(gap)
        print("Elapsed time of PF = ", time.time() - start_t)
        print("ABS_YAW: %.3f (=%.3f OR %.3f)" % (self.abs_yaw, self.abs_yaw - 2 * math.pi, self.abs_yaw - 4 * math.pi))
        print("State X:\n", self.pf.mean_state)
        '''
        if self.visual:
            self.rt_show()
        '''


    def rt_run(self, gap):

        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            final_xyZ = final_pose[:2]
            # Add ROT_Z, convert from 'degree' to 'Rad', as measurement!
            final_xyZ.append(final_pose[-1] * math.pi/180)
            # Populate ONE Rssi for a 'gap' of Poses
            if self.anchor:
                final_xyZ.append(self.smoother(self.rssi_list))
            if self.rssi_list2:
                final_xyZ.append(self.smoother(self.rssi_list2))
            if self.rssi_list3:
                final_xyZ.append(self.smoother(self.rssi_list3))

            z = np.asarray(final_xyZ, dtype=float).reshape(1, -1)
            #print("Measurement z: ", z)
            # TODO Add data integraty check: X+ value explodes

            # Feed Observation to Update
            self.pf.update(z)

            # Log Posterior State x
            self.xs.append(self.pf.mean_state)
            #print("X+:\n", self.pf.mean_state)
            # print("EKF per round takes %.6f s" % (time.time() - start_t))


    def smoother(self, lst):
        if len(lst) > 3:
            ave_rssi = 0.6 * lst[-1] + 0.25*lst[-2] + 0.15*lst[-3]
        elif len(lst) == 0:
            return []
        else:
            ave_rssi = lst[-1]
        return float(ave_rssi)



if __name__ == "__main__":

    fuse_engine = PF_Fusion(anchor=1, num_particle=500, resample_rate=0.2)

    for filename in os.listdir('TEST'):
        if filename.endswith('.txt'):
            with open('TEST/' + filename, "r") as f:
                recv_list = f.readlines()

            # Add synthetic RSSIs
            data_len = len(recv_list)
            if 0:
                rssi_y2 = synthetic_rssi(data_len=data_len, period=1)
                rssi_y3 = synthetic_rssi(data_len=data_len, period=1, Amp=15, phase=-math.pi / 2, noiseAmp=0.3,
                                         mean=-45)
            else:
                rssi_y2, rssi_y3 = [], []

            rssi_idx = 0

            for item in recv_list:
                rssi_list = []

                parts = item.split(';')
                t = parts[0]
                msgs = parts[1]
                vals = msgs.split(',')
                rssi1 = int(parts[2])
                # Append RXs measurements
                rssi_list.append(rssi1)
                if rssi_y2:
                    rssi_list.append(rssi_y2[rssi_idx])
                if rssi_y3:
                    rssi_list.append(rssi_y3[rssi_idx])
                rssi_idx += 1

                msg_list = [float(i) for i in vals]
                if fuse_engine.anchor:
                    msg_list.extend(rssi_list)

                fuse_engine.new_measure(*msg_list)

                # plt.pause(0.01)
                time.sleep(.05)

