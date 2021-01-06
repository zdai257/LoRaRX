import os
from os.path import join
import time
from pfilter import (
    ParticleFilter,
    gaussian_noise,
    cauchy_noise,
    squared_error,
    independent_sample,
    stratified_resample,
    systematic_resample,
    residual_resample,
    multinomial_resample,
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
from scipy.signal import savgol_filter
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
        uniform(loc=-math.pi, scale=math.pi).rvs,  #norm(loc=0, scale=0.1).rvs,
        gamma(a=1, loc=0, scale=1).rvs,
        norm(loc=0, scale=0.01).rvs,
        norm(loc=0, scale=0.05).rvs,
    ]
)


def main():
    fuse_engine = PF_Fusion(anchor=1, num_particle=10000, resample_rate=0.005, visual=True)

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
                time.sleep(.001)

    fuse_engine.fig2.savefig("replay_pf.png")


def measurement(X, dt=0.1, anchor=1):
    """Given an Nx(3 + RSSIs) matrix of derived measurements,
    compute measurement of [TRANS_X, TRANS_Y, ROT_Z, RSSIs...]^T that would correspond to state x.
    """
    Z = np.zeros((X.shape[0], 3 + anchor))
    for i, x in enumerate(X):
        V = x[3]
        W = x[4]
        trans_x = dt * V * math.cos(dt * W)
        trans_y = dt * V * math.sin(dt * W)
        rot_z = dt * W
        h = np.array([trans_x, trans_y, rot_z]).reshape((-1, 1))
        for row in range(0, anchor):
            dis = np.linalg.norm(x[:2] - R1[row, :2])
            thres_dis = 0.1
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
def constantAW(X, dt=0.1):
    # Still Using the 1st order Taylor Series of ConstantA model as State Transition!?
    xp = np.zeros(X.shape)
    for i, x in enumerate(X):
        '''
        xp[i] = (
                x
                @ np.array(
            [
                [1, 0, dt * x[3] * math.sin(x[2]) - 0.5 * dt ** 2 * x[5] * math.sin(x[2]),
                 dt * math.cos(x[2]), 0, 0.5 * dt ** 2 * math.cos(x[2])],
                [0, 1, dt * x[3] * math.cos(x[2]) + 0.5 * dt ** 2 * x[5] * math.sin(x[2]),
                 dt * math.sin(x[2]), 0, 0.5 * dt ** 2 * math.sin(x[2])],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        ).T
        )
        '''
        F = np.array([[1, 0, dt * x[3] * math.sin(x[2]) - 0.5 * dt ** 2 * x[5] * math.sin(x[2]),
                 dt * math.cos(x[2]), 0, 0.5 * dt ** 2 * math.cos(x[2])],
                [0, 1, dt * x[3] * math.cos(x[2]) + 0.5 * dt ** 2 * x[5] * math.sin(x[2]),
                 dt * math.sin(x[2]), 0, 0.5 * dt ** 2 * math.sin(x[2])],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
        xp[i] = np.dot(F, x.reshape(-1, 1)).reshape((1, -1))
    #print(xp.shape)
    return xp


# names (this is just for reference for the moment!)
columns = ["x", "y", "theta", "v", "w", "a"]


class PF_Fusion():
    def __init__(self, dt=0.1, anchor=1, num_particle=200, resample_rate=0.2, blit=True, visual=False):
        self.dt = dt
        self.anchor = anchor
        self.visual = visual
        # create the Particle Filter
        self.pf = ParticleFilter(
            prior_fn=prior_fn,
            observe_fn=lambda x: measurement(x, self.dt, self.anchor),
            resample_fn=multinomial_resample,
            n_particles=num_particle,
            dynamics_fn=lambda x: constantAW(x, self.dt),
            noise_fn=lambda x: cauchy_noise(x, sigmas=[0.00001, 0.00001, 0.00000001, 0.0000001, 0.00000001, 0.0000005]),
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
        self.rssi_list, self.smoothed_rssi_list, self.rssi_list2, self.rssi_list3 = [], [], [], []

        # Visualisation init
        self.blit = blit
        self.handle_scat = None
        self.handle_arrw = None
        self.handle_scat_pf = None
        self.handle_arrw_pf = None
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
        self.set_view()

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
        self.smoothed_rssi_list.append(self.smoother(self.rssi_list))
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

        if self.visual:
            self.rt_show()


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
                final_xyZ.append(self.smoothed_rssi_list[-1])
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
            # print("PF per round takes %.6f s" % (time.time() - start_t))


    def smoother(self, lst, window_size=5, mode='conv'):

        if mode == 'conv':
            if len(lst) >= window_size:
                window = np.ones(int(window_size)) / float(window_size)
                y = np.convolve(np.asarray(lst), window, 'valid')
                #print(y)
                return y[-1]
            else:
                return lst[-1]
        elif mode == 'savgol':
            if len(lst) >= window_size:
                y = savgol_filter(np.asarray(lst), window_size, 3)
                #print(y)
                return y[-2]
            else:
                return lst[-1]
        else:
            if len(lst) > 3:
                ave_rssi = 0.6 * lst[-1] + 0.25 * lst[-2] + 0.15 * lst[-3]
            elif len(lst) == 0:
                return []
            else:
                ave_rssi = lst[-1]
            return float(ave_rssi)


    def rt_show(self, t_limit=0.85):
        start_t = time.time()
        # u, v, w = self.odom_quat[0], self.odom_quat[1], self.odom_quat[2]

        self.handle_scat.set_alpha(.2)
        self.handle_scat_pf.set_alpha(.2)
        self.handle_arrw.remove()
        # Remove Range Circle
        if self.anchor:
            self.cir1.remove()
        if self.rssi_list2:
            self.cir2.remove()
        if self.rssi_list3:
            self.cir3.remove()

        self.handle_scat = self.ax21.scatter([self.path[-1][0]], [self.path[-1][1]], [self.path[-1][2]], color='b',
                                             marker='o', alpha=.9, label='MIO')
        self.handle_arrw = self.ax21.quiver([self.path[-1][0]], [self.path[-1][1]], [self.path[-1][2]],
                                            self.U, self.V, self.W, color='b', length=2., arrow_length_ratio=0.3,
                                            linewidths=3., alpha=.7)
        self.handle_scat_pf = self.ax21.scatter([self.xs[-1][0]], [self.xs[-1][1]], [0.], color='r', marker='o',
                                                 alpha=.9, label='LoRa-MIO')
        # Not Attempting to Visual PF Updated Orientation
        # self.handle_arrw_pf = self.ax21.quiver([self.my_kf.x[0, 0]], [self.my_kf.x[1, 0]], [self.my_kf.x[2, 0]], self.U_pf, self.V_pf, self.W_pf, color='r', length=1., alpha=.7)
        # Manually Equal Axis and Limit
        self.ax21.auto_scale_xyz([-12, 18], [-18, 12], [-1, 3])

        # Plot Range
        if self.anchor:
            radius = 10 ** ((self.smoothed_rssi_list[-1] + BETA) / ALPHA)
            circle1 = plt.Circle((R1[0, 0], R1[0, 1]), radius, color='g', fill=False, alpha=.6, linewidth=0.5)
            self.cir1 = self.ax21.add_patch(circle1)
            art3d.pathpatch_2d_to_3d(circle1, z=0, zdir="z")
        if self.rssi_list2:
            radius = 10 ** ((self.rssi_list2[-1] + BETA) / ALPHA)
            circle2 = plt.Circle((R1[1, 0], R1[1, 1]), radius, color='g', fill=False, alpha=.4, linewidth=0.5)
            self.cir2 = self.ax21.add_patch(circle2)
            art3d.pathpatch_2d_to_3d(circle2, z=0, zdir="z")
        if self.rssi_list3:
            radius = 10 ** ((self.rssi_list3[-1] + BETA) / ALPHA)
            circle3 = plt.Circle((R1[2, 0], R1[2, 1]), radius, color='g', fill=False, alpha=.4, linewidth=0.5)
            self.cir3 = self.ax21.add_patch(circle3)
            art3d.pathpatch_2d_to_3d(circle3, z=0, zdir="z")

        self.ax22.clear()
        self.ax22.set_facecolor('white')
        self.ax22.grid(False)
        self.ax22.set_title("REAL-TIME LORA RSSI", fontweight='bold', fontsize=9, pad=-2)
        self.ax22.set_ylabel("RSSI (dBm)", labelpad=-3)
        # self.ax22.set_ylim(-90, -10)
        if self.anchor:
            self.ax22.plot(self.rssi_list, 'coral', label='RX Commander')
            self.ax22.plot(self.smoothed_rssi_list, 'green', label='RX Cmd Smoothed')
        if self.rssi_list2:
            self.ax22.plot(self.rssi_list2, 'b', alpha=.5, label='RX 2')
        if self.rssi_list3:
            self.ax22.plot(self.rssi_list3, 'cyan', alpha=.5, label='RX 3')
        if self.anchor:
            self.ax22.legend(loc='lower left')

        if self.blit:
            # restore background
            self.fig2.canvas.restore_region(self.ax1background)
            self.fig2.canvas.restore_region(self.ax2background)

            # redraw just the points
            self.ax21.draw_artist(self.handle_scat)
            self.ax21.draw_artist(self.handle_arrw)
            self.ax21.draw_artist(self.handle_scat_pf)

            # fill in the axes rectangle
            self.fig2.canvas.blit(self.ax21.bbox)
            self.fig2.canvas.blit(self.ax22.bbox)

        else:
            self.fig2.canvas.draw()

        self.fig2.canvas.flush_events()

        stop_t = time.time() - start_t
        print("Elapsed time of VISUALISATION = ", stop_t)
        # Constrain PLOT time to avoid msg Overflow
        if stop_t > t_limit:
            self.fig2.savefig("replay_pf.png")
            self.reset_view()
            self.set_view(self.path[-1][0], self.path[-1][1], self.path[-1][2], self.U, self.V, self.W,
                          self.xs[-1][0], self.xs[-1][1], 0.)

    def set_view(self, x=0, y=0, z=0, u=1, v=0, w=0, X=0, Y=0, Z=0):

        self.ax21.view_init(elev=75., azim=-75)
        self.ax21.set_title("REAL-TIME TRAJECTORY", fontweight='bold', fontsize=9, pad=-5.0)
        self.ax21.grid(True)
        self.ax21.set_xlabel('X Axis (m)')
        self.ax21.set_ylabel('Y Axis (m)')
        self.ax21.set_zlabel('Z Axis (m)')
        '''
        self.ax21.set_xlim(-2, 2)
        self.ax21.set_ylim(-2, 2)
        self.ax21.set_zlim(-2, 2)
        '''
        quiv_len = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        self.handle_scat = self.ax21.scatter(x, y, z, color='b', marker='o', alpha=.9, label='MIO')
        self.handle_arrw = self.ax21.quiver(x, y, z, u, v, w, color='b', length=2., arrow_length_ratio=0.3,
                                            linewidths=3., alpha=.7)
        self.handle_scat_pf = self.ax21.scatter(X, Y, Z, color='r', marker='o', alpha=.9, label='LoRa-MIO')
        self.ax21.legend(loc='upper left')

        # Show RXs
        for anc in range(0, self.anchor):
            self.ax21.scatter(R1[int(anc), 0], R1[int(anc), 1], R1[int(anc), 2], marker='1', s=100, color='magenta')

        # Init Range Display
        if not self.rssi_list:
            radius = .5
        else:
            radius = self.rssi_list[-1]
        circle1 = plt.Circle((R1[0, 0], R1[0, 1]), radius, color='g', fill=False, alpha=.1, linewidth=0.5)
        self.cir1 = self.ax21.add_artist(circle1)
        circle2 = plt.Circle((R1[1, 0], R1[1, 1]), radius, color='g', fill=False, alpha=.1, linewidth=0.5)
        self.cir2 = self.ax21.add_artist(circle2)
        circle3 = plt.Circle((R1[2, 0], R1[2, 1]), radius, color='g', fill=False, alpha=.1, linewidth=0.5)
        self.cir3 = self.ax21.add_artist(circle3)

    def reset_view(self):
        self.rssi_list = []
        self.rssi_list2 = []
        self.rssi_list3 = []
        self.ax21.clear()



if __name__ == "__main__":

    main()