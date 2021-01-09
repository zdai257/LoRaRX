import sys
import math
import numpy as np
import time
from math import sqrt
from numpy.random import randn
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
from scipy.signal import savgol_filter
from eulerangles import *
from utility import *
from plot_util import *
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
#matplotlib.use('agg')


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


def HJacobian_at(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    #Z = x[2, 0]
    theta = x[2, 0]
    V = x[3, 0]
    a = ALPHA*math.log10(math.e)
    # HJabobian in (3, 4) if ONE LoRa RX; (5, 4) if THREE LoRa RXs available
    Jacob = array([[0, 0, -dt * V * math.sin(theta), dt * math.cos(theta)],
                                     [0, 0, dt * V * math.cos(theta), dt * math.sin(theta)]])
    
    for row in range(0, anchor):
        denom = (X - R1[row, 0]) ** 2 + (Y - R1[row, 1]) ** 2
        if denom < 1.:
            denom = 1.
        Jacob = np.vstack((Jacob, array([[a*(X - R1[row, 0])/denom, a*(Y - R1[row, 1])/denom, 0, 0]])))
    
    #print("HJacobian return: ", Jacob)
    return Jacob


def hx(x, anchor=1):
    """ compute measurement of [X, Y, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    trans_x = dt * x[3, 0] * math.cos(x[2, 0])
    trans_y = dt * x[3, 0] * math.sin(x[2, 0])
    h = array([trans_x, trans_y]).reshape((-1, 1))
    for row in range(0, anchor):
        dis = np.linalg.norm(x[:2, 0] - R1[row, :2])
        thres_dis = 1.
        if dis > thres_dis:
            # RSSI Regression Model
            rssi = ALPHA*math.log10(dis) + BETA
            
        else:
            rssi = ALPHA*math.log10(thres_dis) + BETA
        
        # Measurement comprises (X, Y, RSSIs)
        h = np.vstack((h, array([[rssi]])))
    #print("hx return shape: ", h.shape)
    return h


def HJacobian_at_ZYaw(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    # Z = x[2, 0]
    theta = x[2, 0]
    V = x[3, 0]
    a = ALPHA * math.log10(math.e)
    # HJabobian in (4, 4) if ONE LoRa RX; (6, 4) if THREE LoRa RXs available
    Jacob = array([[0, 0, -dt * V * math.sin(theta), dt * math.cos(theta)],
                   [0, 0, dt * V * math.cos(theta), dt * math.sin(theta)],
                   [0, 0, 1, 0]])
    for row in range(0, anchor):
        denom = (X - R1[row, 0]) ** 2 + (Y - R1[row, 1]) ** 2
        if denom < 1.:
            denom = 1.
        Jacob = np.vstack((Jacob, array([[a * (X - R1[row, 0]) / denom, a * (Y - R1[row, 1]) / denom, 0, 0]])))
    # print("HJacobian return: ", Jacob)
    return Jacob


def hx_ZYaw(x, anchor=1):
    """ compute measurement of [X, Y, Yaw, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    trans_x = dt * x[3, 0] * math.cos(x[2, 0])
    trans_y = dt * x[3, 0] * math.sin(x[2, 0])
    abs_yaw = x[2, 0]
    h = array([trans_x, trans_y, abs_yaw]).reshape((-1, 1))
    for row in range(0, anchor):
        dis = np.linalg.norm(x[:2, 0] - R1[row, :2])
        thres_dis = 1.
        if dis > thres_dis:
            # RSSI Regression Model
            rssi = ALPHA * math.log10(dis) + BETA
        else:
            rssi = ALPHA * math.log10(thres_dis) + BETA

        # Measurement comprises (X, Y, abs_Yaw, RSSIs...)
        h = np.vstack((h, array([[rssi]])))
    # print("hx return shape: ", h.shape)
    return h



def HJacobian_at_AngularV(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    theta = x[2, 0]
    V = x[3, 0]
    W = x[4, 0]
    # Fix a BUG: denom is variable
    a = ALPHA * math.log10(math.e)
    # HJabobian in (4, 5) if ONE LoRa RX; (6, 5) if THREE LoRa RXs available
    Jacob = array([[0, 0, 0, dt * math.cos(dt*W), -dt**2 * V * math.sin(dt*W)],
                   [0, 0, 0, dt * math.sin(dt*W), dt**2 * V * math.cos(dt*W)],
                   [0, 0, 0, 0, dt]])
    for row in range(0, anchor):
        denom = (X - R1[row, 0]) ** 2 + (Y - R1[row, 1]) ** 2
        if denom < 1.:
            denom = 1.
        Jacob = np.vstack((Jacob, array([[a * (X - R1[row, 0]) / denom, a * (Y - R1[row, 1]) / denom, 0, 0, 0]])))
        #Jacob = np.vstack((Jacob, array([[0, 0, 0, 0, 0]])))
    # print("HJacobian return: ", Jacob)
    return Jacob


def hx_AngularV(x, anchor=1):
    """ compute measurement of [X, Y, ROT_Z, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    trans_x = dt * x[3, 0] * math.cos(dt * x[4, 0])
    trans_y = dt * x[3, 0] * math.sin(dt * x[4, 0])
    rot_z = dt * x[4, 0]
    h = array([trans_x, trans_y, rot_z]).reshape((-1, 1))
    for row in range(0, anchor):
        dis = np.linalg.norm(x[:2, 0] - R1[row, :2])
        thres_dis = 1.
        if dis > thres_dis:
            # RSSI Regression Model
            rssi = ALPHA * math.log10(dis) + BETA
        else:
            rssi = ALPHA * math.log10(thres_dis) + BETA

        # Measurement comprises (X, Y, Rot_Z, RSSIs...)
        h = np.vstack((h, array([[rssi]])))
        #h = np.vstack((h, array([[0]])))
    # print("hx return shape: ", h.shape)
    return h


def HJacobian_at_ConstantA(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    theta = x[2, 0]
    V = x[3, 0]  # Attempt to suppress Velocity
    W = x[4, 0]
    A = x[5, 0]
    # Fix a BUG: denom is variable
    a = ALPHA * math.log10(math.e)
    # HJabobian in (3, 6) if ZERO LoRa RX; (6, 6) if THREE LoRa RXs available
    Jacob = array([[0, 0, 0, dt * math.cos(dt*W), -dt**2 * V * math.sin(dt*W), 0],
                   [0, 0, 0, dt * math.sin(dt*W), dt**2 * V * math.cos(dt*W), 0],
                   [0, 0, 0, 0, dt, 0]])
    for row in range(0, anchor):
        denom = (X - R1[row, 0]) ** 2 + (Y - R1[row, 1]) ** 2
        if denom < 0.01:
            denom = 0.01
        Jacob = np.vstack((Jacob, array([[a * (X - R1[row, 0]) / denom, a * (Y - R1[row, 1]) / denom, 0, 0, 0, 0]])))

    # print("HJacobian return: ", Jacob)
    return Jacob


def hx_ConstantA(x, anchor=1):
    """ compute measurement of [X, Y, ROT_Z, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    V = x[3, 0]
    W = x[4, 0]
    trans_x = dt * V * math.cos(dt * W)
    trans_y = dt * V * math.sin(dt * W)
    rot_z = dt * W
    h = array([trans_x, trans_y, rot_z]).reshape((-1, 1))
    for row in range(0, anchor):
        dis = np.linalg.norm(x[:2, 0] - R1[row, :2])
        thres_dis = 0.1
        if dis > thres_dis:
            rssi = ALPHA * math.log10(dis) + BETA
        else:
            rssi = ALPHA * math.log10(thres_dis) + BETA
        # Measurement comprises (X, Y, Rot_Z, RSSIs...)
        h = np.vstack((h, array([[rssi]])))
    # print("hx return shape: ", h.shape)
    return h


def HJacobian_at_PosVel(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    Vx = x[2, 0]
    Vy = x[3, 0]
    W = x[4, 0]
    a = ALPHA * math.log10(math.e)
    # HJabobian in (3, 5) if ZERO LoRa RX; (6, 5) if THREE LoRa RXs available
    Jacob = array([[0, 0, dt*Vx*math.cos(dt*W) / math.sqrt(Vx**2+Vy**2), dt*Vy*math.cos(dt*W) / math.sqrt(Vx**2+Vy**2), -dt**2 * math.sin(dt*W)*math.sqrt(Vx**2+Vy**2)],
                   [0, 0, dt*Vx*math.sin(dt*W) / math.sqrt(Vx**2+Vy**2), dt*Vy*math.sin(dt*W) / math.sqrt(Vx**2+Vy**2), dt**2 * math.cos(dt*W)*math.sqrt(Vx**2+Vy**2)],
                   [0, 0, 0, 0, dt]])
    for row in range(0, anchor):
        denom = (X - R1[row, 0]) ** 2 + (Y - R1[row, 1]) ** 2
        if denom < 1.:
            denom = 1.
        Jacob = np.vstack((Jacob, array([[a * (X - R1[row, 0]) / denom, a * (Y - R1[row, 1]) / denom, 0, 0, 0]])))

    # print("HJacobian return: ", Jacob)
    return Jacob


def hx_PosVel(x, anchor=1):
    """ compute measurement of [X, Y, ROT_Z, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    trans_x = dt * math.sqrt(x[2, 0]**2+x[3, 0]**2) * math.cos(dt * x[4, 0])
    trans_y = dt * math.sqrt(x[2, 0]**2+x[3, 0]**2) * math.sin(dt * x[4, 0])
    rot_z = dt * x[4, 0]
    h = array([trans_x, trans_y, rot_z]).reshape((-1, 1))
    for row in range(0, anchor):
        dis = np.linalg.norm(x[:2, 0] - R1[row, :2])
        thres_dis = 1.
        if dis > thres_dis:
            rssi = ALPHA * math.log10(dis) + BETA
        else:
            rssi = ALPHA * math.log10(thres_dis) + BETA
        # Measurement comprises (X, Y, Rot_Z, RSSIs...)
        h = np.vstack((h, array([[rssi]])))
    # print("hx return shape: ", h.shape)
    return h


def HJacobian_Origin(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    theta = x[2, 0]
    a = ALPHA * math.log10(math.e)
    # HJabobian in (3, 3) if ZERO LoRa RX; (6, 3) if THREE LoRa RXs available
    Jacob = eye(3)
    for row in range(0, anchor):
        denom = (X - R1[row, 0]) ** 2 + (Y - R1[row, 1]) ** 2
        if denom < 0.01:
            denom = 0.01
        Jacob = np.vstack((Jacob, array([[a * (X - R1[row, 0]) / denom, a * (Y - R1[row, 1]) / denom, 0]])))

    # print("HJacobian return: ", Jacob)
    return Jacob


def hx_Origin(x, anchor=1):
    """ compute measurement of [X, Y, ROT_Z, RSSIs...]^T that would correspond to state x.
    """
    dt = .1
    h = array([x[0, 0], x[1, 0], x[2, 0]]).reshape((-1, 1))
    for row in range(0, anchor):
        dis = np.linalg.norm(x[:2, 0] - R1[row, :2])
        thres_dis = 0.1
        if dis > thres_dis:
            rssi = ALPHA * math.log10(dis) + BETA
        else:
            rssi = ALPHA * math.log10(thres_dis) + BETA
        # Measurement comprises (X, Y, Rot_Z, RSSIs...)
        h = np.vstack((h, array([[rssi]])))
    # print("hx return shape: ", h.shape)
    return h


class Simu(object):
    """ Simulates the Path in 2D.
    """
    def __init__(self, dt, pos, rot, vel):
        self.pos = pos
        self.rot = rot
        self.vel = vel
        self.dt = dt
        self.t_idx = 0

    def get_measure(self, anchor=1):
        """ Returns Observation/Measurement. Call once for each
        new measurement at dt time from last call.
        """
        
        # add some process noise to the system
        self.vel = self.vel + (.1 * randn() + 0.)
        self.rot = self.rot + (.1 * randn(3, 1) + 0.)
        self.pos = self.pos + array([self.vel * self.dt * math.cos(self.rot[2, 0]), self.vel * self.dt * math.sin(self.rot[2, 0]), 0])

        # Constrain Path to 2D
        #self.vel[2, 0] = 0.
        self.rot[0, 0], self.rot[1, 0] = 0., 0.
        self.pos[2, 0] = 0.

        # Add measurement noise of 5%
        z_pos = self.pos * (1 + 0.05 * randn(3, 1))
        z_rot = self.rot * (1 + 0.05 * randn(3, 1))

        # Simulate a decreasing/increasing RSSI
        rssi = -80 + .8*self.t_idx - 10*np.random.rand()
        self.t_idx += 1
        #print("Measured RSSI: ", rssi)

        # Generate Observation
        z0 = array([[z_pos[0, 0]], [z_pos[1, 0]], [rssi]])
        
        for dim in range(1, anchor):
            rssi = -.8*self.t_idx - 10*np.random.rand()
            z0 = np.vstack((z0, array([[rssi]])))
        #print("Generate Measurement shape: ", z0.shape)
        return z0



class EKF_Fusion():
    def __init__(self, dt=0.1, anchor=1, dim_x=4, dim_z=3, blit=True, visual=False, dense=False):
        self.visual = visual
        self.dense = dense
        self.anchor = anchor
        # Current Pose handler
        self.pred_transform_t_1 = np.array(
        [[1., 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]])
        '''
        z0, y0, x0 = mat2euler(self.pred_transform_t_1[:3, :3])
        print(euler2quat(z0, y0, x0))
        '''
        self.out_pred_array = []
        self.final_list = []
        self.sigma_list = []
        self.angle = 0
        self.rssi_list, self.smoothed_rssi_list, self.rssi_list2, self.rssi_list3 = [], [], [], []
        self.vec = np.array([[0], [1], [0], [0]], dtype=float)
        self.odom_quat = self.vec
        
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
        #self.ax21 = self.fig2.add_subplot(2, 1, 1, projection='3d') #fig1.gca(projection='3d') #Axes3D(fig1)
        #self.ax22 = self.fig2.add_subplot(2, 1, 2)
        plt.ion()
        plt.tight_layout()
        self.set_view()

        self.fig2.canvas.draw()
        if self.blit:
            self.ax1background = self.fig2.canvas.copy_from_bbox(self.ax21.bbox)
            self.ax2background = self.fig2.canvas.copy_from_bbox(self.ax22.bbox)
        if self.visual:
            plt.show(block=False)
        
        # Creating EKF
        self.dt = dt
        self.my_kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        # Create synthetic Pose and Velocity
        self.pv = Simu(dt=self.dt, pos=np.zeros(shape=(3, 1)), rot=np.zeros(shape=(3, 1)), vel=30.)

        # make an imperfect starting guess
        self.my_kf.x = array([0., 0., 0., 0.01]).reshape(-1, 1)

        # State Transition Martrix: F
        self.my_kf.F = eye(4) + array([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]) * self.dt

        # Measurement Noise: R Can be defined Dynamic with MDN-Sigma!
        # my_kf.R = 4.887 # if using 1-dimension Measurement
        self.my_kf.R = np.diag(np.array([1.**2, 1.**2, SIGMA**2]))

        # Process Noise: Q
        self.my_kf.Q = array([[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, .01, 0],
                                         [0, 0, 0, .1]])
        # Initial Error Covariance: P0
        self.my_kf.P *= 50

        # logging
        self.track = []
        self.time = []
        self.xs = []
        self.path, self.path_dense = [], []
        self.abs_x, self.abs_y, self.abs_yaw = [], [], []
        self.U, self.V, self.W = [], [], []

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
        if self.anchor:
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
            # Compute ABS_POSE
            self.abs_x.append(abs_pred_transform[0, 3])
            self.abs_y.append(abs_pred_transform[1, 3])
            self.path_dense.append([self.abs_x[-1], self.abs_y[-1], 0])

            euler_rot = np.array([[abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2]],
                                  [abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2]],
                                  [abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2]]],
                                 dtype=float)
            # euler_rad = (yaw, pitch, roll)
            euler_rad = mat2euler(euler_rot)
            self.abs_yaw.append(euler_rad[0])
            # print("Current Eular = ", euler_rad)
            self.odom_quat = np.array(euler2quat(euler_rad[0], euler_rad[1], euler_rad[2]))
            # print("Current Quaternion = ", self.odom_quat)
            if self.dense:
                # Unit Vector from Eular Angle; Simplify Orientation Representation by Pitch = 0
                self.U.append(math.cos(self.abs_yaw[-1]))  # math.cos(euler_rad[0])*math.cos(euler_rad[1])
                self.V.append(math.sin(self.abs_yaw[-1]))  # math.sin(euler_rad[0])*math.cos(euler_rad[1])
                self.W.append(0.)  # math.sin(euler_rad[1])

        gap = int(len(msg_list) / len_pose)
        # print(self.out_pred_array[-1])

        if not self.dense:
            self.U.append(math.cos(self.abs_yaw[-1]))  # math.cos(euler_rad[0])*math.cos(euler_rad[1])
            self.V.append(math.sin(self.abs_yaw[-1]))  # math.sin(euler_rad[0])*math.cos(euler_rad[1])
            self.W.append(0.)  # math.sin(euler_rad[1])

        self.path.append([self.abs_x[-1], self.abs_y[-1], 0])

        print("Elapsed time PoseTransform = ", time.time() - start_t)
        start_t = time.time()

        # Trigger EKF
        self.rt_run(gap)
        print("Elapsed time of EKF = ", time.time() - start_t)
        print("ABS_YAW: %.3f (=%.3f OR %.3f)" % (
        self.abs_yaw[-1], self.abs_yaw[-1] - 2 * math.pi, self.abs_yaw[-1] - 4 * math.pi))
        print("State X:\n", self.my_kf.x)

        if self.visual and not self.dense:
            self.rt_show()

    def rt_run(self, gap):

        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            final_xy = final_pose[:2]
            # print(final_xy)

            # Populate ONE Rssi for a 'gap' of Poses
            if self.anchor:
                final_xy.append(float(self.smoothed_rssi_list[-1]))  # Utilize Smoothed RSSI for Fusion
            if self.rssi_list2:
                final_xy.append(float(self.rssi_list2[-1]))
            if self.rssi_list3:
                final_xy.append(float(self.rssi_list3[-1]))

            z = np.asarray(final_xy, dtype=float).reshape(-1, 1)
            # print("Measurement:\n", z)
            # Refresh Measurement noise R
            '''
            for j in range(0, 2):
                self.my_kf.R[j, j] = self.sigma_list[-g][j]**2 # Sigma stands for Standard Deviation
            '''
            # Refresh State Transition Martrix: F
            self.my_kf.F = eye(4) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]),
                                            self.dt * math.cos(self.my_kf.x[2, 0])],
                                           [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]),
                                            self.dt * math.sin(self.my_kf.x[2, 0])],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]])

            # PREDICTION
            self.my_kf.predict()
            # print("X-:\n", self.my_kf.x)

            # UPDATE
            self.my_kf.update(z, HJacobian_at, hx, args=(self.anchor), hx_args=(self.anchor))

            # Log Posterior State x
            self.xs.append(self.my_kf.x)

            # print("X+:\n", self.my_kf.x)
            # print("EKF per round takes %.6f s" % (time.time() - start_t))
            if self.visual and self.dense:
                self.rt_show(odom_idx=-g)


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


    def sim_run(self, anchor=1):
        start_t = time.time()
        
        z = self.pv.get_measure(anchor=anchor)
        
        # Log track or GroundTruth
        self.track.append((self.pv.pos, self.pv.rot, self.pv.vel))
        
        # Refresh (x, y) of measurement noise R at runtime
        for j in range(0, 2):
            self.my_kf.R[j, j] = 2.*np.random.rand()
        
        # Refresh State Transition Martrix: F
        self.my_kf.F = eye(4) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]), self.dt * math.cos(self.my_kf.x[2, 0])],
                                  [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]), self.dt * math.sin(self.my_kf.x[2, 0])],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]) * self.dt
        
        # PREDICTION
        self.my_kf.predict()
        
        # UPDATE
        self.my_kf.update(z, HJacobian_at, hx, args=(anchor), hx_args=(anchor))

        # Log Posterior State x
        self.xs.append(self.my_kf.x)
        
        print("EKF process time = %.6f s" % (time.time() - start_t))
        

    def sim_show(self):
        self.xs = asarray(self.xs)
        self.track = asarray(self.track)
        self.time = np.arange(0, len(self.xs) * self.dt, self.dt)
        # print(track)
        # print(xs)
        
        x0 = [item[0][0] for item in self.xs]
        y0 = [item[1][0] for item in self.xs]

        x_gt = [item[0][0, 0] for item in self.track]
        y_gt = [item[0][1, 0] for item in self.track]
        
        fig1 = plt.figure()
        self.ax1 = fig1.add_subplot(1,1,1)
        h1 = self.ax1.plot(x0, y0, 'b', label='Predicted')
        h2 = self.ax1.plot(x_gt, y_gt, 'r', label='GroundTruth')
        self.ax1.set_aspect('equal')
        self.ax1.title.set_text("EKF Path v.s. GT")
        self.ax1.set_xlabel("X (m)")
        self.ax1.set_ylabel("Y (m)")
        self.ax1.legend(loc='best')
        fig1.savefig("sim_example.png")
        if self.visual:
            plt.show()
        
        
    def rt_show(self, odom_idx=-1, t_limit=0.85, mark_size=20):
        start_t = time.time()
        if self.dense:
            traj = self.path_dense
        else:
            traj = self.path
        traj_fuse = np.asarray(self.xs)
        u, v, w = self.U[-1], self.V[-1], self.W[-1]

        self.handle_scat.set_alpha(.2)
        self.handle_scat_ekf.set_alpha(.2)
        self.handle_arrw.remove()
        # Remove Range Circle
        if self.anchor:
            self.cir1.remove()
        if self.rssi_list2:
            self.cir2.remove()
        if self.rssi_list3:
            self.cir3.remove()

        self.handle_scat = self.ax21.scatter([traj[odom_idx][0]], [traj[odom_idx][1]], [traj[odom_idx][2]], s=mark_size, color='b', marker='o', alpha=.9, label='MIO')
        self.handle_arrw = self.ax21.quiver([traj[odom_idx][0]], [traj[odom_idx][1]], [traj[odom_idx][2]], u, v, w, color='cyan', length=2., arrow_length_ratio=0.3, linewidths=3., alpha=.7)
        self.handle_scat_ekf = self.ax21.scatter([traj_fuse[-1][0, 0]], [traj_fuse[-1][1, 0]], [0.], s=mark_size, color='r', marker='o', alpha=.9, label='LoRa-MIO')
        # Not Attempting to Visual EKF Updated Orientation
        #self.handle_arrw_ekf = self.ax21.quiver([self.my_kf.x[0, 0]], [self.my_kf.x[1, 0]], [self.my_kf.x[2, 0]], self.U_ekf, self.V_ekf, self.W_ekf, color='r', length=1., alpha=.7)
        # Manually Equal Axis and Limit
        self.ax21.auto_scale_xyz([-5, 15], [-15, 5], [-1, 3])

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
        #self.ax22.set_ylim(-90, -10)
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
            self.ax21.draw_artist(self.handle_scat_ekf)

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
            self.fig2.savefig("live_rx.png")
            self.reset_view()
            self.set_view(traj[odom_idx][0], traj[odom_idx][1], traj[odom_idx][2], u, v, w, traj_fuse[-1][0, 0], traj_fuse[-1][1, 0], 0.)
        
    def set_view(self, mark_size=20, x=0, y=0, z=0, u=1, v=0, w=0, X=0, Y=0, Z=0):
        
        self.ax21.view_init(elev=75., azim=-75)
        self.ax21.set_title("REAL-TIME TRAJECTORY", fontweight='bold', fontsize=9, pad=-5.0)
        self.ax21.grid(True)
        self.ax21.set_facecolor('white')
        self.ax21.set_xlabel('X Axis (m)')
        self.ax21.set_ylabel('Y Axis (m)')
        self.ax21.set_zlabel('Z Axis (m)')
        '''
        self.ax21.set_xlim(-2, 2)
        self.ax21.set_ylim(-2, 2)
        self.ax21.set_zlim(-2, 2)
        '''
        quiv_len = np.sqrt(u**2 + v**2 + w**2)
        self.handle_scat = self.ax21.scatter(x, y, z, s=mark_size, color='b', marker='o', alpha=.9, label='MIO')
        self.handle_arrw = self.ax21.quiver(x, y, z, u, v, w, color='b', length=2., arrow_length_ratio=0.3, linewidths=3., alpha=.7)
        self.handle_scat_ekf = self.ax21.scatter(X, Y, Z, s=mark_size, color='r', marker='o', alpha=.9, label='LoRa-MIO')
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


if __name__=="__main__":
    
    dt = 0.1
    ekf = EKF_Fusion(dt=dt)
    
    T = 10.
    for i in range(int(T / dt)):
        ekf.sim_run()
        time.sleep(.1)
        
    ekf.sim_show()
    


