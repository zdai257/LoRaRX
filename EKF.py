import sys
import math
import numpy as np
import time
from math import sqrt
from numpy.random import randn
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
from eulerangles import *
from utility import *
from plot_util import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# LoRa RX1 Coordinates
R1 = np.array([20., 20., 0])


def HJacobian_at(x):
    """ compute Jacobian of H matrix for state x """

    X = x[0, 0]
    Y = x[1, 0]
    Z = x[2, 0]
    denom = (X - R1[0])**2 + (Y - R1[1])**2 + (Z - R1[2])**2
    a = 28.57*math.log10(math.e)
    # HJabobian in (7, 9) if ONE LoRa RX; (9, 9) if THREE LoRa RXs available
    Jacob = array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [a*(X - R1[0])/denom, a*(Y - R1[1])/denom, a*(Z - R1[2])/denom, 0, 0, 0, 0, 0, 0]])
    #return array ([[(X - R1[0])/denom, (Y - R1[1])/denom, (Z - R1[2])/denom, 0, 0, 0, 0, 0, 0]]) * 28.57*math.log10(math.e)
    
    #print("HJacobian return: ", Jacob.shape)
    return Jacob


def hx(x):
    """ compute measurement for (Pose + RSSI) that would correspond to state x.
    """
    # PL Regression Model
    rssi = 22 - (28.57*math.log10(np.linalg.norm(x[:3] - R1)) + 27.06)

    # Measurement comprises 6DoF Pose + RSSIs
    h = array([x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0], x[5, 0], rssi]).reshape((-1, 1))
    #print("hx return shape: ", h.shape)
    return h



class PoseVel(object):
    """ Simulates the Path in 2D.
    """

    def __init__(self, dt, pos, rot, vel):
        self.pos = pos
        self.rot = rot
        self.vel = vel
        self.dt = dt
        self.t_idx = 0

    def get_measure(self):
        """ Returns Observation/Measurement. Call once for each
        new measurement at dt time from last call.
        """
        # add some process noise to the system
        self.vel = self.vel + 1. * (randn(3, 1) - .5)
        self.rot = self.rot + .01 * randn(3, 1)
        self.pos = self.pos + self.vel * self.dt

        # Constrain Path to 2D
        self.vel[2, 0] = 0.
        self.rot[0, 0], self.rot[1, 0] = 0., 0.
        self.pos[2, 0] = 0.

        # Add measurement noise of 5%
        self.pos += self.pos * 0.05 * randn(3, 1)

        # Simulate a decreasing RSSI
        #rssi = 22 - (28.57*math.log10(np.linalg.norm(self.pos - R1)) + 27.06)
        rssi = -.9*self.t_idx - 10*np.random.rand()
        self.t_idx += 1
        #print("Measured RSSI: ", rssi)

        # Generate Observation
        z0 = array([[self.pos[0, 0]], [self.pos[1, 0]], [self.pos[2, 0]], [self.rot[0, 0]], [self.rot[1, 0]], [self.rot[2, 0]], [rssi]])
        #print("Generate Measurement shape: ", z0.shape)
        return z0



class EKF_Fusion():
    def __init__(self, dt=0.1, dim_x=9, dim_z=7, blit=False):
        # Current Pose handler
        self.pred_transform_t_1 = np.array(
        [[1., 0, 0, 0],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]])
        self.out_pred_array = []
        self.final_list = []
        self.sigma_list = []
        self.angle = 0
        self.rssi_list = []
        self.vec = np.array([[0], [1], [0], [0]], dtype=float)
        self.odom_quat = self.vec
        
        self.blit = blit
        self.ax1 = None
        self.ax2 = None
        self.handle_scat = None
        self.handle_arrw = None
        self.ax1background = None
        self.ax2background = None
        
        # Creating EKF
        self.dt = dt
        self.my_kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.pv = PoseVel(dt=self.dt, pos=np.zeros(shape=(3, 1)), rot=np.zeros(shape=(3, 1)), vel=np.zeros(shape=(3, 1)))

        # make an imperfect starting guess
        self.my_kf.x = array([0.1, -0.2, 0., 0.01, 0.01, 0.05, -0.15, 0.1, 0.]).reshape(-1, 1)

        # State Transition Martrix: F
        self.my_kf.F = eye(9) + array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]) * self.dt

        # Measurement Noise: R Can be defined Dynamic with MDN-Sigma!
        # my_kf.R = 4.887 # if using 1-dimension Measurement
        self.my_kf.R = np.array([[.2, 0, 0, 0, 0, 0, 0],
                            [0, .2, 0, 0, 0, 0, 0],
                            [0, 0, .2, 0, 0, 0, 0],
                            [0, 0, 0, .2, 0, 0, 0],
                            [0, 0, 0, 0, .2, 0, 0],
                            [0, 0, 0, 0, 0, .2, 0],
                            [0, 0, 0, 0, 0, 0, 4.887 ** 2]])

        # Process Noise: Q
        self.my_kf.Q = 0.01 * eye(9) + array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 1]]) * 0.04
        # Initial Error Covariance: P0
        self.my_kf.P *= 50

        # logging
        self.xs = []
        self.track = []
        self.time = []


    def rt_run(self, gap):
        start_t = time.time()
        
        for g in range(gap, 0, -1):
            
            # Get Measurement
            final_pose = self.final_list[-g]
            
            # Populate ONE Rssi for a 'gap' of Poses
            final_pose.append(float(self.rssi_list[-1]))
            
            z = np.asarray(final_pose, dtype=float).reshape(-1, 1)
            print("Measurement:\n", z)
            # Refresh Measurement noise R
            for j in range(0, 6):
                self.my_kf.R[j, j] = self.sigma_list[-g][j]
                
            # UPDATE
            self.my_kf.update(z, HJacobian_at, hx)
            print("X-:\n", self.my_kf.x)
            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            
            # PREDICTION
            self.my_kf.predict()
            print("X+:\n", self.my_kf.x)
        
        print("EKF process time = %.6f s" % (time.time() - start_t))
        
        
    def new_measure(self, *args):
        start_t = time.time()
        len_pose=12
        gap = 0
        msg_list = []
        # First (args) is Object!?
        for arg in args:
            msg_list.append(arg)
            #print(type(arg))
        rssi = msg_list[-1]
        msg_list = msg_list[:-1]
        self.rssi_list.append(rssi)
        #print(len(msg_list))
        for idx in range(0, len(msg_list), len_pose):
            final_pose = msg_list[idx:idx+6]
            sigma_pose = msg_list[idx+6:idx+12]
            #print(final_pose)
            #print(sigma_pose)
            self.final_list.append(final_pose)
            self.sigma_list.append(sigma_pose)
            
            pred_transform_t = convert_eul_to_matrix(0, 0, 0, final_pose)
            abs_pred_transform = np.dot(self.pred_transform_t_1, pred_transform_t)
            self.out_pred_array.append([abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
                                            abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
                                            abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2], abs_pred_transform[2, 3]])
            self.pred_transform_t_1 = abs_pred_transform
            
            #pos.odom_quat = tf.transformations.quaternion_from_matrix(pos.pred_transform_t_1)
            #print(pos.odom_quat)
        gap = int(len(msg_list)/len_pose)
        print(gap)
        #print(self.out_pred_array[-1])
        
        euler_rot = np.array([[abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2]],
                              [abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2]],
                              [abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2]]], dtype=float)
        euler_rad = mat2euler(euler_rot)
        self.odom_quat = np.array(euler2quat(euler_rad[0], euler_rad[1], euler_rad[2]))
        print("Current Quaternion = ", self.odom_quat)
        
        # Unit Vector from Eular Angle
        U = math.cos(euler_rad[0])*math.cos(euler_rad[1])
        V = math.sin(euler_rad[0])*math.cos(euler_rad[1])
        W = math.sin(euler_rad[1])
        
        print("Elapsed time Pose2TrfMtx = ", time.time() - start_t)
        start_t = time.time()
        
        # Trigger EKF
        self.rt_run(gap)
        print("Elapsed time of EKF = ", time.time() - start_t)
        
        
        
    def sim_run(self):
        start_t = time.time()
        
        z = self.pv.get_measure()
        # Log track or GroundTruth
        self.track.append((self.pv.pos, self.pv.rot, self.pv.vel))
        
        # Refresh measurement noise R at runtime
        for j in range(0, 6):
            self.my_kf.R[j, j] = 2*np.random.rand()
            
        # UPDATE
        self.my_kf.update(z, HJacobian_at, hx)

        # Log Posterior State x
        self.xs.append(self.my_kf.x)

        # PREDICTION
        self.my_kf.predict()
        
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
        


if __name__=="__main__":
    
    dt = 0.1
    ekf = EKF_Fusion(dt=dt, blit=False)
    
    T = 10.
    for i in range(int(T / dt)):
        ekf.sim_run()
        time.sleep(.1)
        
    ekf.sim_show()
    plt.show()


