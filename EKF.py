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
import matplotlib
#matplotlib.use('agg')
import sympy
from sympy import Matrix, symbols


# LoRa RX Coordinates
R1 = np.array([[0., 0., 0.],
               [5., 0., 0.],
               [0., 4., 0.],
               [5., 4., 0.],
               [2., -2.5, 0.]])


def HJacobian_at(x, anchor=1):
    """ compute Jacobian of H matrix for state x """
    dt = .1
    X = x[0, 0]
    Y = x[1, 0]
    #Z = x[2, 0]
    theta = x[2, 0]
    V = x[3, 0]
    denom = (X - R1[anchor-1, 0])**2 + (Y - R1[anchor-1, 1])**2
    if denom < 0.25:
        denom = 0.25
    a = -28.57*math.log10(math.e)
    # HJabobian in (3, 4) if ONE LoRa RX; (5, 4) if THREE LoRa RXs available
    Jacob = array([[0, 0, -dt * V * math.sin(theta), dt * math.cos(theta)],
                                     [0, 0, dt * V * math.cos(theta), dt * math.sin(theta)]])
    
    for row in range(0, anchor):
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
        
        if dis > 0.5:
            # RSSI Regression Model
            rssi = -28.57*math.log10(dis) - 5.06
            
        else:
            rssi = -28.57*math.log10(0.5) - 5.06
        
        # Measurement comprises (X, Y, RSSIs)
        h = np.vstack((h, array([[rssi]])))
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
    def __init__(self, dt=0.1, dim_x=4, dim_z=3, blit=True, visual=False):
        self.visual = visual
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
        self.rssi_list = []
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
        
        self.fig2 = plt.figure(figsize=(8, 9))
        self.ax21 = self.fig2.add_subplot(2, 1, 1, projection='3d') #fig1.gca(projection='3d') #Axes3D(fig1)
        self.ax22 = self.fig2.add_subplot(2, 1, 2)
        plt.ion()
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
        self.pv = PoseVel(dt=self.dt, pos=np.zeros(shape=(3, 1)), rot=np.zeros(shape=(3, 1)), vel=30.)

        # make an imperfect starting guess
        self.my_kf.x = array([0., 0., 0., 0.01]).reshape(-1, 1)

        # State Transition Martrix: F
        self.my_kf.F = eye(4) + array([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]) * self.dt

        # Measurement Noise: R Can be defined Dynamic with MDN-Sigma!
        # my_kf.R = 4.887 # if using 1-dimension Measurement
        self.my_kf.R = np.diag(np.array([1.**2, 1.**2, 4.887**2]))

        # Process Noise: Q
        self.my_kf.Q = array([[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, .01, 0],
                                         [0, 0, 0, .1]])
        # Initial Error Covariance: P0
        self.my_kf.P *= 50

        # logging
        self.xs = []
        self.track = []
        self.time = []
        self.path = []
        self.path_ekf = []
        self.abs_yaw = 0
        
        
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
        #print(self.out_pred_array[-1])
        
        euler_rot = np.array([[abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2]],
                              [abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2]],
                              [abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2]]], dtype=float)
        # euler_rad = (yaw, pitch, roll)
        euler_rad = mat2euler(euler_rot)
        self.abs_yaw = euler_rad[0]
        #print("Current Eular = ", euler_rad)
        self.odom_quat = np.array(euler2quat(euler_rad[0], euler_rad[1], euler_rad[2]))
        #print("Current Quaternion = ", self.odom_quat)
        
        # Unit Vector from Eular Angle; Simplify Orientation Representation by Pitch = 0
        self.U = math.cos(self.abs_yaw)#math.cos(euler_rad[0])*math.cos(euler_rad[1])
        self.V = math.sin(self.abs_yaw)#math.sin(euler_rad[0])*math.cos(euler_rad[1])
        self.W = 0#math.sin(euler_rad[1])
        
        self.path.append([abs_pred_transform[0, 3], abs_pred_transform[1, 3], 0])
        
        print("Elapsed time Pose2TrfMtx = ", time.time() - start_t)
        start_t = time.time()
        
        # Trigger EKF
        self.rt_run(gap)
        print("Elapsed time of EKF = ", time.time() - start_t)
        print("ABS_YAW: ", self.abs_yaw)
        print("State X:\n", self.my_kf.x)
        if self.visual:
            self.rt_show()
        
        
    def rt_run(self, gap):
        
        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            final_xy = final_pose[:2]
            #print(final_xy)
            
            # Populate ONE Rssi for a 'gap' of Poses
            final_xy.append(float(self.rssi_list[-1]))
            
            z = np.asarray(final_xy, dtype=float).reshape(-1, 1)
            #print("Measurement:\n", z)
            # Refresh Measurement noise R
            for j in range(0, 2):
                self.my_kf.R[j, j] = sigma_list[-g][j]**2 # Sigma stands for Standard Deviation
                
            # Refresh State Transition Martrix: F
            self.my_kf.F = eye(4) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]), self.dt * math.cos(self.my_kf.x[2, 0])],
                                  [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]), self.dt * math.sin(self.my_kf.x[2, 0])],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]) * self.dt
            
            # PREDICTION
            self.my_kf.predict()
            #print("X-:\n", self.my_kf.x)
            
            # UPDATE
            self.my_kf.update(z, HJacobian_at, hx, args=(anchor), hx_args=(anchor))
            
            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            
            #print("X+:\n", self.my_kf.x)
            
            #print("EKF per round takes %.6f s" % (time.time() - start_t))
        
        
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
        
        
    def rt_show(self, t_limit=0.85):
        start_t = time.time()
        #u, v, w = self.odom_quat[0], self.odom_quat[1], self.odom_quat[2]
        
        self.handle_scat.set_alpha(.2)
        self.handle_scat_ekf.set_alpha(.2)
        self.handle_arrw.remove()
        
        #self.handle_arrw_ekf.remove()
        self.handle_scat = self.ax21.scatter([self.path[-1][0]], [self.path[-1][1]], [self.path[-1][2]], color='b', marker='o', alpha=.9, label='MIO')
        self.handle_arrw = self.ax21.quiver([self.path[-1][0]], [self.path[-1][1]], [self.path[-1][2]],
            self.U, self.V, self.W, color='b', length=2., arrow_length_ratio=0.05, linewidths=3., alpha=.7)
        self.handle_scat_ekf = self.ax21.scatter([self.xs[-1][0, 0]], [self.xs[-1][1, 0]], [0.], color='r', marker='o', alpha=.9, label='LoRa-MIO')
        # Not Attempting to Visual EKF Updated Orientation
        #self.handle_arrw_ekf = self.ax21.quiver([self.my_kf.x[0, 0]], [self.my_kf.x[1, 0]], [self.my_kf.x[2, 0]], self.U_ekf, self.V_ekf, self.W_ekf, color='r', length=1., alpha=.7)
        
        
        self.ax22.clear()
        self.ax22.set_title("Real-Time LoRa Signal Strength", fontweight='bold')
        self.ax22.set_ylabel("RSSI (dBm)")
        self.ax22.plot(self.rssi_list, 'coral')
        
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
            self.set_view(self.path[-1][0], self.path[-1][1], self.path[-1][2], self.U, self.V, self.W, self.xs[-1][0, 0], self.xs[-1][1, 0], 0.)
        
    def set_view(self, x=0, y=0, z=0, u=1, v=0, w=0, X=0, Y=0, Z=0):
        self.ax21.view_init(elev=60., azim=-75)
        self.ax21.set_title("Real-Time Pose", fontweight='bold')
        self.ax21.set_xlabel('X Axis (m)')
        self.ax21.set_ylabel('Y Axis (m)')
        self.ax21.set_zlabel('Z Axis (m)')
        
        '''
        self.ax21.set_xlim(-2, 2)
        self.ax21.set_ylim(-2, 2)
        self.ax21.set_zlim(-2, 2)
        '''
        quiv_len = np.sqrt(u**2 + v**2 + w**2)
        self.handle_scat = self.ax21.scatter(x, y, z, color='b', marker='o', alpha=.9, label='MIO')
        self.handle_arrw = self.ax21.quiver(x, y, z, u, v, w, color='b', length=2., arrow_length_ratio=0.05, linewidths=3., alpha=.7)
        self.handle_scat_ekf = self.ax21.scatter(X, Y, Z, color='r', marker='o', alpha=.9, label='LoRa-MIO')
        self.ax21.legend(loc='upper left')
        
        
    def reset_view(self):
        self.rssi_list = []
        self.ax21.clear()


if __name__=="__main__":
    
    dt = 0.1
    ekf = EKF_Fusion(dt=dt)
    
    T = 10.
    for i in range(int(T / dt)):
        ekf.sim_run()
        time.sleep(.1)
        
    ekf.sim_show()
    


