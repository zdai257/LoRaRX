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


# LoRa RX1 Coordinates
R1 = np.array([200., 200., 0])


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
        self.vel = self.vel + .3 * (randn(3, 1) + 1.)
        self.rot = self.rot + 10. * (randn(3, 1) - 25.5)
        self.pos = self.pos + self.vel * self.dt

        # Constrain Path to 2D
        self.vel[2, 0] = 0.
        self.rot[0, 0], self.rot[1, 0] = 0., 0.
        self.pos[2, 0] = 0.

        # Add measurement noise of 15%
        z_pos = self.pos * (1 + 0.15 * randn(3, 1))
        z_rot = self.rot * (1 + 0.15 * randn(3, 1))

        # Simulate a decreasing/increasing RSSI
        #rssi = 22 - (28.57*math.log10(np.linalg.norm(self.pos - R1)) + 27.06)
        rssi = -100 + .8*self.t_idx - 10*np.random.rand()
        self.t_idx += 1
        #print("Measured RSSI: ", rssi)

        # Generate Observation
        z0 = array([[z_pos[0, 0]], [z_pos[1, 0]], [z_pos[2, 0]], [z_rot[0, 0]], [z_rot[1, 0]], [z_rot[2, 0]], [rssi]])
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
        
        # Visualisation init
        self.blit = blit
        self.handle_scat = None
        self.handle_arrw = None
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
        plt.show(block=False)
        
        # Creating EKF
        self.dt = dt
        self.my_kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        # Create synthetic Pose and Velocity
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
        
        
        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            
            # Populate ONE Rssi for a 'gap' of Poses
            final_pose.append(float(self.rssi_list[-1]))
            
            z = np.asarray(final_pose, dtype=float).reshape(-1, 1)
            #print("Measurement:\n", z)
            # Refresh Measurement noise R
            for j in range(0, 6):
                self.my_kf.R[j, j] = self.sigma_list[-g][j]
                
            # UPDATE
            self.my_kf.update(z, HJacobian_at, hx)
            #print("X-:\n", self.my_kf.x)
            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            
            # PREDICTION
            self.my_kf.predict()
            #print("X+:\n", self.my_kf.x)
        
            print("EKF per round takes %.6f s" % (time.time() - start_t))
        
        
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
        euler_rad = mat2euler(euler_rot)
        #print("Current Eular = ", euler_rad)
        self.odom_quat = np.array(euler2quat(euler_rad[0], euler_rad[1], euler_rad[2]))
        #print("Current Quaternion = ", self.odom_quat)
        
        # Unit Vector from Eular Angle
        self.U = math.cos(euler_rad[2])*math.cos(euler_rad[1])
        self.V = math.sin(euler_rad[2])*math.cos(euler_rad[1])
        self.W = math.sin(euler_rad[1])
        
        print("Elapsed time Pose2TrfMtx = ", time.time() - start_t)
        start_t = time.time()
        
        # Trigger EKF
        self.rt_run(gap)
        print("Elapsed time of EKF = ", time.time() - start_t)
        
        # Trigger EKF
        self.rt_show()
        
        
        
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
        fig1.savefig("sim_example.png")
        
        
    def rt_show(self):
        start_t = time.time()
    
        u, v, w = self.odom_quat[0], self.odom_quat[1], self.odom_quat[2]
    
        self.handle_scat.set_alpha(.2)
        self.handle_arrw.remove()
        self.handle_scat = self.ax21.scatter([self.final_list[-1][0]], [self.final_list[-1][1]], [self.final_list[-1][2]], color='b', marker='o', alpha=.9)
        self.handle_arrw = self.ax21.quiver([self.final_list[-1][0]], [self.final_list[-1][1]], [self.final_list[-1][2]],
            self.U, self.V, self.W, color='r', length=0.25, alpha=.9)
        
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

            # fill in the axes rectangle
            self.fig2.canvas.blit(self.ax21.bbox)
            self.fig2.canvas.blit(self.ax22.bbox)
            
        else:
            self.fig2.canvas.draw()
        
        self.fig2.canvas.flush_events()
    
        stop_t = time.time() - start_t
        print("Elapsed time of VISUALISATION = ", stop_t)
        if stop_t > 0.8:
            self.fig2.savefig("live_rx.png")
            self.reset_view()
            self.set_view([self.final_list[-1][0]], [self.final_list[-1][1]], [self.final_list[-1][2]], self.U, self.V, self.W)
        
    def set_view(self, x=0, y=0, z=0, u=1, v=0, w=0):
        self.ax21.view_init(elev=60., azim=-75)
        self.ax21.set_title("Real-Time Pose", fontweight='bold')
        self.ax21.set_xlabel('X Axis (m)')
        self.ax21.set_ylabel('Y Axis (m)')
        self.ax21.set_zlabel('Z Axis (m)')
        
        self.ax21.set_xlim(-2, 2)
        self.ax21.set_ylim(-2, 2)
        self.ax21.set_zlim(-2, 2)
        
        self.handle_scat = self.ax21.scatter(x, y, z, color='b', marker='o', alpha=.9)
        self.handle_arrw = self.ax21.quiver(x, y, z, u, v, w, color='r', length=0.25, alpha=.9)
        
    def reset_view(self):
        self.rssi_list = []
        self.ax21.clear()


if __name__=="__main__":
    
    dt = 0.1
    ekf = EKF_Fusion(dt=dt, blit=False)
    
    T = 10.
    for i in range(int(T / dt)):
        ekf.sim_run()
        time.sleep(.1)
        
    ekf.sim_show()
    plt.show()


