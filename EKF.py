import math
import numpy as np
import time
from math import sqrt
from numpy.random import randn
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
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
    Jacob = array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [a*(X - R1[0])/denom, a*(Y - R1[1])/denom, a*(Z - R1[2])/denom, 0, 0, 0, 0, 0, 0]])
    #return array ([[(X - R1[0])/denom, (Y - R1[1])/denom, (Z - R1[2])/denom, 0, 0, 0, 0, 0, 0]]) * 28.57*math.log10(math.e)
    # HJabobian in (7, 9) if ONE LoRa RX
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



class EKF_Fusion(object):
    def __init__(self, dt=0.1, dim_x=9, dim_z=7):
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


    def sim_run(self, T=10.):
        for i in range(int(T / self.dt)):
            # start_t = time.time()
            z = self.pv.get_measure()
            # Log track or GroundTruth
            self.track.append((self.pv.pos, self.pv.rot, self.pv.vel))

            # Refresh measurement noise R at runtime
            for j in range(0, 6):
                self.my_kf.R[j, j] = np.random.rand()

            # UPDATE
            self.my_kf.update(z, HJacobian_at, hx)

            # Log Posterior State x
            self.xs.append(self.my_kf.x)

            # PREDICTION
            self.my_kf.predict()

            # print("EKF process time = %.7f s" % (time.time() - start_t))

        self.xs = asarray(self.xs)
        self.track = asarray(self.track)
        self.time = np.arange(0, len(self.xs) * self.dt, self.dt)
        # print(track)
        # print(xs)

    def sim_show(self):
        x0 = [item[0][0] for item in self.xs]
        y0 = [item[1][0] for item in self.xs]

        x_gt = [item[0][0, 0] for item in self.track]
        y_gt = [item[0][1, 0] for item in self.track]

        plt.plot(x0, y0, 'r')
        plt.plot(x_gt, y_gt, 'b')
        plt.show()



ekf = EKF_Fusion()
ekf.sim_run(20.)
ekf.sim_show()


