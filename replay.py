import os
import sys
import time
import numpy as np
from EKF import *




class EKF_Fusion_MultiRX(EKF_Fusion):
    def __init__(self, anchor, dt=0.1, visual=True):
        if type(anchor) != int or anchor < 1:
            print("Number of anchor should be integer greater than 0")
            raise TypeError

        super().__init__(anchor=anchor, dim_z=2+anchor, dt=dt, visual=visual)
        # TWEEK PARAMS
        self.my_kf.x = np.array([0., 0., 0., 0.1]).reshape(-1, 1)
        self.my_kf.P = np.diag(np.array([1., 1., 10., 20.]))
        self.my_kf.Q = np.diag(np.array([.01, .01, 1.0, 0.1]))
        
        measure_noise = np.array([1.**2, 1.**2, SIGMA**2])
        for dim in range(1, self.anchor):
            measure_noise = np.hstack((measure_noise, np.array([SIGMA**2])))
        self.my_kf.R = np.diag(measure_noise)
        
    def sim_run(self):
        super(EKF_Fusion_MultiRX, self).sim_run(anchor=self.anchor)
        pass
    
    def rt_show(self):
        super(EKF_Fusion_MultiRX, self).rt_show(t_limit=100.)
        pass
        


class EKF_Fusion_MultiRX_ZYaw(EKF_Fusion):
    def __init__(self, anchor, dt=0.1, visual=True):
        if type(anchor) != int or anchor < 1:
            print("Number of anchor should be integer greater than 0")
            raise TypeError

        super().__init__(anchor=anchor, dim_z=3+anchor, dt=dt, visual=visual)
        # TWEEK PARAMS
        self.my_kf.x = np.array([1., -1., 0., 0.2]).reshape(-1, 1)
        # Process Noise Cov
        self.my_kf.Q = np.diag(np.array([.01, .01, 1.0, 1.0]))
        # Error Cov of Initial State
        self.my_kf.P = np.diag(np.array([1., 1., 0., 1.]))
        # Initial R doesn't matter; it is updated @run_rt
        measure_noise = np.array([1.**2, 1.**2, .1**2, SIGMA**2])
        for dim in range(1, self.anchor):
            measure_noise = np.hstack((measure_noise, np.array([SIGMA**2])))
        self.my_kf.R = np.diag(measure_noise)
        
        
    def rt_run(self, gap):
        
        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            final_xy = final_pose[:2]
            final_xy.append(self.abs_yaw) #Add Yaw as measurement!
            # Populate ONE Rssi for a 'gap' of Poses
            final_xy.append(float(self.rssi_list[-1]))
            z = np.asarray(final_xy, dtype=float).reshape(-1, 1)
            #print("Measurement:\n", z)
            
            # Refresh Measurement noise R
            # Tip1: (x, y) should be noisy; Tip2: large noise for RSSI
            self.my_kf.R[0, 0] = 10.0#self.sigma_list[-g][0]*1000
            self.my_kf.R[1, 1] = 10.0#self.sigma_list[-g][1]*1000
            self.my_kf.R[2, 2] = 1.0#self.sigma_list[-g][5]**2 # Sigma of rot_z or Yaw
            self.my_kf.R[3, 3] = 1.0#5*SIGMA**2
            # Refresh State Transition Martrix: F

            self.my_kf.F = eye(4) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]), self.dt * math.cos(self.my_kf.x[2, 0])],
                                  [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]), self.dt * math.sin(self.my_kf.x[2, 0])],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])

            #self.my_kf.F = eye(4)

            # PREDICTION
            self.my_kf.predict()
            #print("X-:\n", self.my_kf.x)
            
            # UPDATE
            self.my_kf.update(z, HJacobian_at_ZYaw, hx_ZYaw, args=(self.anchor), hx_args=(self.anchor))
            
            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            #print("X+:\n", self.my_kf.x)
            #print("EKF per round takes %.6f s" % (time.time() - start_t))
    
    
    def rt_show(self):
        super(EKF_Fusion_MultiRX_ZYaw, self).rt_show(t_limit=100.)
        pass



class EKF_Fusion_MultiRX_AngularV(EKF_Fusion):
    def __init__(self, anchor, dt=0.1, visual=True):
        if type(anchor) != int or anchor < 1:
            print("Number of anchor should be integer greater than 0")
            raise TypeError

        super().__init__(anchor=anchor, dim_x=5, dim_z=3 + anchor, dt=dt, visual=visual)

        # State Transition Martrix: F
        self.my_kf.F = eye(5)

        # TWEEK PARAMS
        self.my_kf.x = np.array([-1.0, -0.5, 0., 0.1, 0.0]).reshape(-1, 1)
        # Error Cov of Initial State
        self.my_kf.P = np.diag(np.array([4.0, 4.0, 0.1, 1.0, 0.01]))
        # Process Noise Cov
        self.my_kf.Q = np.diag(np.array([0.0, 0.0, 0.0, 0.001, .0001]))

        # Initial R doesn't matter; it is updated @run_rt
        measure_noise = np.array([1. ** 2, 1. ** 2, .1 ** 2, SIGMA ** 2])
        for dim in range(1, self.anchor):
            measure_noise = np.hstack((measure_noise, np.array([SIGMA ** 2])))
        self.my_kf.R = np.diag(measure_noise)

    def smoother(self, lst):
        if len(lst) > 3:
            ave_rssi = 0.6 * lst[-1] + 0.25*lst[-2] + 0.15*lst[-3]
        elif len(lst) == 0:
            return []
        else:
            ave_rssi = lst[-1]
        return float(ave_rssi)

    def rt_run(self, gap):

        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            final_xyZ = final_pose[:2]
            # Add ROT_Z, convert from 'degree' to 'Rad', as measurement!
            final_xyZ.append(final_pose[-1] * math.pi/180)
            # Populate ONE Rssi for a 'gap' of Poses
            final_xyZ.append(self.smoother(self.rssi_list))
            if self.rssi_list2:
                final_xyZ.append(self.smoother(self.rssi_list2))
            if self.rssi_list3:
                final_xyZ.append(self.smoother(self.rssi_list3))

            z = np.asarray(final_xyZ, dtype=float).reshape(-1, 1)
            #print("Measurement z: ", z)
            # TODO Add data integraty check: X+ value explodes

            # Refresh Measurement noise R
            # Tip1: TRANS_X uncertainty larger than TRANS_Y
            # Tip2: Large ROT_Z noise loses abs_yaw; Small ROT_Z noise loses track
            self.my_kf.R[0, 0] = 0.04  # self.sigma_list[-g][0]*1000
            self.my_kf.R[1, 1] = 0.001  # self.sigma_list[-g][1]*1000
            self.my_kf.R[2, 2] = 0.0001  # self.sigma_list[-g][-1]**2 # Sigma of ROT_Z
            for rowcol in range(3, 3+self.anchor):
                self.my_kf.R[rowcol, rowcol] = 4 * SIGMA**2

            # Refresh State Transition Martrix: F
            self.my_kf.F = eye(5) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]),
                                            self.dt * math.cos(self.my_kf.x[2, 0]), 0],
                                           [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]),
                                            self.dt * math.sin(self.my_kf.x[2, 0]), 0],
                                           [0, 0, 0, 0, self.dt],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]]) #Fix a BUG:self.dt multiplied twice

            # PREDICTION
            self.my_kf.predict()
            #print("X-:\n", self.my_kf.x)

            # UPDATE
            self.my_kf.update(z, HJacobian_at_AngularV, hx_AngularV, args=(self.anchor), hx_args=(self.anchor))
            '''
            # IMPOSE CONSTRAINTS
            self.my_kf.x, self.my_kf.P = self.constraints()
            # print("Constraint State = ", self.my_kf.x)
            # print("Constraint Error Cov = ", self.my_kf.P)
            '''
            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            #print("X+:\n", self.my_kf.x)
            # print("EKF per round takes %.6f s" % (time.time() - start_t))

    def constraints(self):
        A = np.diag(np.array([0, 0, 0, 1., 1.]))
        A_1 = np.linalg.pinv(A)
        b = np.array([0, 0, 0, 3., 1*math.pi/2]).reshape((-1, 1))
        x_k = self.my_kf.x
        Pk = self.my_kf.P
        # Constrained State
        AWA = np.dot(A, Pk).dot(A.transpose())
        AWA_1 = np.linalg.pinv(AWA)
        x_pk = x_k - np.dot(Pk, A.transpose()).dot(AWA_1).dot(np.dot(A, x_k) - b)
        '''
        Y = np.dot(Pk, np.eye(5)).dot(A_1)
        x_pk = x_k - np.dot(Y, np.dot(A, x_k) - b)
        '''
        s = x_pk - x_k
        step = 0.01
        for lamda in range(1, 101):
            x_pk0 = x_k + step*lamda*s
            if x_pk0[3, 0] <= b[3, 0] and x_pk0[4, 0] <= b[4, 0]:
                continue
            else:
                x_pk0 = x_k + step*(lamda-1)*s
                print(lamda)
                break

        x_pk = x_pk0
        Pk_p = np.dot(x_pk, x_pk.reshape((1, -1)))
        return x_pk, Pk_p


    def rt_show(self):
        super(EKF_Fusion_MultiRX_AngularV, self).rt_show(t_limit=100.)
        pass


def synthetic_rssi(data_len, period=1., Amp=20., phase=0., mean=-43., noiseAmp=0.2):
    rssi_x = np.arange(0, data_len).tolist()
    rssi_y = [Amp * (math.sin(x / (data_len / period) * 2 * math.pi + phase) + noiseAmp * 2 * (np.random.rand() - 1)) + mean for x in rssi_x]
    '''
    fig3 = plt.figure(3)
    ax31 = fig3.add_subplot(1, 1, 1)
    ax31.plot(rssi_x, rssi_y)
    '''
    return rssi_y


if __name__=="__main__":
    
    ekf = EKF_Fusion_MultiRX_AngularV(anchor=1)
    # TODO Sync Multiple RX RSSIs and Replay


    for filename in os.listdir('TEST'):
        if filename.endswith('.txt'):
            with open('TEST/' + filename, "r") as f:
                recv_list = f.readlines()

            # Add synthetic RSSIs
            data_len = len(recv_list)
            if 0:
                rssi_y2 = synthetic_rssi(data_len=data_len, period=1)
                rssi_y3 = synthetic_rssi(data_len=data_len, period=1, Amp=15, phase=-math.pi/2, noiseAmp=0.3, mean=-45)
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
                msg_list.extend(rssi_list)

                ekf.new_measure(*msg_list)

                #plt.pause(0.01)
                time.sleep(.001)
               
            ekf.fig2.savefig("replay_rx.png")

    '''
            ekf.reset_view()
            ekf.set_view()
    '''
    
    
    


