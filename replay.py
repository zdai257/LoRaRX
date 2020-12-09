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
        self.anchor = anchor
        super().__init__(dim_z=2+self.anchor, dt=dt, visual=visual)
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
        self.anchor = anchor
        super().__init__(dim_z=3+self.anchor, dt=dt, visual=visual)
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
                                  [0, 0, 0, 0]]) * self.dt

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
        self.anchor = anchor
        super().__init__(dim_x=5, dim_z=3 + self.anchor, dt=dt, visual=visual)
        # State Transition Martrix: F
        self.my_kf.F = eye(5)

        # TWEEK PARAMS
        self.my_kf.x = np.array([0., 0., 0., 0.01, 0.0]).reshape(-1, 1)
        # Error Cov of Initial State
        self.my_kf.P = np.diag(np.array([0.0, 0.0, 0.0, 1.0, 0.01]))
        # Process Noise Cov
        self.my_kf.Q = np.diag(np.array([0.0, 0.0, 0.0, 0.01, .0001]))

        # Initial R doesn't matter; it is updated @run_rt
        measure_noise = np.array([1. ** 2, 1. ** 2, .1 ** 2, SIGMA ** 2])
        for dim in range(1, self.anchor):
            measure_noise = np.hstack((measure_noise, np.array([SIGMA ** 2])))
        self.my_kf.R = np.diag(measure_noise)

    def rt_run(self, gap):

        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement
            final_pose = self.final_list[-g]
            final_xyZ = final_pose[:2]
            # Add ROT_Z, convert from 'degree' to 'Rad', as measurement!
            final_xyZ.append(final_pose[5] * math.pi/180)
            # Populate ONE Rssi for a 'gap' of Poses
            final_xyZ.append(float(self.rssi_list[-1]))
            z = np.asarray(final_xyZ, dtype=float).reshape(-1, 1)
            #print("Measurement ROT_Z:", z[2, 0])

            # Refresh Measurement noise R
            # Tip1: (x, y) should be noisy; Tip2: large noise for RSSI
            self.my_kf.R[0, 0] = 4.0  # self.sigma_list[-g][0]*1000
            self.my_kf.R[1, 1] = 4.0  # self.sigma_list[-g][1]*1000
            self.my_kf.R[2, 2] = .01  # self.sigma_list[-g][5]**2 # Sigma of ROT_Z
            self.my_kf.R[3, 3] = 300.0  # 5*SIGMA**2
            # Refresh State Transition Martrix: F
            self.my_kf.F = eye(5) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]),
                                            self.dt * math.cos(self.my_kf.x[2, 0]), 0],
                                           [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]),
                                            self.dt * math.sin(self.my_kf.x[2, 0]), 0],
                                           [0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]]) * self.dt

            # PREDICTION
            self.my_kf.predict()
            # print("X-:\n", self.my_kf.x)

            # UPDATE
            self.my_kf.update(z, HJacobian_at_AngularV, hx_AngularV, args=(self.anchor), hx_args=(self.anchor))

            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            # print("X+:\n", self.my_kf.x)
            # print("EKF per round takes %.6f s" % (time.time() - start_t))

    def rt_show(self):
        super(EKF_Fusion_MultiRX_AngularV, self).rt_show(t_limit=100.)
        pass


        

if __name__=="__main__":
    
    ekf = EKF_Fusion_MultiRX_AngularV(anchor=1)
    # TODO replay the logged LoRa RX messages
    dt = 0.1
    # SIMULATION
    '''
    T = 10.
    for i in range(int(T / dt)):
        ekf.sim_run()
        time.sleep(.1)
    ekf.sim_show()
    '''
    
    for filename in os.listdir('TEST'):
        if filename.endswith('.txt'):
            with open('TEST/' + filename, "r") as f:
                recv_list = f.readlines()
                
            for item in recv_list:
                parts = item.split(';')
                t = parts[0]
                msgs = parts[1]
                vals = msgs.split(',')[:-1]
                rssi = int(parts[2])
                
                msg_list = [float(i) for i in vals]
                msg_list.append(rssi)
                ekf.new_measure(*msg_list)
                
                time.sleep(.001)
               
            ekf.fig2.savefig("replay_rx.png")
            plt.show()
            '''
            ekf.reset_view()
            ekf.set_view()
            '''
    
    
    
    


