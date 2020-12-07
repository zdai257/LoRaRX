import os
import sys
import time
import numpy as np
from EKF import *



def HJacobian_at_ZYaw(x, anchor=1):
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
    a = ALPHA*math.log10(math.e)
    # HJabobian in (4, 4) if ONE LoRa RX; (6, 4) if THREE LoRa RXs available
    Jacob = array([[0, 0, -dt * V * math.sin(theta), dt * math.cos(theta)],
                   [0, 0, dt * V * math.cos(theta), dt * math.sin(theta)],
                   [0, 0, 1, 0]])
    for row in range(0, anchor):
        Jacob = np.vstack((Jacob, array([[a*(X - R1[row, 0])/denom, a*(Y - R1[row, 1])/denom, 0, 0]])))
    #print("HJacobian return: ", Jacob)
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
        thres_dis = 0.5
        if dis > thres_dis:
            # RSSI Regression Model
            rssi = ALPHA*math.log10(dis) - BETA
        else:
            rssi = ALPHA*math.log10(thres_dis) - BETA
        
        # Measurement comprises (X, Y, abs_Yaw, RSSIs...)
        h = np.vstack((h, array([[rssi]])))
    #print("hx return shape: ", h.shape)
    return h



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
        self.my_kf.x = np.array([0., 0., 0., 0.2]).reshape(-1, 1)
        # Error Cov of Initial State
        self.my_kf.Q = np.diag(np.array([.01, .01, 1.0, 1.0]))
        # Process Noise Cov
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
            self.my_kf.R[0, 0] = 10.#self.sigma_list[-g][0]**2
            self.my_kf.R[1, 1] = 10.#self.sigma_list[-g][1]**2
            self.my_kf.R[2, 2] = 0.01#self.sigma_list[-g][5]**2 # Sigma of rot_z or Yaw
            self.my_kf.R[3, 3] = 10*SIGMA**2
            # Refresh State Transition Martrix: F
            self.my_kf.F = eye(4) + array([[0, 0, -self.dt * self.my_kf.x[3, 0] * math.sin(self.my_kf.x[2, 0]), self.dt * math.cos(self.my_kf.x[2, 0])],
                                  [0, 0, self.dt * self.my_kf.x[3, 0] * math.cos(self.my_kf.x[2, 0]), self.dt * math.sin(self.my_kf.x[2, 0])],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]) * self.dt
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
        
        
        
        

if __name__=="__main__":
    
    ekf = EKF_Fusion_MultiRX_ZYaw(anchor=1)
    
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
            '''
            ekf.reset_view()
            ekf.set_view()
            '''
    
    
    
    


