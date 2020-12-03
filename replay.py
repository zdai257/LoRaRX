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
        self.my_kf.x = np.array([0., 0., math.pi/4, 0.1]).reshape(-1, 1)
        self.my_kf.P = np.diag(np.array([1., 1., 10., 20.]))
        self.my_kf.Q = np.diag(np.array([.01, .01, 1.0, 0.1]))
        
        measure_noise = np.array([1.**2, 1.**2, 4.887**2])
        for dim in range(1, self.anchor):
            measure_noise = np.hstack((measure_noise, np.array([4.887**2])))
        self.my_kf.R = np.diag(measure_noise)
        
    def sim_run(self):
        super(EKF_Fusion_MultiRX, self).sim_run(anchor=self.anchor)
        pass
    
    def rt_show(self):
        super(EKF_Fusion_MultiRX, self).rt_show(t_limit=100.)
        pass
        




if __name__=="__main__":
    
    ekf = EKF_Fusion_MultiRX(anchor=1)
    
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
    
    
    
    


