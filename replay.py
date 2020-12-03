import sys
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from EKF import *





class EKF_Fusion_MultiRX(EKF_Fusion):
    def __init__(self, anchor):
        if type(anchor) != int or anchor < 1:
            print("Number of anchor should be integer greater than 0")
            raise TypeError
        self.anchor = anchor
        super().__init__(dim_z=2+self.anchor)
        
        #self.my_kf = ExtendedKalmanFilter(dim_x=4, dim_z=2+self.anchor)
        measure_noise = np.array([.1**2, .1**2, 4.887**2])
        for dim in range(1, self.anchor):
            measure_noise = np.hstack((measure_noise, np.array([4.887**2])))
        self.my_kf.R = np.diag(measure_noise)
        
    def sim_run(self):
        super(EKF_Fusion_MultiRX, self).sim_run(anchor=self.anchor)
        pass
        




if __name__=="__main__":
    
    ekf = EKF_Fusion_MultiRX(anchor=3)
    
    # TODO replay the logged LoRa RX messages

    dt = 0.1
    
    T = 10.
    for i in range(int(T / dt)):
        ekf.sim_run()
        time.sleep(.1)
    
    
    ekf.sim_show()
    
    


