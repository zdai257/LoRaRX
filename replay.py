import sys
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from EKF import EKF_Fusion



class EKF_Fusion2(EKF_Fusion):
    def __init__(self):
        super().__init__()
        self.my_kf = ExtendedKalmanFilter(dim_x=4, dim_z=4)
    

ekf = EKF_Fusion2()
print(ekf.my_kf.P)
print(ekf.visual)


# TODO replay the logged LoRa RX messages



