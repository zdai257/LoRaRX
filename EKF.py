import os
import sys
import numpy as np
import math
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

my_filter = ExtendedKalmanFilter(dim_x=2, dim_z=1)


