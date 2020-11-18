"""
Test the model using two thermal images as input, 20 imu data as input
"""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
import numpy as np
from eulerangles import euler2quat, euler2mat
import math
import imageio 

SCALER = 1.0 # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi

def get_image(img_path):
    img = imageio.imread(img_path)
    img = img.astype('float32')
    img = (img - 21828.0) * 1.0 / (26043.0 - 21828.0)
    img -= 0.17684562275397941
    img = np.expand_dims(img, axis=-1)
    return img

def load_normalize_1channel_img(img):
    min_range = 0
    max_range = 255
    '''
    # master_path = data_dir + '/' + sampled_files[k].split(',')[0] # idx 0 is always for the master!
    img_path = data_dir + '/' + img_name
    # normalize master image
    img = misc.imread(img_path)
    '''
    img = img.astype('float32')
    img = (img - min_range) * 1.0 / (max_range - min_range)
    img -= 0.04128679635635311
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0) # add dimension for timestamp
    return img

def transform44(l):
    _EPS = np.finfo(float).eps * 4.0
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)


def convert_eul_to_matrix(rot_x, rot_y, rot_z, pose):
    R_pred = euler2mat(rot_x, rot_y, rot_z)
    rotated_pose = np.dot(R_pred, pose[0:3])
    DEGREE_2_RADIUS = np.pi / 180.0
    pred_quat = euler2quat(z=pose[5] * DEGREE_2_RADIUS, y=pose[4] * DEGREE_2_RADIUS,
                           x=pose[3] * DEGREE_2_RADIUS)
    pred_transform_t = transform44([0, rotated_pose[0], rotated_pose[1], rotated_pose[2],
                                    pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
    return pred_transform_t

def convert_eul_to_matrix_2D(rot_x, rot_y, rot_z, pose):
    R_pred = euler2mat(rot_x, rot_y, rot_z)
    rotated_pose = np.dot(R_pred, [pose[0], pose[1], 0])
    DEGREE_2_RADIUS = np.pi / 180.0
    pred_quat = euler2quat(z=pose[2] * DEGREE_2_RADIUS, y=0 * DEGREE_2_RADIUS,
                           x=0 * DEGREE_2_RADIUS)
    pred_transform_t = transform44([0, rotated_pose[0], rotated_pose[1], rotated_pose[2],
                                    pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
    return pred_transform_t