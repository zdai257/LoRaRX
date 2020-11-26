import time
import math
import sys
import datetime
import numpy as np
from eulerangles import *
from utility import *
from plot_util import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
import tf


class Pos:
    def __init__(self, blit=False):
        self.pred_transform_t_1 = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
        self.out_pred_array = []
        self.final_list = []
        self.sigma_list = []
        self.angle = 0
        
        self.rssi_list = []
        
        self.vec = np.array([[0], [1], [0], [0]], dtype=float)
        self.odom_quat = self.vec
        
        self.blit = blit
        self.ax1 = None
        self.ax2 = None
        self.handle_scat = None
        self.handle_arrw = None
        self.ax1background = None
        self.ax2background = None
        
        
    def set_view(self, x=0, y=0, z=0, u=1, v=0, w=0):
        self.ax1.view_init(elev=30., azim=-75)
        self.ax1.set_title("Real-Time Pose", fontweight='bold')
        self.ax1.set_xlabel('X Axis (m)')
        self.ax1.set_ylabel('Y Axis (m)')
        self.ax1.set_zlabel('Z Axis (m)')
        '''
        self.ax1.set_xlim(-0, 1)
        self.ax1.set_ylim(-0, 1)
        self.ax1.set_zlim(-0, 1)
        '''
        self.handle_scat = self.ax1.scatter(x, y, z, color='b', marker='o', alpha=.9)
        self.handle_arrw = self.ax1.quiver(x, y, z, u, v, w, color='r', length=0.25, alpha=.9)
        
    def reset_view(self):
        self.rssi_list = []
        self.ax1.clear()



pos = Pos(True)

fig1 = plt.figure(figsize=(8, 9))
pos.ax1 = fig1.add_subplot(2, 1, 1, projection='3d') #fig1.gca(projection='3d') #Axes3D(fig1)
pos.ax2 = fig1.add_subplot(2, 1, 2)
plt.ion()

pos.set_view()

fig1.canvas.draw()
if pos.blit:
    pos.ax1background = fig1.canvas.copy_from_bbox(pos.ax1.bbox)
    pos.ax2background = fig1.canvas.copy_from_bbox(pos.ax2.bbox)
plt.show(block=False)


def parse_msg(ether_msg, rssi, len_pose=12, visual=False):
    start_t = time.time()
    
    pos.rssi_list.append(rssi)
    
    for idx in range(0, len(ether_msg), len_pose):
        final_pose = ether_msg[idx:idx+6]
        sigma_pose = ether_msg[idx+6:idx+12]

        #print(final_pose)
        #print(sigma_pose)
        pos.final_list.append(final_pose)
        pos.sigma_list.append(sigma_pose)

        pred_transform_t = convert_eul_to_matrix(0, 0, 0, final_pose)
        abs_pred_transform = np.dot(pos.pred_transform_t_1, pred_transform_t)
        pos.out_pred_array.append(
        [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
        abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
        abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2], abs_pred_transform[2, 3]])
        pos.pred_transform_t_1 = abs_pred_transform
        
        #pos.odom_quat = tf.transformations.quaternion_from_matrix(pos.pred_transform_t_1)
        #print(pos.odom_quat)

    print(pos.out_pred_array[-1])
    '''
    # Generate static trajectory
    out_pred_array_np = np.array(pos.out_pred_array)
    out_gt_array_np = np.zeros(out_pred_array_np.shape)
    plot2d(out_pred_array_np, out_gt_array_np, join('traj.png'))
    '''
    print("Elapsed time 1 = ", time.time() - start_t)
    start_t = time.time()
    
    euler_rot = np.array([[abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2]],
    [abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2]],
    [abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2]]], dtype=float)
    euler_rad = mat2euler(euler_rot)
    pos.odom_quat = np.array(euler2quat(euler_rad[0], euler_rad[1], euler_rad[2]))
    print(pos.odom_quat)
    
    # Unit Vector from Eular Angle
    U = math.cos(euler_rad[0])*math.cos(euler_rad[1])
    V = math.sin(euler_rad[0])*math.cos(euler_rad[1])
    W = math.sin(euler_rad[1])
    
    print("Elapsed time 2 = ", time.time() - start_t)
    if visual:
        plot_pos()
    
    
def plot_pos():
    start_t = time.time()
    
    u, v, w = pos.odom_quat[1], pos.odom_quat[2], pos.odom_quat[3]
    
    pos.handle_scat.set_alpha(.2)
    pos.handle_arrw.remove()
    pos.handle_scat = pos.ax1.scatter([pos.final_list[-1][0]], [pos.final_list[-1][1]], [pos.final_list[-1][2]], color='b', marker='o', alpha=.9)
    pos.handle_arrw = pos.ax1.quiver([pos.final_list[-1][0]], [pos.final_list[-1][1]], [pos.final_list[-1][2]],
            u, v, w, color='r', length=0.25, alpha=.9)
    
    pos.ax2.clear()
    pos.ax2.set_title("Real-Time LoRa Signal Strength", fontweight='bold')
    pos.ax2.set_ylabel("RSSI (dBm)")
    pos.ax2.plot(pos.rssi_list, 'coral')
    
    
    if pos.blit:
        # restore background
        fig1.canvas.restore_region(pos.ax1background)
        fig1.canvas.restore_region(pos.ax2background)

        # redraw just the points
        pos.ax1.draw_artist(pos.handle_scat)
        pos.ax1.draw_artist(pos.handle_arrw)

        # fill in the axes rectangle
        fig1.canvas.blit(pos.ax1.bbox)
        fig1.canvas.blit(pos.ax2.bbox)

    else:
        fig1.canvas.draw()
        
    fig1.canvas.flush_events()
    
    stop_t = time.time() - start_t
    print("Elapsed time 3 = ", stop_t)
    if stop_t > 1.0:
        fig1.savefig("live_rx.png")
        pos.reset_view()
        pos.set_view([pos.final_list[-1][0]], [pos.final_list[-1][1]], [pos.final_list[-1][2]], u, v, w)
    
