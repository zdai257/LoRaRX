import os
from os.path import join
import numpy as np
import math
import time
from replay import EKF_Origin
from EKF import HJacobian_Origin, hx_Origin


def compute_rmse(path_s, path_l, delay):

    path_length = len(path_s)
    path_lt = path_l[delay:delay + path_length]
    sum_sqr = 0
    for i in range(0, path_length):
        sum_sqr += (path_lt[i][0] - path_s[i][0]) ** 2 + (path_lt[i][1] - path_s[i][1]) ** 2

    RMSE = np.sqrt(sum_sqr / path_length)
    #print("\nRMSE = %.5f m" % RMSE)
    return RMSE


class EKF_OriginFusion(EKF_Origin):
    def __init__(self, anchor, anchorLst, dt=0.1, ismdn=False, visual=True, dense=False):
        # Xk = [x, y, theta]
        super().__init__(anchor=anchor, anchorLst=anchorLst, dt=dt, ismdn=ismdn, visual=visual, dense=dense)
        # State Transition Martrix: F
        self.my_kf.F = np.eye(3)

        # TWEEK PARAMS
        self.my_kf.x = np.array([-0.1, -0.1, 0.]).reshape(-1, 1)
        # Error Cov of Initial State
        self.my_kf.P = np.diag(np.array([10.0, 10.0, 1.0]))
        # Process Noise Cov
        self.my_kf.Q = np.diag(np.array([0.1, 0.1, 0.01]))

    def rt_run(self, gap):

        for g in range(gap, 0, -1):
            start_t = time.time()
            # Get Measurement [ABS_X, ABS_Y, ABS_YAW('Rad')]
            final_xyZ = [self.abs_x[-g], self.abs_y[-g], self.abs_yaw[-g]]
            # Populate ONE Rssi for a 'gap' of Poses
            # Apply 'ave10' mode on every EKF iteration to smooth the path
            for anchor_idx in range(0, self.anchor):
                self.rssi_dict_smth[anchor_idx][-1] = self.smoother(self.rssi_dict[anchor_idx], g=g, mode='ave10')
                final_xyZ.append(self.rssi_dict_smth[anchor_idx][-1])

            z = np.asarray(final_xyZ, dtype=float).reshape(-1, 1)
            #print("Measurement z: ", z)

            # Refresh Measurement noise R
            rot_z = self.final_list[-g][-1]
            #print(-abs(rot_z))
            R_scalar = 10 * (-math.e ** (-0.2 * abs(rot_z)) + 1.)
            #print(R_scalar)
            self.my_kf.R[0, 0] = 0.15 * 1  # ABS_X
            self.my_kf.R[1, 1] = 0.15 * 1  # ABS_Y
            self.my_kf.R[2, 2] = 1. * R_scalar  # ABS_YAW
            for rowcol in range(3, 3+self.anchor):
                self.my_kf.R[rowcol, rowcol] = 0.2 * 4.887**2

            # PREDICTION
            self.my_kf.predict()
            #print("X-:\n", self.my_kf.x)

            # UPDATE
            self.my_kf.update(z, HJacobian_Origin, hx_Origin, args=(self.anchorLst), hx_args=(self.anchorLst))

            # Log Posterior State x
            self.xs.append(self.my_kf.x)
            #print("X+:\n", self.my_kf.x)
            # print("EKF per round takes %.6f s" % (time.time() - start_t))
            if self.visual and self.dense:
                self.rt_show(odom_idx=-g, mark_size=10)


def main():
    # Specify StaticIP of Anchors that participate in computation
    RxIP_lst = ['94', '95', '97']
    RxLst = [int(idx) - 93 for idx in RxIP_lst]

    ekf = EKF_OriginFusion(anchor=len(RxIP_lst), anchorLst=RxLst, ismdn=False, dense=True)

    for filename in os.listdir('TEST'):
        if filename.endswith('.txt'):
            with open(join('TEST', filename), "r") as f:
                recv_list = f.readlines()

            # Add synthetic RSSIs
            data_len = len(recv_list)
            print("Odometry Data Length = {}".format(data_len * 10))
            recv_idx = 0

            for item in recv_list:
                rssi_list = []
                parts = item.split(';')
                t = parts[0]

                msgs = parts[1]
                vals = msgs.split(',')

                for rssi_id in ekf.anchorLst:
                    rssi0 = int(parts[rssi_id + 2])
                    rssi_list.append(rssi0)

                msg_list = [float(i) for i in vals]
                if ekf.anchor:
                    msg_list.extend(rssi_list)

                ekf.new_measure(*msg_list)

                # plt.pause(0.01)
                time.sleep(.001)
                print("RMSE between traj1 & 2 = %.4f m" % ekf.rms_traj())
                recv_idx += 1
                #print(ekf.xs)

                if recv_idx > 120:
                    break


            ekf.fig2.savefig("replay_ekf.png")

            print("Path Length = %d; GT Length = %d" % (len(ekf.path_dense), len(ekf.gt_path)))
            max_delay = abs(len(ekf.path_dense) - len(ekf.gt_path))

            path_lora = [[item[0, 0], item[1, 0]] for item in ekf.xs]
            RMSE_odom, RMSE_lora = [], []

            for delay in range(0, max_delay):
                rmse0 = compute_rmse(path_s=ekf.gt_path, path_l=ekf.path_dense, delay=delay)
                rmse1 = compute_rmse(path_s=ekf.gt_path, path_l=path_lora, delay=delay)
                RMSE_odom.append(rmse0)
                RMSE_lora.append(rmse1)

            best_RMSE_lora, best_delay = min((val, idx) for (idx, val) in enumerate(RMSE_lora))
            corr_RMSE_odom = RMSE_odom[best_delay]
            print("\nBest RMSE = %.5f with delay = %d\nCorresponding Odometry RMSE = %.5f" %
                  (best_RMSE_lora, best_delay, corr_RMSE_odom))


if __name__=="__main__":

    main()
