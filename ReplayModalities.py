import os
from os.path import join
import numpy as np
import time
from replay import EKF_Origin




def main():

    ekf = EKF_Origin(anchor=5, ismdn=False, dense=False)

    # Sync Multiple RX RSSIs and Replay
    RxIP_lst = ['93', '94', '95', '96', '97']

    for filename in os.listdir('TEST'):
        if filename.endswith('.txt'):
            with open(join('TEST', filename), "r") as f:
                recv_list = f.readlines()

            # Add synthetic RSSIs
            data_len = len(recv_list)
            rssi_idx = 0

            for item in recv_list:
                rssi_list = []
                parts = item.split(';')
                t = parts[0]

                msgs = parts[1]
                vals = msgs.split(',')

                for rssi_id in range(2, 2 + ekf.anchor):
                    rssi0 = int(parts[rssi_id])
                    rssi_list.append(rssi0)

                msg_list = [float(i) for i in vals]
                if ekf.anchor:
                    msg_list.extend(rssi_list)

                ekf.new_measure(*msg_list)

                # plt.pause(0.01)
                time.sleep(.001)
                print("RMSE between traj1 & 2 = %.4f m" % ekf.rms_traj())

            #ekf.fig2.savefig("replay_ekf.png")

    '''
            ekf.reset_view()
            ekf.set_view()
    '''


if __name__=="__main__":

    main()
