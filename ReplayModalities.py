import os
from os.path import join
import numpy as np
import time
from replay import EKF_Origin










def main():

    ekf = EKF_Origin(anchor=1, ismdn=False, dense=False)

    # Sync Multiple RX RSSIs and Replay
    RxIP_lst = ['93', '94', '95', '96', '97']

    for ip in RxIP_lst:
      for filename in os.listdir(join('TEST', 'test0323', ip)):
        if filename.startswith('2021-03-23_14_09'):
            with open(join('TEST', 'test0323', ip, filename), "r") as f:
                recv_list = f.readlines()

            # Add synthetic RSSIs
            data_len = len(recv_list)
            if 0:
                rssi_y2 = synthetic_rssi(data_len=data_len, period=1)
                rssi_y3 = synthetic_rssi(data_len=data_len, period=1, Amp=15, phase=-math.pi / 2, noiseAmp = 0.3, mean = -45)
            else:
                rssi_y2, rssi_y3 = [], []
            print(data_len)
            rssi_idx = 0

            for item in recv_list:
                rssi_list = []

                parts = item.split(';')
                t = parts[0]
                msgs = parts[1]
                vals = msgs.split(',')
                rssi1 = int(parts[2])
                if rssi1 < -90 or rssi1 > 0:
                    print("Corrupted RSSI val")
                    break

                # Append RXs measurements
                rssi_list.append(rssi1)
                if rssi_y2:
                    rssi_list.append(rssi_y2[rssi_idx])
                if rssi_y3:
                    rssi_list.append(rssi_y3[rssi_idx])
                rssi_idx += 1

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

