import os
from os.path import join
import numpy as np
from datetime import datetime
import shutil


DestFile = '2021-03-23_14_09_23_021876'
RxIP_lst = ['93', '94', '95', '96', '97']
MasterIP = 0
#shutil.copyfile(join('TEST', 'test0323', RxIP_lst[MasterIP], DestFile+'.txt'), join('TEST', DestFile + '_left2.txt'))


mf = open(join('TEST', 'test0323', RxIP_lst[MasterIP], DestFile+'.txt'))
master_list = mf.readlines()
mf.close()

for mline in master_list:
    mparts = mline.split(';')
    master_t = mparts[0]
    master_t_obj = datetime.strptime(master_t, '%Y-%m-%d %H:%M:%S.%f')
    master_t_stp = datetime.timestamp(master_t_obj)
    #print(master_t_stp)
    master_rssi = int(mparts[2])

    line_append = np.zeros(5, dtype=int)
    line_append[MasterIP] = master_rssi

    str_rssis = ''
    for ip in RxIP_lst:
        for filename in os.listdir(join('TEST', 'test0323', ip)):
            if filename.startswith(DestFile[:15]) and ip != RxIP_lst[MasterIP]:
                with open(join('TEST', 'test0323', ip, filename), "r") as f:
                    recv_list = f.readlines()

                # Add synthetic RSSIs
                data_len = len(recv_list)
                # print(data_len)

                for item in recv_list:
                    parts = item.split(';')
                    t = parts[0]
                    t_obj = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
                    t_stp = datetime.timestamp(t_obj)
                    if abs(t_stp - master_t_stp) < 0.8:  # Threshold for timestamp matching (s)
                        rssi0 = int(parts[2])
                        if rssi0 < -90 or rssi0 > 0:
                            print("Corrupted RSSI val")
                            rssi0 = 99
                        line_append[RxIP_lst.index(ip)] = rssi0

    with open(join('TEST', DestFile + '_left2.txt'), 'a+') as master_f:
        for val in line_append:
            str_rssis = str_rssis + '; ' + str(val)
        str_line = mparts[0] + ';' + mparts[1] + str_rssis + '\n'
        master_f.write(str_line)
        master_f.flush()


