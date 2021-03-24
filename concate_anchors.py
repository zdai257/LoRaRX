import os
from os.path import join
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

    line_append = ''
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
                    if abs(t_stp - master_t_stp) < 0.7:
                        rssi0 = int(parts[2])
                        if rssi0 < -90 or rssi0 > 0:
                            print("Corrupted RSSI val")
                            rssi0 = 99
                        line_append = line_append + '; ' + str(rssi0)

                        if ip == RxIP_lst[-1]:
                            with open(join('TEST', DestFile + '_left2.txt'), 'a+') as master_f:
                                master_f.write(mline[:-1] + line_append + '\n')
                                master_f.flush()
                                break
