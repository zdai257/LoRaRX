#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import RPi.GPIO as GPIO
import serial
import time
import sys
import datetime
import struct
import string
import math
#from visualise import *
from EKF import EKF_Fusion


ekf = EKF_Fusion(dt=0.1, visual=True)

def UtcNow():
    now = datetime.datetime.utcnow()
    return str(now)

UTC_LEN = len(UtcNow())
RSSI_REG = [b'\xC0\xC1\xC2\xC3\x00\x02']

def read_rssi(ser):
    ser.write(RSSI_REG[0])
    while True:
        if ser.inWaiting() > 0 :
            print("Checking last msg...")
            time.sleep(0.1)
            r_buff = ser.read(ser.inWaiting())
            rssi_hex = r_buff[-1].encode("hex")
            rssi = int(rssi_hex, base=16) - 256
            print(str(rssi) + " dBm\r\n")
            return rssi

M0 = 22
M1 = 27
MODE = ["BRC","P2P"]
'''
CFG_REG = [b'\xC2\x00\x09\xFF\xFF\x00\x62\x00\x17\x03\x00\x00',
           b'\xC2\x00\x09\x00\x00\x00\x62\x00\x17\x03\x00\x00']
RET_REG = [b'\xC1\x00\x09\xFF\xFF\x00\x62\x00\x17\x03\x00\x00',
           b'\xC1\x00\x09\x00\x00\x00\x62\x00\x17\x03\x00\x00']
'''
CFG_REG = [b'\xC2\x00\x09\x00\x00\x44\x63\x20\x17\x83\x00\x00',
           b'\xC2\x00\x09\x00\x00\x44\x67\x20\x17\x83\x00\x00'] # BRC mode (actually P2P) defined to test Air Speed
RET_REG = [b'\xC1\x00\x09\x00\x00\x44\x63\x20\x17\x83\x00\x00',
           b'\xC1\x00\x09\x00\x00\x44\x67\x20\x17\x83\x00\x00']

if len(sys.argv) != 3 :
    print("there's too much or less arguments,please input again!!!")
    sys.exit(0)
elif (str(sys.argv[1]) == MODE[0]) or (str(sys.argv[1]) == MODE[1]) :
    time.sleep(0.001)
else :
    print("parameters is error")
    sys.exit(0)

# Number of packed floats data; Note if >243 Bytes it splits into TWO messages
num_f = 12*int(sys.argv[2])
len_num_bytes = num_f*4 + math.ceil(num_f/61)
r_buff = b""
msg_buff = b""
buff_len = 0
delay_temp = 1
msg_list = []
log_filename = UtcNow().replace(':', '_')
log_filename = log_filename.replace('.', '_')
log_filename = log_filename.replace(' ', '_')
log_filename += '.txt'
print(log_filename)



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(M0,GPIO.OUT)
GPIO.setup(M1,GPIO.OUT)

GPIO.output(M0,GPIO.LOW)
GPIO.output(M1,GPIO.HIGH)
time.sleep(1)

ser = serial.Serial("/dev/ttyS0",9600)
ser.flushInput()
try :
    if str(sys.argv[1]) == MODE[0] :
        if ser.isOpen() :
            print("It's setting BROADCAST and MONITOR mode")
            ser.write(CFG_REG[0])
        while True :
            if ser.inWaiting() > 0 :
                time.sleep(0.1)
                r_buff = ser.read(ser.inWaiting())
                if r_buff == RET_REG[0] :
                    print("BROADCAST and MONITOR mode was actived")
                    GPIO.output(M1,GPIO.LOW)
                    time.sleep(0.01)
                    r_buff = ""
                if r_buff != "" :
                    print("monitor message:")
                    print(r_buff)

                    now_tx = r_buff[:UTC_LEN]
                    rssi = int(r_buff[-1]) - 256
                    print(now_tx)
                    if now_tx[:3] == b'202':
                        with open(log_filename, "a+") as f:
                            f.write("%s; %s; %d\n" % (now_tx.decode('utf-8'), UtcNow(), rssi))
                            f.flush()
                    r_buff = ""
            delay_temp += 1
            '''
            if delay_temp > 800000: #400000 :
                                msg = "Pose [x y z alpha beta theta] at "+UtcNow()+"\r\n"
                                print(msg)
                                ser.write(msg.encode())
                                delay_temp = 0
            '''
    elif str(sys.argv[1]) == MODE [1]:
        if ser.isOpen() :
            print("It's setting P2P mode")
            ser.write(CFG_REG[1])
        while True :
            if ser.inWaiting() > 0 :
                time.sleep(0.1)
                r_buff = ser.read(ser.inWaiting())
                if r_buff == RET_REG[1] :
                    print("P2P mode was actived")
                    GPIO.output(M1,GPIO.LOW)
                    time.sleep(0.01)
                    r_buff = ""
                if r_buff != "" :
                    
                    now_rx = UtcNow()
                    #print("receive a P2P message at "+now_rx+" :")
                    msg_len = len(r_buff)
                    #print("Received Data Length = ", msg_len)
                    msg = []
                    #print(r_buff)
                    '''
                    for id in range(0, 240, 4):
                        msg0 = r_buff[id:id+4]
                        msg.append(msg0)
                    print(msg)
                    '''
                    # Parse Msg with UtcNow() at Transmitter
                    buff_len += msg_len
                    print("%d bytes in buffer" % buff_len)
                    if buff_len < len_num_bytes : # 241
                        msg_buff += r_buff
                        r_buff = ""
                    elif buff_len == len_num_bytes:
                        msg_buff += r_buff[:-1]
                        #print("Complete Msg = ", msg_buff)
                        
                        rssi = int(r_buff[-1]) - 256
                        
                        print("RSSI = "+str(rssi)+" dBm\r\n")
                        r_buff = ""
                        print("Total Length of Msg = %d\n" % len(msg_buff))
                        # Note bytearray 0-240, 241-481 .. need to be segmented if there are TWO messages
                        if num_f < 61:
                          for id in range(0, 4*num_f, 4):
                            msg0 = struct.unpack('f', msg_buff[id:id+4])
                            msg_list.append(msg0[0])
                        else:
                          for id in range(0, 4*60, 4):
                            msg0 = struct.unpack('f', msg_buff[id:id+4])
                            msg_list.append(msg0[0])
                          for id in range(4*60+1, 4*num_f+1, 4):
                            msg0 = struct.unpack('f', msg_buff[id:id+4])
                            msg_list.append(msg0[0])
                        
                        msg_list.append(rssi)
                        try:
                            # Visulisation
                            #parse_msg(msg_list, rssi, visual=True)
                            # EKF
                            ekf.new_measure(*msg_list)
                        except:
                            print("EKF FAILED\n")
                            
                        # Logging
                        with open(log_filename, "a+") as f:
                            f.write("%s; %s; %d\n" % (now_rx, ','.join('%.8f' % item for item in msg_list[:-1]), rssi))
                            f.flush()
                        msg_list = []
                        msg_buff = b""
                        buff_len = 0
                    else:
                        r_buff = ""
                        msg_buff = b""
                        buff_len = 0
                        msg_list = []
                        print("\nERROR: EXCESSIVE MSG LENGTH\n")
                        #break
            delay_temp += 1
            '''
            if delay_temp > 400000 :
                                msg = "Pose [x y z alpha beta theta] at "+UtcNow()+"\r\n"
                                print(msg)

                                ser.write(msg.encode())
                                delay_temp = 0
            '''
except :
    if ser.isOpen() :
        ser.close()
    GPIO.cleanup()
