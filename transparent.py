#!/usr/bin/python
# -*- coding: UTF-8 -*-

import RPi.GPIO as GPIO
import serial
import time
import sys
import datetime

def UtcNow():
    now = datetime.datetime.utcnow()
    return str(now)

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
CFG_REG = [b'\xC2\x00\x09\xFF\xFF\x44\x67\x20\x17\x83\x00\x00',
           b'\xC2\x00\x09\x00\x00\x44\x67\x20\x17\x83\x00\x00']
RET_REG = [b'\xC1\x00\x09\xFF\xFF\x44\x67\x20\x17\x83\x00\x00',
           b'\xC1\x00\x09\x00\x00\x44\x67\x20\x17\x83\x00\x00']
r_buff = ""
delay_temp = 1

if len(sys.argv) != 2 :
    print("there's too much or less arguments,please input again!!!")
    sys.exit(0)
elif (str(sys.argv[1]) == MODE[0]) or (str(sys.argv[1]) == MODE[1]) :
    time.sleep(0.001)
else :
    print("parameters is error")
    sys.exit(0)
    
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
                    
                    r_buff = ""
            delay_temp += 1
            if delay_temp > 800000:#400000 :
                                msg = "Pose [x y z alpha beta theta] at "+UtcNow()+"\r\n"
                                print(msg)
                                ser.write(msg.encode())
                                delay_temp = 0
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
                    print("receive a P2P message at "+now_rx+" :")
                    print(r_buff)
                    '''
                    now_tx = r_buff[0:0+len(now_rx)]
                    print(now_tx)
                    msg = r_buff[1+len(now_rx):-2]
                    print(msg)
                    '''
                    #print(r_buff[-3:-2])
                    if r_buff[-3:-2] != b'\r':
                        now_tx = r_buff[0:0+len(now_rx)]
                        #print(now_tx)
                        msg = r_buff[1+len(now_rx):]
                        #print(msg)
                        msg_buff = msg
                        r_buff = ""
                    else:
                        msg = r_buff[:-3]
                        msg_buff += msg
                        print("Complete Msg = ", msg_buff)
                        #print(str(r_buff[-1]))
                        rssi = int(r_buff[-1]) - 256
                        #rssi_hex = r_buff[-1].encode("hex") # Python2 needs Encoding, but not Python3!
                        #rssi = int(rssi_hex, base=16) - 256
                        print("RSSI = "+str(rssi)+" dBm\r\n")
                        r_buff = ""
                        #read_rssi(ser)
                        with open("log.txt", "a+") as f:
                            f.write("%s %s %s %d\n" % (now_tx.decode("utf-8"), now_rx, msg_buff.decode("utf-8"), rssi))
                        msg_buff = ""
                    
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
