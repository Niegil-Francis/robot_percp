import requests
import re
import os
import zipfile
import collections
import urllib.request
from collections import defaultdict
from IPython.core.display import HTML
from collections import defaultdict
import time
import random

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import sys    
sys.path.insert(1, './imports')
import UdpComms as U
import time

import threading
import time
import signal
# Testing the saved model
# model_load = load_model('/content/drive/MyDrive/Robot perception /Modeling/models/lstm_data_trim.h5')
model_load = load_model('./models/sep_1/lstm_rnn.h5')
# model_load = load_model('./models/rnn.h5')
# model_load.evaluate(x_test,y_test)
print(model_load.summary())
data_ =[]

interrupted = False
reset = False
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
target_points = np.array([[-0.314,1.661,0.45],[0,1.661,0.45],[0.314,1.661,0.45],[-0.314,1.347,0.45],[0,1.347,0.45],[0.314,1.347,0.45],[-0.314,1.033,0.45],[0,1.033,0.45],[0.314,1.033,0.45]])
r= 0.9
temp = []
for i in range(9): 
    temp.append(np.array([target_points[i][0]*r,(target_points[i][1]-target_points[4][1])*r+target_points[4][1], target_points[i][2]]))
target_points = temp
def model(ip_data):

    t_interval = 1
    p = 2
    hand_position = np.asarray(ip_data)
    si = len(hand_position)
    if si <= 2:
        return -1
    else:
        f = 0
        P = np.zeros([9])
        current_hand_position = hand_position[-1]
        previous_hand_position = hand_position[-p]          
        for i in range(0,9):
            g = (np.linalg.norm(np.asarray(target_points[i]) - np.asarray(current_hand_position)) - np.linalg.norm(np.asarray(target_points[i]) - np.asarray(previous_hand_position)))/t_interval
            if g<0:
                f = -g
            else:
                f = 0
            P[i] = f
        target = np.argmax(P)

    return target

def thread_1():
    global interrupted
    global data_
    global sock
    old = [0.,0.,0.]
    while True:
        time.sleep(0.01)
        if len(data_) > 1:
            if old == data_[-1]:
                continue
            else:
                old = data_[-1]
                data_ip = tf.expand_dims(tf.convert_to_tensor(np.array(data_)),axis =0)
                pred = model(data_)
                print(pred)
                if pred != -1:
                    sock.SendData(str(pred))
        if interrupted:
            break

        

def thread_2():
    global interrupted
    global data_
    global sock
    while True:
        
        data = sock.ReadReceivedData() # read data
        
        if data != None: # if NEW data has been received since last ReadReceivedData function call
            if data == "reset":
                data_ = []
                print("Resetted")
                continue
            data.split(",")
            dt = [float(i) for i in data.split(",") if len(i)]
            cent = [np.mean(dt[0::3]),np.mean(dt[1::3]),np.mean(dt[2::3])]
            if len(data_) == 150:
                data_.pop(0)
            data_.append(cent)

        if interrupted:
            break
        
        time.sleep(0.01)

def thread_3():
    global interrupted
    keystrk=input('Press a key \n')
    # thread doesn't continue until key is pressed
    interrupted=True
    print('Interrupted!')


x_1 = threading.Thread(target=thread_1, args=())
x_2 = threading.Thread(target=thread_2, args=())
x_3 = threading.Thread(target=thread_3, args=())

x_1.start()
x_2.start()
x_3.start()

x_1.join()
x_2.join()
x_3.join()