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
model_load = load_model('./models/sep_1/lstm.h5')
# model_load = load_model('./models/rnn.h5')
# model_load.evaluate(x_test,y_test)
print(model_load.summary())
data_ =[]

interrupted = False
reset = False
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
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
                ip_data = tf.expand_dims(tf.transpose(tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(np.array(data_).T, padding='post', dtype='float', maxlen=200))),axis =0)
                prediction = model_load.predict(ip_data,verbose = 0)
                conf = prediction[0][len(data_)-1][np.argmax(prediction,axis=2)[0][-1]]
                pred = np.argmax(prediction,axis=2)[0][-1]
                if conf >= 0.85:
                    print(pred)
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