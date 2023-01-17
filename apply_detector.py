# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:11:15 2023

@author: Administrator
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
import sys
import scipy.io.wavfile as wav
import scipy.io
from scipy import signal
import tensorflow as tf

import random
from skimage.transform import  resize
from functools import partial
import multiprocessing  

import tensorflow as tf
#%%
os.chdir(r'F:\cod_sound_analysis\2018\WA_hydrophones\tutorial')
    

model_cnn  = tf.keras.models.load_model('cod_grunt_tensorflow')


# audio_folder=r'F:\cod_sound_analysis\2018\WA_hydrophones\tutorial'
audiopath_list=glob.glob('*.wav_sg.h5')

index=0
for index in range(len( audiopath_list )) :
    name=audiopath_list[index]
    
    h5f = h5py.File(name,'r')
    x_train_sg = h5f['x_train_sg'][:]
    t1 = h5f['t1'][:]
    t2 = h5f['t2'][:]
    window = h5f['window'][()]
    overlap = h5f['overlap'][()]
    h5f.close()
    
    pred = model_cnn.predict( x_train_sg )    
    # score_predict= pred[pred.argmax(axis=1)]

    
    dfd=pd.Series(pred[:,1])
    slider = int(window/(window*overlap))
    # score = dfd.rolling(slider,center=True).sum() /5
    score = dfd.rolling(slider,center=True).mean() 
    score.index= pd.to_timedelta(t1 + window*0.5 ,'s') 

    score.to_csv(name[:-10]+'_cnn_detections.csv')
