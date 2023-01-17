# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:46:12 2023

@author: Administrator
"""


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

import tensorflow as tf
from scipy.signal import find_peaks


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
    
    
    detections=pd.DataFrame()
    detections.index= score.index
   
    detections['cod']=0
  
    threshold = 0.4   
    ix_peaks , b =  find_peaks(score.values,height=threshold)
    detections.iloc[ix_peaks,0]=1
   
    a = detections['cod'].resample('1min').mean()
    a.index=pd.Timestamp(2000,1,1) + a.index
    
    plt.figure(0)
    plt.clf()
    plt.plot( a)
    plt.grid()
    plt.ylabel( 'Detections per minute' )
    plt.savefig(name[:-10]+'_detections.png')

    detections.to_csv(name[:-10]+'_detections.csv')