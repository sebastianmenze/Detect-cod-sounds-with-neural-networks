# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:10:08 2023

@author: Administrator
"""


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os


from scipy.signal import find_peaks


#%%
plt.figure(8)
plt.clf()


audio_folder=r'F:\cod_sound_analysis\2018\WA_hydrophones\tutorial'
audiopath_list=glob.glob(audio_folder+'\*.wav_sg.h5')

index=0
for index in range(len( audiopath_list )) :
    name=audiopath_list[index]
    
    h5f = h5py.File(name,'r')
    t1 = h5f['t1'][:]
    t2 = h5f['t2'][:]
    window = h5f['window'][()]
    overlap = h5f['overlap'][()]
    h5f.close()
    
    score = pd.read_csv(name[:-10]+'_cnn_detections.csv',index_col=0  )
    score.index= pd.to_timedelta(score.index)

    reclength = score.index.total_seconds().max()-overlap

    df=pd.read_csv(name[:-14]+'.csv')
    t_ann=df['tmin'] + (df['tmax']-df['tmin'])*0.5
    
    
    t1=np.arange(0,reclength-window,window)
    t2=t1+window
    
    score_annotation=np.zeros(len(t1))
    for ix in range(len(t1)):      
        sumcall =   np.sum( (t_ann>=t1[ix]) & (t_ann<t2[ix]) )
        if sumcall>0:
            score_annotation[ix]=1
            
    np.sum(score_annotation)       
    
    n=50
    
    tr = np.linspace(0,1,n)
    fpr=np.empty(n)
    
    tpr=np.empty(n)
    i=0
    for threshold in tr: 
        
        ix_p,b= find_peaks(score['0'].values,height=threshold)
        # score_detector=np.zeros(len(t1))
        # score_detector[ix_p] = 1
        t_det = score.index[ix_p].total_seconds()
    
        score_detector=np.zeros(len(t1))
        for ix in range(len(t1)):      
            sumcall =   np.sum( (t_det>=t1[ix]) & (t_det<t2[ix]) )
            if sumcall>0:
                score_detector[ix]=1    
            
        
        tpr[i]= np.sum( (score_detector==1) & (score_annotation==1)) / np.sum(score_annotation)
        
        fpr[i]= np.sum( (score_detector==1) & (score_annotation==0)) / np.sum( score_annotation==0 )
        i=i+1
        # print('tpr '+str(tpr) + ' fpr '+str(fpr))    
    df_roc=pd.DataFrame([])
    df_roc['fpr']=fpr
    df_roc['tpr']=tpr
    df_roc['threshold']=tr
    df_roc.to_csv(name[:-8]+'_roc.csv')
    
    plt.plot(fpr,tpr,label=name)
    for i, txt in enumerate(tr):
        plt.annotate("{:.2f}".format(txt), ( fpr[i] , tpr[i]))


plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.grid()
       
# plt.legend()

plt.savefig('roc.png')