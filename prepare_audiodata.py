# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:49:47 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:11:58 2023

@author: Administrator
"""


import h5py
import pandas as pd
import numpy as np
import glob 
import os
import scipy.io.wavfile as wav
from scipy import signal

from skimage.transform import  resize
from functools import partial
import multiprocessing  



#%%


def parafunc(afiles,window=3,overlap= 0.2,fftsize=16000,fmin=10, fmax=120,fileindex=0):
    
    name= afiles[fileindex]
    
    if not os.path.isfile(  os.path.basename(name)+'_sg.h5' ):
        print(name)
      
        # name='20180306_062633.wav'
        fs, p = wav.read(name)
        
        os.path.basename(name)[:-4]
        
            
        reclength= len(p)/fs
        
        # window = 3 #s
        # overlap= 0.2
        # fftsize=16000
        # fmin=10
        # fmax=120      
        npix=100
        
        t1=np.arange(0,reclength-window,window*overlap)
        t2=t1+window
        
        # t1_datetime= starttime + pd.to_timedelta(t1,'s')
        # t2_datetime= starttime + pd.to_timedelta(t2,'s')
        
        x=p[  int(fs*t1[0]) : int(fs*t2[0])]   
        f, t, Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fftsize,noverlap=int(fftsize*0.9))
        ixf = (f>= fmin) & (f<fmax)
        db=10*np.log10(Sxx)
        ixf = (f>= 10) & (f<200)
        
        db = resize( db[ixf,:], [npix,npix])        

        # plt.figure(8)
        # plt.clf()
        # plt.subplot(311)
        # plt.imshow(db,aspect='auto')
        # plt.subplot(312)
        # plt.imshow(db_m,aspect='auto')
        # plt.subplot(313)
        # plt.imshow(db-db_m ,aspect='auto')
        
        x_train_sg=np.empty( [len(t1) ,    npix,npix,1])
        
        for ix in range(len(t1)):
            
            x=p[  int(fs*t1[ix]) : int(fs*t2[ix])]
            f, t, Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fftsize,noverlap=int(fftsize*0.9))
            db=10*np.log10(Sxx)        
            ixf = (f>= fmin) & (f<fmax)
            
            db = resize( db[ixf,:], [npix,npix])        
            db_m=  np.transpose( np.tile( np.mean(db,axis=1) ,(db.shape[1],1) ) )
            db=db-db_m 
     
            # if np.shape(db) is not  np.shape(x_train_sg)[1:2]:
            #     db = resize( db, [np.shape(x_train_sg)[1],np.shape(x_train_sg)[2]])        
            x_train_sg[ix,:,:,0]= (db  - db.min() ) /(db.max() - db.min())
        
        
        # os.mkdir('training_fw')
        h5f = h5py.File(  os.path.basename(name)+'_sg.h5', 'w')    
        h5f.create_dataset('x_train_sg', data=x_train_sg)
        h5f.create_dataset('t1', data=t1)
        h5f.create_dataset('t2', data=t2)
        h5f.create_dataset('window', data=window)
        h5f.create_dataset('overlap', data=overlap)
        h5f.create_dataset('fftsize', data=fftsize)
        h5f.create_dataset('fmin', data=fmin)
        h5f.create_dataset('fmax', data=fmax)
        h5f.create_dataset('npix', data=npix)
        h5f.close()
        


#%%

if __name__ == '__main__':
    

    audio_folder=r'F:\cod_sound_analysis\2018\WA_hydrophones\tutorial\*.wav'
    audiopath_list=glob.glob(audio_folder)
    print(audiopath_list)
        
    cpucounts=multiprocessing.cpu_count()
    
    print(cpucounts)
    
    pool = multiprocessing.Pool(processes=cpucounts)
    index_list=range(len( audiopath_list ))
    pool.map( partial( parafunc,audiopath_list,1,0.2,15000,10,200), index_list)
    pool.close   
    